"""bpy-скрипт: сцена з tier_b_traj.json — дерева, дрон-еліпсоїд, 34 LIDAR-промені,
chase-камера. Blender 5.1-safe (перевірені граблі попередньої сесії):
  • НЕ чіпаємо Action.fcurves/interpolation (layered actions у 5.x);
  • file_format без "FFMPEG" — рендеримо PNG-послідовність, mp4 збирає ffmpeg.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tier_b/blender_scene.py -- \
        tier_b/exports/tier_b_traj.json tier_b/exports/blender_out \
        [--render] [--every 2] [--no-rays]

--no-rays: без LIDAR-променів — лише дрон + ліс (найчистіша анімація;
дані променів у JSON лишаються, це суто рендер-рішення).

Кватерніони у JSON — WXYZ (конвертовано з pybullet XYZW в export_blender.py),
точно формат mathutils.Quaternion.
"""

import json
import pathlib
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
TRAJ = pathlib.Path(argv[0]) if argv else pathlib.Path("tier_b/exports/tier_b_traj.json")
OUT = pathlib.Path(argv[1]) if len(argv) > 1 else TRAJ.parent / "blender_out"
RENDER = "--render" in argv
EVERY = int(argv[argv.index("--every") + 1]) if "--every" in argv else 1
NO_RAYS = "--no-rays" in argv


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name, color, emission=0.0, alpha=1.0, use_object_color=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    if use_object_color:
        info = mat.node_tree.nodes.new("ShaderNodeObjectInfo")
        mat.node_tree.links.new(info.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.blend_method = "BLEND"
    return mat


def main():
    data = json.loads(TRAJ.read_text())
    fps = data["fps"]
    frames = data["frames"][::EVERY]

    clear_scene()
    scene = bpy.context.scene
    scene.render.fps = max(1, fps // EVERY)
    scene.frame_start = 1
    scene.frame_end = len(frames)

    # земля
    bpy.ops.mesh.primitive_plane_add(size=40, location=(6, 0, 0))
    ground = bpy.context.object
    ground.data.materials.append(make_material("ground", (0.12, 0.2, 0.08)))

    # дерева
    bark = make_material("bark", (0.28, 0.18, 0.09))
    for t in data["trees"]:
        bpy.ops.mesh.primitive_cylinder_add(radius=t["r"], depth=t["h"],
                                            location=(t["x"], t["y"], t["h"] / 2))
        bpy.context.object.data.materials.append(bark)

    # дрон = еліпсоїд (масштабована сфера, ті самі півосі, що в колізії)
    a, b, c = data["ellipsoid"]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
    drone = bpy.context.object
    drone.name = "drone"
    drone.scale = (a, b, c)
    drone.rotation_mode = "QUATERNION"
    drone.data.materials.append(make_material("drone", (0.85, 0.85, 0.9)))

    # LIDAR-промені: 34 тонкі циліндри; колір через Object Info → obj.color
    rays = []
    if not NO_RAYS:
        ray_mat = make_material("ray", (1, 1, 1), use_object_color=True)
        for k in range(len(frames[0]["lidar"])):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.004, depth=1.0, vertices=6)
            r = bpy.context.object
            r.name = f"ray_{k:02d}"
            r.rotation_mode = "QUATERNION"
            r.data.materials.append(ray_mat)
            rays.append(r)

    # камера + сонце
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    scene.camera = cam
    bpy.ops.object.light_add(type="SUN", location=(6, -6, 10))
    bpy.context.object.data.energy = 3.0

    from mathutils import Quaternion, Vector

    z_axis = Vector((0, 0, 1))
    for fi, fr in enumerate(frames, start=1):
        pos = Vector(fr["pos"])
        drone.location = pos
        drone.rotation_quaternion = Quaternion(fr["quat_wxyz"])
        drone.keyframe_insert("location", frame=fi)
        drone.keyframe_insert("rotation_quaternion", frame=fi)

        for k, ray in enumerate(rays):
            seg = fr["lidar"][k]
            end = Vector(seg["to"])
            mid = (pos + end) / 2
            d = end - pos
            length = max(d.length, 1e-6)
            ray.location = mid
            ray.rotation_quaternion = z_axis.rotation_difference(d.normalized())
            ray.scale = (1.0, 1.0, length)
            ray.color = (1.0, 0.1, 0.1, 1.0) if seg["hit"] else (0.6, 0.6, 0.6, 0.25)
            ray.keyframe_insert("location", frame=fi)
            ray.keyframe_insert("rotation_quaternion", frame=fi)
            ray.keyframe_insert("scale", frame=fi)
            ray.keyframe_insert("color", frame=fi)

        cam.location = pos + Vector((-1.6, -1.4, 0.9))
        direction = pos - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=fi)
        cam.keyframe_insert("rotation_euler", frame=fi)

    OUT.mkdir(parents=True, exist_ok=True)
    blend = OUT / "tier_b_pirouette.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    print(f"збережено {blend}")

    if RENDER:
        scene.render.image_settings.file_format = "PNG"  # без "FFMPEG" у 5.1
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 720
        scene.render.filepath = str(OUT / "frame_")
        bpy.ops.render.render(animation=True)
        print(f"PNG-кадри: {OUT}/frame_####.png — mp4 збирайте ffmpeg-ом:")
        print(f"  ffmpeg -framerate {scene.render.fps} -i '{OUT}/frame_%04d.png' "
              f"-c:v libx264 -pix_fmt yuv420p {OUT}/tier_b_pirouette.mp4")


main()
