# -*- coding: utf-8 -*-
"""
Анімація дрона в Blender із JSON-траєкторії рушія (src/drone_sim.py).

Скрипт читає out/<сценарій>.json і покадрово виставляє location та
rotation_quaternion об'єкта "Drone" (створює простий меш, якщо такого
об'єкта немає). Додає дві камери: FPV (на корпусі) та зовнішню (стежить
за дроном). Системи координат збігаються: і рушій, і Blender —
правосторонні, Z вгору; кватерніон [w,x,y,z] відповідає порядку
rotation_quaternion, тому жодних перетворень не потрібно.

ЯК ЗАПУСТИТИ
  1) З терміналу (відкриє GUI з готовою анімацією):
       /Applications/Blender.app/Contents/MacOS/Blender -P blender/animate_drone.py -- out/square.json
  2) Headless, зі збереженням .blend:
       .../Blender -b -P blender/animate_drone.py -- out/square.json --fps 30 --save out/square.blend
  3) Зсередини Blender: Scripting → відкрити цей файл → Run Script
     (без аргументів візьме out/square.json відносно кореня проєкту).

АРГУМЕНТИ (після "--")
  <json>        шлях до траєкторії (типово out/square.json)
  --fps N       fps сцени (типово 30); сим-кадри субдискретизуються по часу
  --save PATH   зберегти .blend у файл (для headless-режиму)

Свою 3D-модель дрона: назвіть її об'єкт "Drone" перед запуском скрипта
(або зробіть модель дочірньою до створеного "Drone") — анімація ляже на неї.
"""

import json
import math
import sys
from pathlib import Path

import bpy

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    """Аргументи скрипта — все, що йде після '--' у командному рядку Blender."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    json_path, fps, save, cam = PROJECT_ROOT / "out" / "square.json", 30, None, "chase"
    i = 0
    while i < len(argv):
        if argv[i] == "--fps":
            fps = int(argv[i + 1]); i += 2
        elif argv[i] == "--save":
            save = Path(argv[i + 1]); i += 2
        elif argv[i] == "--cam":
            cam = argv[i + 1]; i += 2       # chase | fpv
        else:
            json_path = Path(argv[i]); i += 1
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path
    return json_path, fps, save, cam


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
    return mat


def setup_world_and_light():
    """Без світла headless-рендер дає порожній сірий кадр — додаємо сонце й небо."""
    world = bpy.data.worlds["World"] if bpy.data.worlds else bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.45, 0.62, 0.85, 1.0)  # небо
    bg.inputs["Strength"].default_value = 1.0

    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))


def clear_scene():
    """Порожня сцена: видаляємо ВСІ об'єкти НАПРЯМУ через API (не select+delete).

    Через select+delete лишалися об'єкти з hide_select=True (їх вмикає «ігровий»
    режим) — тож при ПЕРЕГЕНЕРАЦІЇ карти старі дерева/фури накопичувались, а нові
    отримували імена CP_0_zone.001 і т. п. (баг). API-видалення прибирає все
    незалежно від виділення/блокування. Плюс чистимо осиротілі меші/криві, щоб
    пам'ять не текла при багатьох перегенераціях."""
    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:   # noqa: BLE001
            pass
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)


def make_drone(arm=0.17):
    """Готова текстурована FPV-рама (assets/drone_source.glb — «FPV Drone»,
    kamikaze strike drone): рама, 4 гвинти, акумулятор, антена — напрямок носа
    видно із самої моделі, окремий маркер-конус більше не потрібен (на відміну
    від старого симетричного корпусу з примітивів). Якщо об'єкт 'Drone' уже є
    в сцені (ваша модель) — використовуємо його. Параметр arm лишається для
    сумісності виклику (масштаб/орієнтація вже приведені офлайн-скриптом,
    ніс — вздовж +X)."""
    if "Drone" in bpy.data.objects:
        return bpy.data.objects["Drone"]

    from . import mesh_models
    drone = mesh_models.spawn("drone_source.glb", "Drone")
    return drone


def add_cameras(drone):
    """FPV-камера жорстко на корпусі (успадковує кватерніон дрона) +
    зовнішня камера зі стеженням (constraint TRACK_TO)."""
    # FPV: камера дивиться вздовж −Z власної осі з +Y вгору, тому щоб дивитись
    # уперед по +X дрона з горизонтом по +Z — поворот Ейлера (90°, 0°, −90°)
    cam_data = bpy.data.cameras.new("FPV")
    cam_data.clip_start = 0.01
    cam_data.lens = 18.0                      # ширококутна, як справжня FPV
    fpv = bpy.data.objects.new("FPV", cam_data)
    bpy.context.collection.objects.link(fpv)
    fpv.parent = drone
    fpv.location = (0.20, 0.0, 0.05)          # попереду носового конуса, щоб корпус не закривав кадр
    fpv.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)

    cam_data2 = bpy.data.cameras.new("Chase")
    cam_data2.lens = 50.0
    cam_data2.clip_start = 0.05
    chase = bpy.data.objects.new("Chase", cam_data2)
    bpy.context.collection.objects.link(chase)
    # Жорстко кріпимо ззаду/зверху дрона (як FPV, але зовнішня точка зору).
    # Offset: -3.5 вздовж осі дрона (ззаду), +1.5 вгору.
    # Кут: дивиться в бік носа (+X) з нахилом вниз ~23.2° — щоб дрон був у кадрі.
    chase.location = (-3.5, 0.0, 1.5)
    # Кут: від позиції (-3.5, 0, 1.5) дивитись на (0,0,0) — центр дрона.
    tilt = math.atan2(1.5, 3.5)          # ~23.2° — вертикальний зсув / горизонталь
    chase.rotation_euler = (math.pi / 2 - tilt, 0.0, -math.pi / 2)

    return fpv, chase


def add_ground():
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    checker = nodes.new("ShaderNodeTexChecker")   # шахівниця — видно рух над землею
    checker.inputs["Scale"].default_value = 20.0
    checker.inputs["Color1"].default_value = (0.55, 0.55, 0.55, 1.0)
    checker.inputs["Color2"].default_value = (0.35, 0.40, 0.35, 1.0)
    links.new(checker.outputs["Color"],
              nodes["Principled BSDF"].inputs["Base Color"])
    ground.data.materials.append(mat)


def add_tree(name, location, trunk_h=1.0, crown_r=0.55):
    """Просте дерево: стовбур-циліндр + крона-ікосфера. z=0 — на підлозі."""
    x, y = location
    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=trunk_h,
                                        location=(x, y, trunk_h / 2))
    trunk = bpy.context.object
    trunk.name = f"{name}_Trunk"
    trunk.data.materials.append(
        bpy.data.materials.get("TrunkMat")
        or make_material("TrunkMat", (0.28, 0.18, 0.10, 1.0)))

    bpy.ops.mesh.primitive_ico_sphere_add(radius=crown_r, subdivisions=2,
                                          location=(x, y, trunk_h + crown_r * 0.7))
    crown = bpy.context.object
    crown.name = f"{name}_Crown"
    crown.data.materials.append(
        bpy.data.materials.get("CrownMat")
        or make_material("CrownMat", (0.10, 0.35, 0.12, 1.0)))


def add_trees():
    """Кілька дерев навколо зони польоту (квадрат 0..2 м) — масштаб і глибина кадру."""
    for i, (x, y, h, r) in enumerate((
            (-1.6,  3.1, 1.1, 0.60),
            ( 3.6, -1.3, 0.9, 0.50),
            ( 4.1,  3.2, 1.3, 0.70),
            (-2.1, -1.6, 1.0, 0.55))):
        add_tree(f"Tree{i}", (x, y), trunk_h=h, crown_r=r)


def add_building():
    """Будівля-коробка збоку від зони польоту — орієнтир для FPV та глибина
    сцени. Позиція (8, −4): сонце (азимут 30°) кидає її тінь у бік −X−Y,
    ГЕТЬ від польотного квадрата 0..2 м — інакше тінь накриває старт."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(8.0, -4.0, 1.4))
    bld = bpy.context.object
    bld.name = "Building"
    bld.scale = (3.0, 2.0, 2.8)
    bld.data.materials.append(make_material("BuildingMat", (0.55, 0.30, 0.20, 1.0)))

    # Плаский дах трохи ширший за стіни, щоб силует читався
    bpy.ops.mesh.primitive_cube_add(size=1, location=(8.0, -4.0, 2.86))
    roof = bpy.context.object
    roof.name = "BuildingRoof"
    roof.scale = (3.2, 2.2, 0.12)
    roof.data.materials.append(make_material("RoofMat", (0.30, 0.28, 0.26, 1.0)))


def add_helipad():
    """Посадковий майданчик у точці старту/посадки (0,0): темний диск,
    біле кільце та літера "H". Трохи над підлогою, щоб не було z-fighting."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=0.02, location=(0, 0, 0.01))
    pad = bpy.context.object
    pad.name = "Helipad"
    pad.data.materials.append(make_material("HelipadMat", (0.12, 0.12, 0.14, 1.0)))

    white = make_material("HelipadMark", (0.92, 0.92, 0.92, 1.0))

    bpy.ops.mesh.primitive_torus_add(major_radius=0.42, minor_radius=0.015,
                                     location=(0, 0, 0.022))
    ring = bpy.context.object
    ring.name = "HelipadRing"
    ring.data.materials.append(white)

    bpy.ops.object.text_add(location=(0, 0, 0.022))
    h = bpy.context.object
    h.name = "HelipadH"
    h.data.body = "H"
    h.data.align_x, h.data.align_y = "CENTER", "CENTER"
    h.data.size = 0.5
    h.data.extrude = 0.005
    h.data.materials.append(white)


def animate(drone, frames, fps_sim, fps_scene):
    """Покадрова анімація: для кожного кадру сцени беремо сим-кадр,
    найближчий за часом (субдискретизація fps_sim → fps_scene)."""
    drone.rotation_mode = "QUATERNION"
    t_end = frames[-1]["t"]
    n_scene = int(t_end * fps_scene) + 1

    for f_scene in range(n_scene):
        t = f_scene / fps_scene
        idx = min(round(t * fps_sim), len(frames) - 1)
        fr = frames[idx]
        drone.location = fr["pos"]
        drone.rotation_quaternion = fr["quat"]        # порядок [w,x,y,z] збігається
        frame = f_scene + 1                            # кадри Blender — з 1
        drone.keyframe_insert("location", frame=frame)
        drone.keyframe_insert("rotation_quaternion", frame=frame)

    scene = bpy.context.scene
    scene.render.fps = fps_scene
    scene.frame_start, scene.frame_end = 1, n_scene
    scene.frame_set(1)
    return n_scene


def main():
    json_path, fps_scene, save, cam = parse_args()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta, frames = data["meta"], data["frames"]

    clear_scene()
    setup_world_and_light()
    add_ground()
    add_trees()
    add_building()
    add_helipad()
    drone = make_drone()
    fpv, chase = add_cameras(drone)
    bpy.context.scene.camera = fpv if cam == "fpv" else chase
    n = animate(drone, frames, meta["fps"], fps_scene)

    print(f"Сценарій '{meta['scenario']}': {len(frames)} сим-кадрів "
          f"({meta['fps']:.0f} Гц) → {n} кадрів сцени @ {fps_scene} fps "
          f"({n / fps_scene:.1f} с). Активна камера: {cam}.")

    if save is not None:
        if not save.is_absolute():
            save = PROJECT_ROOT / save
        bpy.ops.wm.save_as_mainfile(filepath=str(save))
        print(f"Збережено: {save}")


if __name__ == "__main__":
    main()
