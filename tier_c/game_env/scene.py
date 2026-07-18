# -*- coding: utf-8 -*-
"""
Побудова 3D-середовища коміту 2e2bc5d (дерева / фури-чекпоінти / тематичні перешкоди).

Код перенесено ДОСЛІВНО з core/simulator.py коміту 2e2bc5d (частина build_scene:
_build_trees / _build_trucks / _build_obstacles / _zone_material), лише відв'язано від
старої фізики — тут будується САМЕ ВІЗУАЛЬНЕ СЕРЕДОВИЩЕ, без рушія/лідара/скорингу.
`self.map/self.cfg/self.terrain` замінено на параметри (md, cfg, terrain).
"""

import math

import bpy
import numpy as np
from mathutils import Vector

from . import animate_drone as scene
from .generator import wreck_yaw

# Частка нормалізованої висоти дерева (0..1), на яку заглиблюємо стовбур у
# рельєф — коренева розтяжка меша лежить у цьому діапазоні знизу; без цього
# вона лежала б ПОВЕРХ землі, а не виростала з неї.
TREE_ROOT_EMBED = 0.15


_FIT_MAX_TILT_DEG = 30.0   # запобіжник — див. докстрінг _fit_ground_plane


def _fit_ground_plane(terrain, cx, cy, yaw, half_len, half_wid):
    """Підібрати площину-опору під ВЕЛИКИЙ плоскодонний об'єкт (фура ~8 м,
    уламок ~5-7 м) — не через terrain.normal_at() в ОДНІЙ точці (той рахує
    нахил кроком ~0.79 м, на порядок менше за сам об'єкт, тож ігнорує, як
    рельєф змінюється аж до фактичних кутків підошви), а через МНК-площину
    крізь СІТКУ 3×3 точок рельєфу під підошвою (з урахуванням курсу yaw).

    РІВНО 4 кутки (як у першій версії) виявились НЕДОСТАТНІМИ: рельєф — fBm із
    4 октавами шуму, тож на масштабі одного 6-8-метрового об'єкта трапляються
    локальні горби/ямки МІЖ кутками; усього 4 точки для МНК-площини (3
    невідомих) майже не мають надлишку — один випадковий викид у ОДНІЙ точці
    різко перекошував нахил і на ПРОТИЛЕЖНОМУ кутку екстраполяція давала
    зазор >1 м (перевірено й відтворено). Сітка 3×3 (9 точок, явний надлишок)
    згладжує поодинокі викиди. Плюс жорсткий обмежувач нахилу (_FIT_MAX_TILT_DEG)
    про всяк випадок — якщо навіть 9-точкова МНК-площина вийде неправдоподібно
    крутою, краще занизити нахил, ніж дозволити кутку провалитись під землю.

    Повертає (z_center, normal) — висоту опорної площини в центрі й нормаль
    для нахилу об'єкта."""
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    fracs = (-1.0, 0.0, 1.0)
    pts = []
    for flen in fracs:
        for fwid in fracs:
            lx, ly = flen * half_len, fwid * half_wid
            wx = cx + lx * cyaw - ly * syaw
            wy = cy + lx * syaw + ly * cyaw
            pts.append((wx, wy, terrain.height_at(wx, wy)))
    A = np.array([[p[0], p[1], 1.0] for p in pts])
    b_col = np.array([p[2] for p in pts])
    a, b, c = np.linalg.lstsq(A, b_col, rcond=None)[0]
    z_center = a * cx + b * cy + c
    n = np.array([-a, -b, 1.0])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return terrain.height_at(cx, cy), terrain.normal_at(cx, cy)
    normal = n / norm

    max_tilt_cos = math.cos(math.radians(_FIT_MAX_TILT_DEG))
    if normal[2] < max_tilt_cos:
        xy_norm = math.hypot(normal[0], normal[1])
        if xy_norm > 1e-9:
            target_xy = math.sqrt(max(0.0, 1.0 - max_tilt_cos ** 2))
            scale = target_xy / xy_norm
            normal = np.array([normal[0] * scale, normal[1] * scale, max_tilt_cos])

    return float(z_center), tuple(normal)


def _zone_material():
    """Матеріал напівпрозорого циліндра зарахування («маяк»)."""
    mat = bpy.data.materials.get("CheckpointZoneMat")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("CheckpointZoneMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1.0, 0.82, 0.1, 1.0)
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.22
    if "Emission Color" in bsdf.inputs:      # світіння — щоб «маяк» читався
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.7, 0.05, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 1.2
    for attr, val in (("blend_method", "BLEND"),
                      ("surface_render_method", "BLENDED")):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, val)
            except Exception:   # noqa: BLE001
                pass
    return mat


def _build_trees(md, cfg, terrain):
    """Дерева з готової STL-моделі (гола крона-гілки, assets/tree_source.stl),
    вирівняні за нормаллю рельєфу; префікс TREE_ → колізійні. Меш нормалізовано
    (одноразовим офлайн-скриптом) до висоти 1.0 м із підошвою на Z=0 — на місці
    лише рівномірно масштабуємо під випадкову висоту h конкретного дерева.

    РІВНО ОДНЕ дерево (md.wreck_index/md.wreck_kind, обрані генератором) —
    заміняємо тут лише візуально на текстурований GLB. Колізія
    (collision_and_bounds_status) рахує уламок ОКРЕМИМ орієнтованим боксом із
    OBJECT_FOOTPRINTS (generator.py), не по (radius, height) із цього циклу."""
    from . import stl_models, mesh_models
    from .generator import OBJECT_FOOTPRINTS, AVTOZAK_MESH_SCALE
    mat = (bpy.data.materials.get("TrunkMat")
           or scene.make_material("TrunkMat", (0.30, 0.20, 0.11, 1.0)))
    verts, faces = stl_models.get_tree_mesh()
    wreck_index = getattr(md, "wreck_index", -1)
    wreck_kind = getattr(md, "wreck_kind", "")
    for i, (x, y, z_base, r, h) in enumerate(md.trees):
        if i == wreck_index and wreck_kind:
            # Детермінований псевдовипадковий курс (generator.wreck_yaw — та сама
            # формула, що й колізійний бокс уламка, щоб хітбокс завжди збігався
            # з тим, що видно на екрані).
            yaw = wreck_yaw(x, y)
            # МНК-площина під РЕАЛЬНИМИ кутками підошви — точні виміряні
            # half_len/half_wid із OBJECT_FOOTPRINTS (а не груба апроксимація
            # r/r*0.42) — без цього плоскодонний уламок 5-7 м торкався б землі
            # лише в одній точці центру, а протилежний кут «висів» би в повітрі
            # на пагористому рельєфі.
            fp = OBJECT_FOOTPRINTS.get(wreck_kind)
            half_len = fp["half_len"] if fp else r
            half_wid = fp["half_wid"] if fp else r * 0.42
            z_fit, n_fit = _fit_ground_plane(terrain, x, y, yaw,
                                             half_len=half_len, half_wid=half_wid)
            # Автозак — сира геометрія занижена (offline-підготовка не
            # довела її до реального розміру); виправляємо тут неоднорідним
            # масштабом (OBJECT_FOOTPRINTS/WRECK_DIMS для нього вже рахують
            # ЦЕЙ масштаб). tank/dshk — без корекції, вже реалістичні.
            mesh_scale = AVTOZAK_MESH_SCALE if wreck_kind == "wreck_avtozak" else 1.0
            mesh_models.spawn(f"{wreck_kind}_source.glb", f"TREE_{i:03d}",
                              location=(x, y, z_fit), rotation_z=yaw,
                              scale=mesh_scale, align_normal=n_fit)
            continue
        n = Vector(terrain.normal_at(x, y))
        mesh = bpy.data.meshes.new(f"TreeMesh_{i}")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        for poly in mesh.polygons:
            poly.use_smooth = True
        trunk = bpy.data.objects.new(f"TREE_{i:03d}", mesh)
        bpy.context.collection.objects.link(trunk)
        # Коренева розтяжка меша займає низ (0..~0.15 з нормалізованої висоти 1.0) —
        # без заглиблення вона лежала б ПОВЕРХ рельєфу, а не виростала з нього.
        # Заглиблюємо вздовж нормалі (не просто по світовій Z) — коректно й на схилі.
        trunk.location = Vector((x, y, z_base)) - n * (TREE_ROOT_EMBED * h)
        trunk.rotation_mode = "QUATERNION"
        trunk.rotation_quaternion = n.to_track_quat("Z", "Y")  # локальна Z → нормаль
        trunk.scale = (h, h, h)
        trunk.data.materials.append(mat)


def _build_trucks(md, cfg, terrain):
    """Чекпоінти — текстурована фура-цистерна (assets/checkpoint_truck_source.glb)
    + напівпрозорий циліндр-проліт. Меш нормалізовано (одноразовим офлайн-скриптом)
    до довжини ~5.0 м уздовж X із підошвою на Z=0 (TRUCK_FOOTPRINT — реальні
    виміряні розміри, generator.py)."""
    from . import mesh_models
    from .generator import TRUCK_FOOTPRINT
    zone_mat = _zone_material()

    for i, (cx, cy, cz) in enumerate(md.checkpoints):
        # МНК-площина під РЕАЛЬНИМИ кутками підошви фури — навколо ФАКТИЧНОГО
        # геометричного центру меша (TRUCK_FOOTPRINT["center"], зсунутого від
        # точки прив'язки (cx,cy) — сам меш не центрований на власному
        # початку координат), а не точки прив'язки напряму. Фура не має
        # курсу (yaw=0), тож кутки лежать точно вздовж світових X/Y.
        lcx, lcy = TRUCK_FOOTPRINT["center"]
        z_fit, n_fit = _fit_ground_plane(terrain, cx + lcx, cy + lcy, 0.0,
                                         half_len=TRUCK_FOOTPRINT["half_len"],
                                         half_wid=TRUCK_FOOTPRINT["half_wid"])
        mesh_models.spawn("checkpoint_truck_source.glb", f"CP_{i}_truck",
                          location=(cx, cy, z_fit), align_normal=n_fit)

        bpy.ops.mesh.primitive_cylinder_add(
            radius=cfg.cp_cyl_radius, depth=cfg.cp_cyl_height, location=(cx, cy, cz))
        zone = bpy.context.object
        zone.name = f"CP_{i}_zone"
        zone.data.materials.append(zone_mat)


def _build_obstacles(md, cfg, terrain):
    """Тематичні перешкоди: Ка-52 (виключно в небі), пес Патрон."""
    from . import mesh_models
    for i, (kind, x, y, z, r, collidable) in enumerate(md.obstacles):
        name = f"{'OBST' if collidable else 'DECOR'}_{i}_{kind}"
        if kind == "ka52":
            # Детермінований псевдовипадковий курс (лише візуальна різноманітність,
            # не впливає на колізію — та лишається сферою радіуса r з генератора).
            yaw = (x * 12.9898 + y * 78.233) % (2.0 * math.pi)
            mesh_models.spawn("ka_52.glb", name,
                              location=(x, y, z), rotation_z=yaw)
        elif kind == "patron":
            from . import patron_model
            verts, faces = patron_model.get_patron_mesh()
            mesh = bpy.data.meshes.new("PatronMesh")
            mesh.from_pydata(verts, [], faces)
            mesh.update()
            for poly in mesh.polygons:
                poly.use_smooth = True
            body = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(body)
            body.location = (x, y, terrain.height_at(x, y))
            mat = (bpy.data.materials.get("PatronStatueMat")
                   or scene.make_material("PatronStatueMat", (0.66, 0.63, 0.58, 1.0)))
            body.data.materials.append(mat)


# ── Дощ (візуал) ─────────────────────────────────────────────────────────────
# Гра рухає об'єкти таймером, а НЕ кадрами (frame_set лише раз), тому частки з
# фізикою не падали б. Робимо дощ власним bpy-таймером — так само, як анімується
# дрон. Суто косметика: усе в try/except, щоб дощ НІКОЛИ не ламав гру.
RAIN_ENABLED = True
RAIN_COUNT = 450          # к-ть крапель-стріків
RAIN_HEIGHT = 16.0        # висота, з якої падають, м
RAIN_FALL_SPEED = 12.0    # швидкість падіння, м/с

_RAIN = {"drops": []}


def _build_rain(cfg):
    """Створити краплі й запустити таймер падіння. Frame-незалежно."""
    if not RAIN_ENABLED:
        return
    try:
        import random
        b = float(cfg.bounds)
        mat = scene.make_material("RainMat", (0.55, 0.66, 0.85, 1.0))

        bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.40,
                                            location=(0.0, 0.0, -999.0))
        proto = bpy.context.active_object
        proto.name = "RainProto"
        proto.data.materials.append(mat)

        drops = []
        for _ in range(RAIN_COUNT):
            d = proto.copy()                       # linked-mesh дублікат — дешево
            d.location = (random.uniform(-b, b), random.uniform(-b, b),
                          random.uniform(0.0, RAIN_HEIGHT))
            bpy.context.collection.objects.link(d)
            drops.append(d)
        proto.hide_set(True)                       # ховаємо шаблон (копії вже зроблено)
        _RAIN["drops"] = drops

        step = RAIN_FALL_SPEED / 30.0

        def _fall():
            try:
                for d in _RAIN["drops"]:
                    z = d.location.z - step
                    if z < 0.0:                    # долетіла до землі — на початок
                        z = RAIN_HEIGHT
                        d.location.x = random.uniform(-b, b)
                        d.location.y = random.uniform(-b, b)
                    d.location.z = z
            except ReferenceError:
                return None                        # сцену перебудували — цей таймер завершується
            return 1.0 / 30.0

        bpy.app.timers.register(_fall, first_interval=0.5)
    except Exception as exc:                        # noqa: BLE001 — дощ не має ламати сцену
        print("Дощ: не вдалося створити —", exc)


def _build_mission_markers(md, cfg, terrain):
    """Маркери місії: аптечка (червоний куб з хрестом) на предметі + людина на цілі."""
    try:
        import solution                            # верхнього рівня (tier_c на sys.path)
        # зелена платформа зарядки (окрема точка)
        cx, cy = solution.mission_charge(md, cfg)
        cz = terrain.height_at(cx, cy)
        green = scene.make_material("ChargePadMat", (0.15, 0.95, 0.35, 1.0))
        bpy.ops.mesh.primitive_cylinder_add(radius=1.2, depth=0.15, location=(cx, cy, cz + 0.08))
        pad = bpy.context.active_object
        pad.name = "ChargePad"
        pad.data.materials.append(green)
        # аптечка на точці предмета
        sx, sy = solution.mission_pickup(md, cfg)
        sz = terrain.height_at(sx, sy)
        red = scene.make_material("MedkitMat", (0.90, 0.12, 0.12, 1.0))
        white = scene.make_material("MedkitCross", (0.97, 0.97, 0.97, 1.0))
        bpy.ops.mesh.primitive_cube_add(size=0.6, location=(sx, sy, sz + 0.3))
        box = bpy.context.active_object
        box.name = "Medkit"
        box.data.materials.append(red)
        bpy.ops.mesh.primitive_cube_add(size=0.62, location=(sx, sy, sz + 0.42))
        bar = bpy.context.active_object
        bar.name = "MedkitCrossH"
        bar.scale = (1.0, 0.28, 0.18)
        bar.data.materials.append(white)

        # «людина» на цілі-фурі (синій стовпчик)
        gx, gy = float(md.checkpoints[0][0]), float(md.checkpoints[0][1])
        gz = terrain.height_at(gx, gy)
        blue = scene.make_material("PersonMat", (0.20, 0.45, 0.95, 1.0))
        bpy.ops.mesh.primitive_cylinder_add(radius=0.28, depth=1.7,
                                            location=(gx + 1.5, gy, gz + 0.85))
        person = bpy.context.active_object
        person.name = "Person"
        person.data.materials.append(blue)
    except Exception as exc:                        # noqa: BLE001 — маркери не мають ламати сцену
        print("Маркери місії: не створено —", exc)


def build_environment(md, cfg, terrain):
    """Повне середовище 2e2bc5d: рельєф-меш + ліс + фури-чекпоінти + тематичні перешкоди.
    (Світло/небо й дрон/камери — окремо через animate_drone у launcher'і.)"""
    terrain.build_mesh()
    _build_trees(md, cfg, terrain)
    _build_trucks(md, cfg, terrain)
    _build_obstacles(md, cfg, terrain)
    _build_mission_markers(md, cfg, terrain)       # аптечка на предметі + людина на цілі
    _build_rain(cfg)                               # ← дощ останнім: збій не зачепить решту
