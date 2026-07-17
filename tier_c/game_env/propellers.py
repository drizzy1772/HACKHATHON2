# -*- coding: utf-8 -*-
"""
Анімація гвинтів — і на дроні, і на «Потужнольоті» — швидкість обертання
залежить від газу (throttle). Обидва джерельні меші (assets/drone_source.glb,
assets/potuznolit_source.glb) — фрагментована AI-реконструкція («Tripo»): дрон ~15
тис. окремих острівців, а в літака навіть візуально ізольований гвинт на носі
виявився складеним із кількох майже накладених копій лопатей (вирізання
bmesh'ом за формою острівця залишало видиму лопать позаду — перевірено
рендером). Тому для ОБОХ апаратів гвинт не вирізається з наявного меша — це
надто ризиковано на такій геометрії, — а ДОДАЄТЬСЯ окремим простим
дволопатевим примітивом поверх існуючого носа/моторів: статична модельована
лопать лишається на місці, а нова, що крутиться, лежить точно поверх неї —
на робочих оборотах читається як звичайний розмитий диск гвинта.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Quaternion

# Локальні (x, y) 4 моторів дрона + висота лопатей (z) — виміряно офлайн
# bmesh-аналізом assets/drone_source.glb (верхівки лопатей близько до рами,
# НЕ на рівні акумулятора, який сидить вище по центру дрона).
DRONE_MOTOR_XY = ((0.128, -0.107), (-0.111, -0.107), (0.128, 0.107), (-0.111, 0.107))
DRONE_PROP_Z = 0.02
DRONE_PROP_RADIUS = 0.09
# Діагональні пари крутяться в протилежні боки — компенсація реактивного
# моменту, як у справжнього квадрокоптера (суто візуальна деталь).
DRONE_PROP_SPIN_DIRS = (1, -1, -1, 1)

# Ніс «Потужнольоту» (assets/potuznolit_source.glb, СИРИЙ масштаб — той самий, що й
# у дрона, до PLANE_SCALE) — виявлено рендером (реальний 2-лопатевий гвинт).
PLANE_PROP_CENTER = (0.19, 0.0, 0.0)
PLANE_PROP_RADIUS = 0.08

_PROP_PREFIX = "PROP_"
_spin_angle = {}   # obj.name -> накопичений кут (рад); скидається при (пере)створенні гвинтів


def _make_blade_mesh(radius, width):
    """Плоска дволопатева «хрестовина» в локальній XY-площині — на швидкому
    обертанні читається як розмитий диск гвинта без складної геометрії лопаті."""
    r, w = radius, width / 2.0
    verts = [
        (-r, -w, 0.0), (r, -w, 0.0), (r, w, 0.0), (-r, w, 0.0),
        (-w, -r, 0.0), (w, -r, 0.0), (w, r, 0.0), (-w, r, 0.0),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    mesh = bpy.data.meshes.new("PropBladeMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _prop_material():
    mat = bpy.data.materials.get("PropMat")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("PropMat")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.04, 0.04, 0.05, 1.0)
    return mat


def _spawn_drone_propellers(drone):
    mesh = _make_blade_mesh(DRONE_PROP_RADIUS, DRONE_PROP_RADIUS * 0.12)
    mat = _prop_material()
    for i, (mx, my) in enumerate(DRONE_MOTOR_XY):
        name = f"{_PROP_PREFIX}{i}"
        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(mat)
        bpy.context.collection.objects.link(obj)
        obj.parent = drone
        obj.location = (mx, my, DRONE_PROP_Z)
        _spin_angle[name] = 0.0


def _plane_prop_quat(spin_angle):
    """Нахил (нормаль диска Z → уздовж носа +X, ФІКСОВАНИЙ) СКЛАДЕНИЙ ІЗ
    обертанням (у ВЛАСНІЙ, ще не нахиленій площині диска) — порядок важливий:
    спершу крутимо лопать у її рідній XY-площині (spin), ПОТІМ нахиляємо весь
    уже повернутий диск уперед (tilt). Просте додавання кута до rotation_euler
    тут НЕ працює: 'XYZ'-ейлер компонує як Rz·Ry·Rx, тож zміна e.z після
    фіксованого нахилу по Y обертала б диск навколо СВІТОВОЇ Z, а не навколо
    його власної нормалі (уздовж носа)."""
    tilt = Quaternion((0.0, 1.0, 0.0), math.pi / 2.0)
    spin = Quaternion((0.0, 0.0, 1.0), spin_angle)
    return tilt @ spin


def _spawn_plane_propeller(drone):
    """Додати гвинт-«хрестовину» ПОВЕРХ існуючого носа «Потужнольоту» (не
    чіпаючи drone.data) — той самий трюк, що й для дрона."""
    mesh = _make_blade_mesh(PLANE_PROP_RADIUS, PLANE_PROP_RADIUS * 0.12)
    mat = _prop_material()
    name = f"{_PROP_PREFIX}plane"
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    obj.parent = drone
    obj.location = PLANE_PROP_CENTER
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = _plane_prop_quat(0.0)
    _spin_angle[name] = 0.0


def clear_propellers():
    """Прибрати всі поточні об'єкти-гвинти (перед перемиканням апарата; після
    clear_scene() вони й так уже зникли разом з рештою сцени)."""
    for obj in [o for o in bpy.data.objects if o.name.startswith(_PROP_PREFIX)]:
        bpy.data.objects.remove(obj, do_unlink=True)
    _spin_angle.clear()


def sync_propellers_for_vehicle(drone, vehicle):
    """Перевстановити гвинти під поточний апарат (дрон/Потужноліт) — викликати
    після build_scene()/switch_to_plane()/switch_to_drone()."""
    clear_propellers()
    if vehicle == "plane":
        _spawn_plane_propeller(drone)
    else:
        _spawn_drone_propellers(drone)


_MIN_SPIN_RATE = 8.0     # рад/с на холостому газу — лопаті завжди трохи «тремтять»
_MAX_SPIN_RATE = 140.0   # рад/с на повному газу


def spin(vehicle, throttle, dt):
    """Покрутити поточні гвинти на dt секунд; швидкість — від СИЛИ ТЯГИ, не
    напряму від важеля газу (throttle 0..1). Статична тяга гвинта росте
    приблизно як КВАДРАТ обертів (T ∝ ω²) — щоб анімована швидкість відповідала
    тязі, яку задає газ (тяга дрона лінійна від throttle — RealisticQuad._integrate;
    для Потужнольоту прямої тяги немає, throttle там — те саме «завдання
    потужності» двигуна), обороти мають рости як sqrt(тяги), тобто sqrt(throttle):
    на малому газі лопаті вже помітно швидші, ніж дав би лінійний лерп, і
    вирівнюються ближче до повного газу — так само, як у справжнього гвинта.
    Дрон: без нахилу (мотори вже «дивляться» вгору), тож просто змінюємо
    rotation_euler.z. Потужноліт: кут накопичується ОКРЕМО (_spin_angle) і
    щоразу перераховується через _plane_prop_quat(), щоб не зіпсувати
    фіксований нахил уперед (просте += на rotation_euler тут дало б хибне
    обертання навколо світової осі — див. коментар _plane_prop_quat)."""
    thrust_frac = math.sqrt(max(0.0, min(1.0, throttle)))
    rate = _MIN_SPIN_RATE + thrust_frac * (_MAX_SPIN_RATE - _MIN_SPIN_RATE)
    for i, obj in enumerate(o for o in bpy.data.objects if o.name.startswith(_PROP_PREFIX)):
        sign = DRONE_PROP_SPIN_DIRS[i % len(DRONE_PROP_SPIN_DIRS)] if vehicle == "drone" else 1
        angle = (_spin_angle.get(obj.name, 0.0) + sign * rate * dt) % (2.0 * math.pi)
        _spin_angle[obj.name] = angle
        if vehicle == "plane":
            obj.rotation_quaternion = _plane_prop_quat(angle)
        else:
            obj.rotation_euler = (0.0, 0.0, angle)
