# -*- coding: utf-8 -*-
"""
Текстуровані меші (assets/*_source.glb) — дрон / фура-чекпоінт / "Потужноліт" /
уламки техніки. Джерела (FBX/OBJ з текстурами) оброблено офлайн: об'єднано в
один меш, декіматовано, зорієнтовано (носом на +X, де це важливо) і
нормалізовано (приземлені об'єкти — підошва на Z=0 і центр по XY на (0,0);
дрон/Потужноліт — літаючі, центровані по всіх 3 осях).

На відміну від stl_models.py (голі verts/faces, один плоский матеріал колір) —
тут важливі текстури/матеріали з glTF, тож підхід інший: імпортуємо ОБ'ЄКТ
(з мешем і матеріалами) ОДИН РАЗ і кешуємо як шаблон; кожне розміщення —
ЗВ'ЯЗАНИЙ дублікат (spawn), що ділить дані меша/матеріалів із шаблоном і має
лише власний трансформ. Шаблон прибирається з активної колекції (не рендериться
сам по собі), але лишається в bpy.data, щоб datablock не вивільнявся.
"""

from __future__ import annotations

import os

import bpy
from mathutils import Quaternion, Vector

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_HERE, "..", "assets")

_templates = {}

# wreck_dshk_source.glb несе в самому glTF (не через імпорт/Blender) alphaMode
# "BLEND" + KHR_materials_transmission=1 (скляна прозорість) майже на ВСІХ
# матеріалах кузова/коліс/салону — вочевидь помилка авторства асета (єдиний
# з усіх assets/*.glb, де так; порівняй із непрозорими wreck_tank/wreck_avtozak/
# checkpoint_truck). Тому виправляємо примусово, за іменем файлу, ОДИН РАЗ на
# шаблон — не через евристику, щоб не зачепити by mistake легітимне скло/розмиття
# гвинта Ка-52 (ka_52.glb) чи маяк-циліндр чекпойнта (scene.py).
_FORCE_OPAQUE_ASSETS = {"wreck_dshk_source.glb"}


def _force_opaque(obj):
    """Прибрати прозорість/просвічування з усіх матеріалів об'єкта: відключити
    Transmission (Weight) і Alpha від будь-яких вузлів-джерел і виставити їх
    у 0.0/1.0 відповідно, плюс blend_method='OPAQUE' (де підтримується)."""
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            for sock_name, opaque_value in (
                    ("Transmission Weight", 0.0), ("Transmission", 0.0), ("Alpha", 1.0)):
                sock = bsdf.inputs.get(sock_name)
                if sock is None:
                    continue
                for link in list(sock.links):
                    mat.node_tree.links.remove(link)
                sock.default_value = opaque_value
        if hasattr(mat, "blend_method"):
            try:
                mat.blend_method = 'OPAQUE'
            except Exception:   # noqa: BLE001
                pass
        if hasattr(mat, "show_transparent_back"):
            mat.show_transparent_back = False


def _load_template(filename):
    cached = _templates.get(filename)
    if cached is not None:
        try:
            cached.name   # animate_drone.clear_scene() wipes ALL bpy.data.objects
            return cached  # on every map rebuild — detect a stale reference below
        except ReferenceError:
            _templates.pop(filename, None)

    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=os.path.join(_ASSETS, filename))
    imported = [o for o in bpy.data.objects
                if o.name not in before and o.type == 'MESH']
    if len(imported) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in imported:
            o.select_set(True)
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
        template = bpy.context.object
    else:
        template = imported[0]

    # Прибрати можливих порожніх батьків (glTF-імпорт інколи створює вузол-корінь
    # сцени) — самому мешу вони не потрібні, локальні координати вже приведені.
    if template.parent is not None:
        bpy.context.view_layer.objects.active = template
        template.select_set(True)
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    template.name = f"_TEMPLATE_{filename}"
    for coll in list(template.users_collection):
        coll.objects.unlink(template)
    if filename in _FORCE_OPAQUE_ASSETS:
        _force_opaque(template)
    _templates[filename] = template
    return template


def spawn(filename, name, location=(0.0, 0.0, 0.0), rotation_z=0.0, scale=1.0,
         align_normal=None):
    """Розмістити один екземпляр assets/<filename> у сцені (зв'язаний дублікат
    шаблону — спільний меш/матеріали, власний трансформ).

    align_normal — якщо задано (нормаль рельєфу-опори в точці спавну), корпус
    ДОДАТКОВО нахиляється так, щоб його «низ» ліг уздовж схилу (як дерева в
    scene.py _build_trees), а не лишався горизонтальним. Без цього великий
    плоскодонний об'єкт (фура ~8 м, уламок ~5-7 м) на схилі торкався б землі
    лише в одній точці центру, а протилежний кут «висів» би в повітрі —
    саме так виглядала «левітація» на пагористому рельєфі (для такого об'єкта
    викликач має рахувати align_normal через scene._fit_ground_plane —
    МНК-площину під РЕАЛЬНИМИ кутками підошви, а не через terrain.normal_at()
    в одній точці, надто грубий крок якого ігнорує рельєф аж до кутків)."""
    template = _load_template(filename)
    obj = template.copy()
    obj.name = name
    obj.location = location
    if align_normal is not None:
        obj.rotation_mode = 'QUATERNION'
        yaw_q = Quaternion((0.0, 0.0, 1.0), rotation_z)
        tilt_q = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector(align_normal))
        obj.rotation_quaternion = tilt_q @ yaw_q
    else:
        obj.rotation_euler = (0.0, 0.0, rotation_z)
    # scale — число (рівномірний масштаб) АБО (sx,sy,sz) для виправлення
    # пропорцій конкретного asset'а (напр. занижена по Z «висота» автозака) —
    # застосовується в ЛОКАЛЬНИХ осях об'єкта (Blender: scale → rotate →
    # translate), тож коректно узгоджується з align_normal/rotation_z вище
    # незалежно від нахилу/курсу розміщення на мапі.
    if isinstance(scale, (int, float)):
        obj.scale = (scale, scale, scale)
    else:
        obj.scale = tuple(scale)
    bpy.context.collection.objects.link(obj)
    return obj
