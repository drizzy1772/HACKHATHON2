# -*- coding: utf-8 -*-
"""
Меші тематичних об'єктів із готових STL-моделей (assets/*_source.stl) замість
процедурних примітивів. На відміну від patron_model.py (3MF, кастомний XML-парсер —
бо ця збірка Blender не має вбудованого імпортера 3MF) тут формат STL Blender уміє
імпортувати напряму (bpy.ops.wm.stl_import), тож просто читаємо готовий файл і
кешуємо verts/faces — жодного кастомного парсера не потрібно.

Кожен *_source.stl уже приведений (одноразовим офлайн-скриптом, не цим модулем) до
готового для розміщення вигляду: підошва на Z=0, центр по XY на (0,0) — можна одразу
obj.location = (x, y, terrain.height_at(x, y)).
"""

from __future__ import annotations

import os

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_HERE, "..", "assets")

_cache = {}


def _load(filename):
    """Імпортувати assets/<filename> у тимчасовий об'єкт, забрати (verts, faces) як
    прості списки (сумісні з mesh.from_pydata), прибрати сам тимчасовий об'єкт/меш зі
    сцени. Кешується за іменем файлу — сам файл не змінюється між перегенераціями мапи."""
    if filename in _cache:
        return _cache[filename]

    before = set(bpy.data.objects.keys())
    bpy.ops.wm.stl_import(filepath=os.path.join(_ASSETS, filename))
    obj = next(o for o in bpy.data.objects if o.name not in before)

    verts = [tuple(v.co) for v in obj.data.vertices]
    faces = [tuple(p.vertices) for p in obj.data.polygons]

    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)

    _cache[filename] = (verts, faces)
    return _cache[filename]


def get_tree_mesh():
    """Дерево (гола крона-гілки, без листя — як і процедурний стовбур раніше):
    висота рівно 1.0 м, підошва на Z=0 — масштабується на місці під (r, h) дерева."""
    return _load("tree_source.stl")


