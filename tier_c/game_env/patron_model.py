# -*- coding: utf-8 -*-
"""
Меш "Патрон" із готової 3MF-моделі (assets/patron_source.3mf) замість процедурних
примітивів. Ця збірка Blender не має вбудованого імпортера 3MF (лише fbx/gltf/obj/
stl/ply), тож парсимо XML-меш формату 3MF (zip-контейнер, 3D/3dmodel.model) напряму
стандартною бібліотекою (zipfile + xml.etree) — без bpy, без зовнішніх залежностей.

3MF зберігає координати в міліметрах і застосовує ще й трансформ build-item (тут —
рівномірний масштаб + зсув на друкований майданчик, специфічний для слайсера).
Тут: застосовуємо цей трансформ, переводимо мм → м, центруємо по XY і "заземлюємо"
підошву на Z=0, і масштабуємо до TARGET_HEIGHT_M — щоб об'єкт можна було просто
розмістити (obj.location = (x, y, terrain.height_at(x, y))) без додаткових підгонок.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_PATH = os.path.join(_HERE, "..", "assets", "patron_source.3mf")

_NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"

TARGET_HEIGHT_M = 0.6   # бажана висота статуетки в сцені ("садовий" масштаб)

_cache = None


def _apply_transform(x, y, z, t):
    """3MF item/component transform: 12 чисел = 3x3 лінійна частина (рядками) +
    зсув; p' = p·M + T (рядковий вектор), як за специфікацією 3MF."""
    a, b, c, d, e, f, g, h, i, tx, ty, tz = t
    return (x * a + y * d + z * g + tx,
            x * b + y * e + z * h + ty,
            x * c + y * f + z * i + tz)


def _parse():
    with zipfile.ZipFile(_SOURCE_PATH) as zf:
        xml_bytes = zf.read("3D/3dmodel.model")
    root = ET.fromstring(xml_bytes)

    mesh_obj = None
    for obj in root.iter(_NS + "object"):
        if obj.find(_NS + "mesh") is not None:
            mesh_obj = obj
            break
    if mesh_obj is None:
        raise ValueError("У 3MF не знайдено об'єкта з мешем")
    mesh_el = mesh_obj.find(_NS + "mesh")
    verts = [(float(v.get("x")), float(v.get("y")), float(v.get("z")))
             for v in mesh_el.find(_NS + "vertices")]
    faces = [(int(t.get("v1")), int(t.get("v2")), int(t.get("v3")))
             for t in mesh_el.find(_NS + "triangles")]

    item_el = root.find(f".//{_NS}build/{_NS}item")
    t_str = item_el.get("transform") if item_el is not None else None
    t = [float(v) for v in t_str.split()] if t_str else [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

    verts_mm = [_apply_transform(x, y, z, t) for x, y, z in verts]
    verts_m = [(x / 1000.0, y / 1000.0, z / 1000.0) for x, y, z in verts_mm]

    xs = [v[0] for v in verts_m]
    ys = [v[1] for v in verts_m]
    zs = [v[2] for v in verts_m]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    z0 = min(zs)
    raw_h = max(zs) - z0
    scale = TARGET_HEIGHT_M / raw_h if raw_h > 1e-6 else 1.0

    verts_local = [((x - cx) * scale, (y - cy) * scale, (z - z0) * scale)
                   for x, y, z in verts_m]
    return verts_local, faces


def get_patron_mesh():
    """(verts, faces) у метрах, центровано по XY, підошва на Z=0. Кешується після
    першого парсингу — сам файл не змінюється між перегенераціями мапи."""
    global _cache
    if _cache is None:
        _cache = _parse()
    return _cache
