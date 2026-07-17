# -*- coding: utf-8 -*-
"""
Процедурний нерівний рельєф (пагорби на основі шуму Перліна).

Рельєф — це квадратна арена 2·bounds × 2·bounds метрів, розбита на регулярну
сітку grid_res×grid_res. Висоту кожного вузла задає фрактальний шум (fBm поверх
mathutils.noise), тож поверхня фізично «складна» для польоту.

ДЖЕРЕЛО ІСТИНИ — heightmap (numpy-масив висот). Саме він серіалізується в
competition_map.json, тому завантажена мапа відновлює РІВНО ту саму поверхню
незалежно від версії Blender чи реалізації шуму — це запорука чесного грейдингу.

Модулі bpy/mathutils імпортуються ЛІНИВО (лише коли реально будуємо меш або
рахуємо шум), щоб `height_at`/`from_heightmap`/`to_heightmap` працювали і поза
Blender (напр. в аналітиці на venv).
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from . import SimConfig


class Terrain:
    """Нерівний рельєф як heightmap + білінійна вибірка висоти й нормалі."""

    def __init__(self, cfg: SimConfig, heightmap: np.ndarray):
        self.cfg = cfg
        # heightmap[row, col]: row ~ вісь Y, col ~ вісь X; форма (grid_res, grid_res)
        self.heightmap = np.asarray(heightmap, dtype=float)
        self.res = self.heightmap.shape[0]
        self.bounds = cfg.bounds
        self._mesh_obj = None

    # ── Побудова ────────────────────────────────────────────────────────────────

    @classmethod
    def generate(cls, cfg: SimConfig, seed: int) -> "Terrain":
        """Згенерувати рельєф детерміновано за насінням.

        Шум Перліна сам по собі детермінований за координатою; аби різні seed
        давали різні пагорби, координати вибірки зсуваємо на псевдовипадковий
        (але відтворюваний) офсет random.Random(seed).
        """
        from mathutils import Vector
        from mathutils import noise as bnoise

        rng = random.Random(seed)
        off_x = rng.uniform(0.0, 1000.0)
        off_y = rng.uniform(0.0, 1000.0)

        res = cfg.grid_res
        h = np.empty((res, res), dtype=float)
        for i in range(res):
            y = cls._axis_world(i, res, cfg.bounds)
            for j in range(res):
                x = cls._axis_world(j, res, cfg.bounds)
                sx = (x + off_x) * cfg.terrain_scale
                sy = (y + off_y) * cfg.terrain_scale
                h[i, j] = cfg.terrain_amp * cls._fbm(bnoise, Vector,
                                                     sx, sy, cfg.terrain_octaves)
        return cls(cfg, h)

    @classmethod
    def from_heightmap(cls, cfg: SimConfig, heightmap) -> "Terrain":
        """Відновити рельєф із збереженого heightmap (для competition_map.json)."""
        return cls(cfg, np.asarray(heightmap, dtype=float))

    def to_heightmap(self) -> List[List[float]]:
        """heightmap як вкладені списки (для JSON-серіалізації)."""
        return [[round(float(v), 6) for v in row] for row in self.heightmap]

    @staticmethod
    def _fbm(bnoise, Vector, x: float, y: float, octaves: int) -> float:
        """Фрактальний броунівський рух: сума октав шуму Перліна, нормована в [−1, 1]."""
        total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += amp * bnoise.noise(Vector((x * freq, y * freq, 0.0)))
            norm += amp
            amp *= 0.5
            freq *= 2.0
        return total / norm if norm else 0.0

    # ── Вибірка висоти/нормалі (чиста математика, без bpy) ───────────────────────

    @staticmethod
    def _axis_world(idx: int, res: int, bounds: float) -> float:
        """Світова координата вузла сітки за індексом (рівномірно на [−bounds, bounds])."""
        return -bounds + 2.0 * bounds * (idx / (res - 1))

    def _to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Світові (x, y) → дробові індекси сітки (col_u, row_v), із затиском у межі."""
        b, res = self.bounds, self.res
        u = (x + b) / (2.0 * b) * (res - 1)
        v = (y + b) / (2.0 * b) * (res - 1)
        u = min(max(u, 0.0), res - 1.0)
        v = min(max(v, 0.0), res - 1.0)
        return u, v

    def height_at(self, x: float, y: float) -> float:
        """Білінійна інтерполяція висоти рельєфу в точці (x, y)."""
        u, v = self._to_grid(x, y)
        j0, i0 = int(np.floor(u)), int(np.floor(v))
        j1 = min(j0 + 1, self.res - 1)
        i1 = min(i0 + 1, self.res - 1)
        fu, fv = u - j0, v - i0
        h = self.heightmap
        top = h[i0, j0] * (1 - fu) + h[i0, j1] * fu
        bot = h[i1, j0] * (1 - fu) + h[i1, j1] * fu
        return float(top * (1 - fv) + bot * fv)

    def normal_at(self, x: float, y: float) -> Tuple[float, float, float]:
        """Одинична нормаль до поверхні в (x, y) з градієнта висоти (центральні різниці)."""
        d = 2.0 * self.bounds / (self.res - 1)  # крок сітки, м
        dzdx = (self.height_at(x + d, y) - self.height_at(x - d, y)) / (2.0 * d)
        dzdy = (self.height_at(x, y + d) - self.height_at(x, y - d)) / (2.0 * d)
        n = np.array([-dzdx, -dzdy, 1.0])
        return tuple(n / np.linalg.norm(n))

    # ── Меш у Blender (потрібен bpy) ─────────────────────────────────────────────

    def build_mesh(self):
        """Створити меш-рельєф у сцені Blender як колізійну поверхню.
        Ім'я з префіксом TERRAIN_ — щоб LiDAR (ray_cast) і детектор зіткнень його бачили."""
        import bpy

        res, b = self.res, self.bounds
        verts = []
        for i in range(res):
            y = self._axis_world(i, res, b)
            for j in range(res):
                x = self._axis_world(j, res, b)
                verts.append((x, y, float(self.heightmap[i, j])))

        faces = []
        for i in range(res - 1):
            for j in range(res - 1):
                a = i * res + j
                faces.append((a, a + 1, a + res + 1, a + res))

        mesh = bpy.data.meshes.new("TERRAIN_mesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        # Згладити нормалі — рельєф читається як пагорби, а не грановані плитки
        for poly in mesh.polygons:
            poly.use_smooth = True

        obj = bpy.data.objects.new("TERRAIN_ground", mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(self._material())
        self._mesh_obj = obj
        return obj

    @staticmethod
    def _material():
        import bpy

        mat = bpy.data.materials.get("TerrainMat")
        if mat is not None:
            return mat
        mat = bpy.data.materials.new("TerrainMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.22, 0.28, 0.16, 1.0)  # трав'янистий ґрунт
        bsdf.inputs["Roughness"].default_value = 0.95
        return mat
