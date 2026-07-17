# -*- coding: utf-8 -*-
"""
ГОТОВА ІНФРАСТРУКТУРА для планування шляху (ХАКАТОН: сам пошук шляху —
find_path() — реалізують учасники в solution.py, не тут).

Тут лишається лише представлення середовища для будь-якого grid-based
алгоритму пошуку, який учасники оберуть: сітка = аренa [-bounds,bounds]²
з кроком cell_size, клітина «зайнята», якщо потрапляє в коло (радіус
перешкоди + drone_radius + запас) навколо будь-якого дерева/колізійної
перешкоди — РІВНО той самий геометричний критерій зіткнення, що й
collision_and_bounds_status, лише спроєктований на 2D-сітку.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Vec2 = Tuple[float, float]

_SAFETY_MARGIN = 0.3    # запас понад drone_radius, м (щоб шлях не тертись об стовбур)


def build_occupancy_grid(md, cfg, cell_size: float = 1.0):
    """Побудувати булеву сітку зайнятості (True = перешкода) над ареною.
    Повертає (grid[ny][nx] як list-of-bytearray, cell_size, nx, ny)."""
    b = cfg.bounds
    nx = int(math.ceil(2.0 * b / cell_size)) + 1
    ny = nx
    grid = [bytearray(nx) for _ in range(ny)]

    obstacles: List[Tuple[float, float, float]] = [(t[0], t[1], t[3]) for t in md.trees]
    for _kind, ox, oy, _oz, r, collidable in md.obstacles:
        if collidable:
            obstacles.append((ox, oy, r))

    clear_r = cfg.drone_radius + _SAFETY_MARGIN
    for (ox, oy, r) in obstacles:
        rad = r + clear_r
        i0 = max(0, int((ox - rad + b) / cell_size))
        i1 = min(nx - 1, int((ox + rad + b) / cell_size))
        j0 = max(0, int((oy - rad + b) / cell_size))
        j1 = min(ny - 1, int((oy + rad + b) / cell_size))
        for j in range(j0, j1 + 1):
            cy = -b + j * cell_size
            row = grid[j]
            for i in range(i0, i1 + 1):
                cx = -b + i * cell_size
                if math.hypot(cx - ox, cy - oy) <= rad:
                    row[i] = 1
    return grid, cell_size, nx, ny


def world_to_cell(x: float, y: float, b: float, cell_size: float) -> Tuple[int, int]:
    """Світові (x,y) → індекс клітини сітки (i,j) — для будь-якого grid-based
    пошуку шляху над build_occupancy_grid()."""
    return int(round((x + b) / cell_size)), int(round((y + b) / cell_size))


def cell_to_world(i: int, j: int, b: float, cell_size: float) -> Vec2:
    """Індекс клітини (i,j) → світові (x,y) її центру."""
    return (-b + i * cell_size, -b + j * cell_size)
