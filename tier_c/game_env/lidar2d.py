# -*- coding: utf-8 -*-
"""
2D binned proximity sensor — плоский лідар навколо дрона в площині XY.

Аналітичний (без рейкасту): коло горизонту ділиться на n_bins азимутних секторів,
у кожен кладеться відстань до ПОВЕРХНІ найближчої перешкоди (циліндра). Векторизовано
через NumPy — рахується миттєво для будь-якої кількості перешкод. Висота (z) НЕ
враховується (площинний сенсор) — узгоджено з collision_and_bounds_status: якщо
горизонтально дрон вільний від осі стовбура/перешкоди, зіткнення не буде за жодної
висоти в межах стелі, тож 2D-проекція є фізично достатньою для уникнення.

Порядок бінів: index i ↔ азимут 2π·i/n_bins у СВІТОВІЙ системі координат (i=0 → +X).
"""

from __future__ import annotations

import numpy as np


def binned_lidar_2d(obs_x, obs_y, obs_r, ax: float, ay: float,
                    n_bins: int, max_r: float) -> np.ndarray:
    """Відстані (n_bins,) до найближчої перешкоди в кожному азимутному секторі,
    обрізані на max_r (де порожньо — повертає max_r). obs_x/obs_y/obs_r — numpy-масиви
    центрів і радіусів перешкод (дерева + колізійні тематичні перешкоди РАЗОМ);
    (ax, ay) — позиція дрона в площині XY."""
    out = np.full(n_bins, max_r, dtype=float)
    obs_r = np.asarray(obs_r, dtype=float)
    if obs_r.size == 0:
        return out
    dx = np.asarray(obs_x, dtype=float) - ax
    dy = np.asarray(obs_y, dtype=float) - ay
    dist_surf = np.hypot(dx, dy) - obs_r          # відстань до стінки (поверхні)
    active = dist_surf <= max_r
    if np.any(active):
        d = np.clip(dist_surf[active], 0.0, max_r)
        ang = np.arctan2(dy[active], dx[active]) % (2.0 * np.pi)
        bins = np.floor(ang / (2.0 * np.pi) * n_bins).astype(int) % n_bins
        np.minimum.at(out, bins, d)               # кілька в один бін → мінімум
    return out


def bin_angles(n_bins: int) -> np.ndarray:
    """Кут (рад, світова система) центру кожного біна — index i ↔ 2π·i/n_bins."""
    return 2.0 * np.pi * np.arange(n_bins) / n_bins


def lidar_obstacles_xyr(md):
    """Зібрати (obs_x, obs_y, obs_r) для лідара з MapData: стовбури дерев + ЛИШЕ
    колізійні тематичні перешкоди (декоративні — Патрон/дороговказ — лідар ігнорує,
    вони не заважають польоту, узгоджено з collision_and_bounds_status)."""
    xs = [t[0] for t in md.trees]
    ys = [t[1] for t in md.trees]
    rs = [t[3] for t in md.trees]
    for _kind, ox, oy, _oz, r, collidable in md.obstacles:
        if collidable:
            xs.append(ox); ys.append(oy); rs.append(r)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(rs, dtype=float)
