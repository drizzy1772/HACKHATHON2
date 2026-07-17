"""Аналітична колізія: еліпсоїд корпусу vs вертикальні циліндри-дерева.

PURE NUMPY. Повертає факт колізії + ГЛИБИНУ проникнення (для масштабу штрафу).

Точна редукція до 2D: ортогональна проєкція еліпсоїда на XY — ТОЧНИЙ еліпс
із матрицею форми M = Ā·Āᵀ, де Ā — верхні 2 рядки A = R·diag(a,b,c).
Для вертикального (нескінченного у робочій смузі) циліндра
«еліпсоїд ∩ циліндр» ⇔ «еліпс-тінь ∩ круг r_t».

Сам тест еліпс–круг — консервативна support-функція:

    n = (t − p)/δ;  h(n) = √(nᵀ M n);  depth = max(0, r_t + h − δ)

h(n) — точна півширина еліпса в напрямку n; консервативність (спрацьовує на
міліметри раніше) — бо найближча точка еліпса може лежати не вздовж n.
Санітарні якорі: рівний політ ⇒ h ≡ a; чистий крен φ поперек щілини ⇒
h = √((a cosφ)² + (c sinφ)²) — тотожно presented_half_width з
env_attitude/occupancy.py (крос-чек у тестах).
"""

from __future__ import annotations

import numpy as np

# дефолт = config.yaml["body"]["semi_axes"] (рішення користувача, CF2X-масштаб)
SEMI_AXES = (0.12, 0.12, 0.04)


def shadow_matrix(rot: np.ndarray, semi_axes=SEMI_AXES) -> np.ndarray:
    """2×2 матриця форми XY-тіні еліпсоїда. rot — R (body→world)."""
    a_full = np.asarray(rot, dtype=np.float64) @ np.diag(semi_axes)
    a_bar = a_full[:2, :]
    return a_bar @ a_bar.T


def z_extent(rot: np.ndarray, semi_axes=SEMI_AXES) -> float:
    """Піввисота еліпсоїда по світовій осі z (для смуги висот дерева)."""
    a_full = np.asarray(rot, dtype=np.float64) @ np.diag(semi_axes)
    return float(np.linalg.norm(a_full[2, :]))


def check_forest(pos: np.ndarray, rot: np.ndarray, trees: np.ndarray,
                 semi_axes=SEMI_AXES, tree_height: float = 4.0) -> tuple[bool, float]:
    """(колізія?, максимальна глибина, м) по всіх деревах одразу.

    pos — (3,) світова позиція центру; rot — R (body→world) 3×3;
    trees — (N,3): x, y, r. Дерева ростуть із землі до tree_height.
    """
    pos = np.asarray(pos, dtype=np.float64)
    trees = np.asarray(trees, dtype=np.float64)
    if len(trees) == 0:
        return False, 0.0

    # смуга висот: якщо весь еліпсоїд вище верхівки — вільно
    if pos[2] - z_extent(rot, semi_axes) > tree_height:
        return False, 0.0

    m = shadow_matrix(rot, semi_axes)
    d = trees[:, :2] - pos[:2]                     # (N,2)
    delta = np.linalg.norm(d, axis=1)              # (N,)
    safe = np.maximum(delta, 1e-9)
    nx, ny = d[:, 0] / safe, d[:, 1] / safe
    h = np.sqrt(np.maximum(
        m[0, 0] * nx**2 + 2.0 * m[0, 1] * nx * ny + m[1, 1] * ny**2, 0.0))
    # δ≈0 (центр усередині стовбура) → глибина = r + h
    depth = trees[:, 2] + h - np.where(delta < 1e-9, 0.0, delta)
    worst = float(np.max(depth))
    return worst > 0.0, max(worst, 0.0)
