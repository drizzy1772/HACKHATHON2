"""Occupancy + grid geometry, shared by the environment and the oracle.

INVARIANT 3: this module is the *only* place a collision radius is applied.
``env.gridworld`` and ``oracle.astar`` both call ``occupancy_grid`` — neither
re-derives it. A cell is occupied iff its centre lies within ``TREE_R`` of a
tree centre.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import CELL, GOAL_XY, GRID_N, START_XY, TREE_R


def cell_center(r: int, c: int) -> tuple[float, float]:
    """World coordinates of the centre of cell (row, col)."""
    return ((c + 0.5) * CELL, (r + 0.5) * CELL)


def cell_centers() -> tuple[np.ndarray, np.ndarray]:
    """``(gx, gy)`` arrays of shape (GRID_N, GRID_N), indexed [row, col]."""
    axis = (np.arange(GRID_N) + 0.5) * CELL
    gx, gy = np.meshgrid(axis, axis)  # gx varies with col, gy with row
    return gx, gy


def xy_to_cell(x: float, y: float) -> tuple[int, int]:
    """Nearest cell (row, col) containing world point (x, y), clamped."""
    c = int(np.clip(x / CELL, 0, GRID_N - 1))
    r = int(np.clip(y / CELL, 0, GRID_N - 1))
    return r, c


def occupancy_grid(trees: np.ndarray) -> np.ndarray:
    """Boolean (GRID_N, GRID_N) mask, True where the cell centre is blocked."""
    blocked = np.zeros((GRID_N, GRID_N), dtype=bool)
    if len(trees) == 0:
        return blocked
    gx, gy = cell_centers()
    for tx, ty in trees:
        blocked |= (gx - tx) ** 2 + (gy - ty) ** 2 < TREE_R**2
    return blocked


START_CELL: tuple[int, int] = xy_to_cell(*START_XY)
GOAL_CELL: tuple[int, int] = xy_to_cell(*GOAL_XY)


def diagonal_blocked(blocked: np.ndarray, r: int, c: int, dr: int, dc: int) -> bool:
    """No corner-cutting: a diagonal move is illegal if either orthogonal
    neighbour it squeezes past is occupied.

    Applied identically by the environment and A*, so no agent path can ever be
    shorter than L* (keeps ``efficiency = L*/L <= 1``).
    """
    if dr == 0 or dc == 0:
        return False
    return bool(blocked[r + dr, c] or blocked[r, c + dc])
