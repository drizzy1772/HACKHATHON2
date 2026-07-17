"""A* oracle: optimal path length L* and feasibility. Runs offline.

Ported from RunSky ``src/utils.py`` (octile heuristic, 8-connectivity, explicit
no-corner-cutting guard), rebuilt on ``env.occupancy`` so the occupancy radius
is *the same object* as the environment's collision radius (INVARIANT 3).

L* is the grid A* cost, not a smoothed length. Because the environment moves on
the same graph with the same corner rule, no agent path can beat it, so
``efficiency = L*/L <= 1`` holds exactly. ``string_pull`` is provided for
drawing a natural-looking line in the report and is deliberately NOT used for
scoring -- smoothing would make efficiency unreachable by construction.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from tier_d.env.constants import ACTIONS, CELL, GRID_N, TREE_R
from tier_d.env.occupancy import (
    GOAL_CELL,
    START_CELL,
    cell_center,
    diagonal_blocked,
    occupancy_grid,
)


def _octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2.0) * min(dr, dc)


def astar(
    blocked: np.ndarray,
    start: tuple[int, int] = START_CELL,
    goal: tuple[int, int] = GOAL_CELL,
) -> tuple[float, np.ndarray] | None:
    """Shortest 8-connected path start->goal. Returns (length_world, path_cells) or None.

    ``start``/``goal`` default to the base kit's fixed A/B cells; Tier-A's
    online planner (``env_attitude/online_planner.py``) is the one caller
    that overrides them, to replan from the drone's current cell rather than
    always from A."""
    if blocked[start] or blocked[goal]:
        return None

    open_heap: list[tuple[float, float, tuple[int, int]]] = [(_octile(start, goal), 0.0, start)]
    g_score = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            cells = [cur]
            while cells[-1] in parent:
                cells.append(parent[cells[-1]])
            cells.reverse()
            return g * CELL, np.array(cells)
        if cur in closed:
            continue
        closed.add(cur)

        r, c = cur
        for dr, dc, _ in ACTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_N and 0 <= nc < GRID_N):
                continue
            if blocked[nr, nc] or diagonal_blocked(blocked, r, c, dr, dc):
                continue
            step = math.sqrt(2.0) if (dr and dc) else 1.0
            ng = g + step
            nxt = (nr, nc)
            if ng < g_score.get(nxt, math.inf):
                g_score[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(open_heap, (ng + _octile(nxt, goal), ng, nxt))
    return None


def is_feasible(trees: np.ndarray) -> bool:
    """Cheap flood-fill connectivity of A and B. No path cost computed."""
    blocked = occupancy_grid(trees)
    if blocked[START_CELL] or blocked[GOAL_CELL]:
        return False
    seen = np.zeros_like(blocked)
    q = deque([START_CELL])
    seen[START_CELL] = True
    while q:
        r, c = q.popleft()
        if (r, c) == GOAL_CELL:
            return True
        for dr, dc, _ in ACTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_N and 0 <= nc < GRID_N):
                continue
            if seen[nr, nc] or blocked[nr, nc] or diagonal_blocked(blocked, r, c, dr, dc):
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return False


def optimal_length(trees: np.ndarray) -> float | None:
    """L* in world units, or None if B is unreachable from A."""
    result = astar(occupancy_grid(trees))
    return None if result is None else result[0]


def optimal_path_xy(trees: np.ndarray) -> np.ndarray | None:
    result = astar(occupancy_grid(trees))
    if result is None:
        return None
    return np.array([cell_center(r, c) for r, c in result[1]])


# -- report-only ------------------------------------------------------------
def _segment_clear(p: np.ndarray, q: np.ndarray, trees: np.ndarray) -> bool:
    """True if segment p->q keeps >= TREE_R from every tree centre."""
    if len(trees) == 0:
        return True
    d = q - p
    L2 = float(d @ d)
    for t in trees:
        if L2 < 1e-12:
            closest = p
        else:
            u = float(np.clip((t - p) @ d / L2, 0.0, 1.0))
            closest = p + u * d
        if float(np.hypot(*(t - closest))) < TREE_R:
            return False
    return True


def string_pull(path_xy: np.ndarray, trees: np.ndarray) -> np.ndarray:
    """Greedy line-of-sight smoothing. Cosmetic: for drawing only, never scoring."""
    if len(path_xy) < 3:
        return path_xy
    out = [path_xy[0]]
    i = 0
    while i < len(path_xy) - 1:
        j = len(path_xy) - 1
        while j > i + 1 and not _segment_clear(path_xy[i], path_xy[j], trees):
            j -= 1
        out.append(path_xy[j])
        i = j
    return np.array(out)
