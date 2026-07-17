"""Ground-truth value functions. Offline; used to verify INVARIANT 6.

Two different objects, easy to conflate:

``dijkstra_cost_to_go`` is the *navigation function* U*(s): the exact
shortest-path cost from each cell to the goal. Its negation is the potential
surface the concept promises -- one minimum, at the goal, no local traps. That
property is structural (every cell on a shortest path has a strictly cheaper
neighbour), which is precisely what makes it testable.

``value_iteration`` is the optimal V* of the *reward* MDP the agent actually
learns. The learned ``max_a Q(s,a)`` should converge to this, so it is the
honest reference for "did the agent learn the right thing".
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from tier_d.env.constants import ACTIONS, CELL, GRID_N, R_COLLISION, R_GOAL, R_STEP
from tier_d.env.occupancy import GOAL_CELL, diagonal_blocked, occupancy_grid


def _legal_neighbours(blocked: np.ndarray, r: int, c: int):
    """Yield (nr, nc, world_cost) for every non-colliding move out of (r, c)."""
    for dr, dc, cost in ACTIONS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < GRID_N and 0 <= nc < GRID_N):
            continue
        if blocked[nr, nc] or diagonal_blocked(blocked, r, c, dr, dc):
            continue
        yield nr, nc, cost


def dijkstra_cost_to_go(trees: np.ndarray) -> np.ndarray:
    """U*(s): world-unit shortest-path cost from each cell to the goal.

    ``inf`` for blocked cells and for cells with no path. This is the
    navigation function; ``-U*`` is the ideal potential surface.
    """
    blocked = occupancy_grid(trees)
    dist = np.full((GRID_N, GRID_N), np.inf)
    if blocked[GOAL_CELL]:
        return dist

    dist[GOAL_CELL] = 0.0
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, GOAL_CELL)]
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        # Moves are symmetric (the corner rule is symmetric), so a reverse
        # search from the goal yields cost-to-go directly.
        for nr, nc, cost in _legal_neighbours(blocked, r, c):
            nd = d + cost
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(pq, (nd, (nr, nc)))
    return dist


def has_single_minimum(cost: np.ndarray) -> bool:
    """INVARIANT 6: every reachable non-goal cell has a strictly cheaper neighbour."""
    finite = np.isfinite(cost)
    for r in range(GRID_N):
        for c in range(GRID_N):
            if not finite[r, c] or (r, c) == GOAL_CELL:
                continue
            best = min(
                (cost[nr, nc] for nr, nc, _ in _legal_neighbours(~finite, r, c)),
                default=math.inf,
            )
            if not best < cost[r, c]:
                return False
    return True


def value_iteration(trees: np.ndarray, gamma: float = 0.99, tol: float = 1e-9) -> np.ndarray:
    """Optimal V* of the reward MDP. ``nan`` on blocked cells (unreachable states).

    Terminals are absorbing with V = 0, matching the Q-learning target. The goal
    cell therefore holds 0 here; ``viz`` patches it to ``R_GOAL`` for display so
    the surface shows a sink rather than a hole.
    """
    blocked = occupancy_grid(trees)
    V = np.zeros((GRID_N, GRID_N))
    free = ~blocked

    while True:
        delta = 0.0
        for r in range(GRID_N):
            for c in range(GRID_N):
                if blocked[r, c] or (r, c) == GOAL_CELL:
                    continue
                best = -math.inf
                for dr, dc, _ in ACTIONS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < GRID_N and 0 <= nc < GRID_N):
                        q = R_STEP + gamma * V[r, c]  # wall bump: stay put
                    elif blocked[nr, nc] or diagonal_blocked(blocked, r, c, dr, dc):
                        q = R_COLLISION  # terminal
                    elif (nr, nc) == GOAL_CELL:
                        q = R_GOAL  # terminal
                    else:
                        q = R_STEP + gamma * V[nr, nc]
                    best = max(best, q)
                delta = max(delta, abs(best - V[r, c]))
                V[r, c] = best
        if delta < tol:
            break

    V[blocked] = np.nan
    return V


def optimal_efficiency_bound(trees: np.ndarray) -> float:
    """L* in world units, straight from the navigation function (cross-check on A*)."""
    from tier_d.env.occupancy import START_CELL

    return float(dijkstra_cost_to_go(trees)[START_CELL])
