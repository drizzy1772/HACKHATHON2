"""Q-table plot helpers, student-visible.

These functions consume a *finished* Q-table for visualisation only — they are
not part of the answer, so they live in the student kit. The reference agent
(``tier_d/admin/agent/qlearning.py``) re-exports them for organizer code.

INVARIANT 2 note: ``experienced_obstacle_value`` reads pain back out of the
Q-table the agent earned by crashing; it never stamps tree coordinates.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import ACTIONS


def experienced_obstacle_value(Q: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """What the agent *learned* about each tree cell, from collisions it suffered.

    A tree cell is never occupied, so ``max_a Q`` says nothing about it. But the
    agent does learn ``Q(neighbour, action_into_tree) -> R_COLLISION`` every time
    it crashes. Reading that back gives each obstacle a height equal to the pain
    it actually caused: trees the agent never touched stay at 0.

    This is what makes the demo honest. The peaks are grown from reward, not
    stamped from the tree coordinates (INVARIANT 2) -- an untouched tree is flat.
    """
    from tier_d.env.constants import GRID_N

    out = np.zeros((GRID_N, GRID_N))
    for r, c in np.argwhere(blocked):
        worst = 0.0
        for k, (dr, dc, _) in enumerate(ACTIONS):
            pr, pc = r - dr, c - dc  # the cell you would enter (r,c) from, via action k
            if not (0 <= pr < GRID_N and 0 <= pc < GRID_N) or blocked[pr, pc]:
                continue
            worst = min(worst, float(Q[pr * GRID_N + pc, k]))
        out[r, c] = worst
    return out


def value_surface(
    Q: np.ndarray,
    patch_goal: bool = True,
    blocked: np.ndarray | None = None,
    learned_peaks: bool = True,
) -> np.ndarray:
    """V(s) = max_a Q(s,a), reshaped to the (row, col) grid.

    Two display fills, neither of which the learning code ever reads:

    ``patch_goal`` -- the goal is terminal, so Q-learning never updates its row
    and it stays at 0, which would draw the sink as a *peak* in ``-V``.

    ``blocked`` -- tree cells stay at 0 because ``max_a Q`` is the value of the
    *best* action and the best action never walks into a tree. Filled from
    ``experienced_obstacle_value`` (default) so peak height reflects collisions
    actually suffered, or from a flat ``R_COLLISION`` if ``learned_peaks=False``.
    """
    from tier_d.env.constants import GRID_N, R_COLLISION, R_GOAL
    from tier_d.env.occupancy import GOAL_CELL

    V = Q.max(axis=1).reshape(GRID_N, GRID_N).copy()
    if blocked is not None:
        V[blocked] = (
            experienced_obstacle_value(Q, blocked)[blocked] if learned_peaks else R_COLLISION
        )
    if patch_goal:
        V[GOAL_CELL] = R_GOAL
    return V


def greedy_arrows(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) direction fields of the greedy policy, on the (row, col) grid."""
    from tier_d.env.constants import GRID_N

    a = Q.argmax(axis=1).reshape(GRID_N, GRID_N)
    u = np.zeros_like(a, dtype=float)
    v = np.zeros_like(a, dtype=float)
    for k, (dr, dc, _) in enumerate(ACTIONS):
        m = a == k
        n = np.hypot(dr, dc)
        u[m], v[m] = dc / n, dr / n
    return u, v
