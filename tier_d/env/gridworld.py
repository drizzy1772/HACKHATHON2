"""The grid MDP. Foundation everything else depends on.

INVARIANT 2: the state is a bare cell index. Tree coordinates never enter it.
``GridWorld.state`` is an ``int``; there is no accessor that hands an agent the
obstacle layout. A LIDAR reading (``lidar_obs``) is offered separately for
Tier 3, and is derived from *sensing*, not from the map.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import (
    ACTIONS,
    GRID_N,
    N_ACTIONS,
    R_COLLISION,
    R_GOAL,
    R_STEP,
)
from tier_d.env.forest import make_forest
from tier_d.env.occupancy import (
    GOAL_CELL,
    START_CELL,
    cell_center,
    diagonal_blocked,
    occupancy_grid,
)


class GridWorld:
    """8-action deterministic grid MDP over a seeded forest."""

    def __init__(self, n_trees: int = 16, seed: int = 0, max_steps: int = 400) -> None:
        self.seed = seed
        self.n_trees = n_trees
        self.max_steps = max_steps
        self.trees = make_forest(n_trees, seed)
        self.blocked = occupancy_grid(self.trees)
        self.n_states = GRID_N * GRID_N
        self.n_actions = N_ACTIONS
        self._rc = START_CELL
        self._steps = 0

    # -- state encoding -----------------------------------------------------
    @staticmethod
    def encode(rc: tuple[int, int]) -> int:
        return rc[0] * GRID_N + rc[1]

    @staticmethod
    def decode(s: int) -> tuple[int, int]:
        return divmod(s, GRID_N)

    @property
    def state(self) -> int:
        return self.encode(self._rc)

    @property
    def goal_state(self) -> int:
        return self.encode(GOAL_CELL)

    @property
    def xy(self) -> tuple[float, float]:
        return cell_center(*self._rc)

    # -- dynamics -----------------------------------------------------------
    def reset(self) -> int:
        self._rc = START_CELL
        self._steps = 0
        return self.state

    def step(self, a: int) -> tuple[int, float, bool, dict]:
        """Return (next_state, reward, done, info).

        Leaving the domain is a no-op move that still costs a step. Entering an
        occupied cell -- or cutting a blocked corner -- is a terminal collision.
        """
        dr, dc, cost = ACTIONS[a]
        r, c = self._rc
        self._steps += 1
        nr, nc = r + dr, c + dc

        info = {"collision": False, "goal": False, "truncated": False, "moved": 0.0}

        # Wall of the domain: stay put, pay a step.
        if not (0 <= nr < GRID_N and 0 <= nc < GRID_N):
            info["truncated"] = self._steps >= self.max_steps
            return self.state, R_STEP, info["truncated"], info

        if self.blocked[nr, nc] or diagonal_blocked(self.blocked, r, c, dr, dc):
            info["collision"] = True
            return self.state, R_COLLISION, True, info

        self._rc = (nr, nc)
        info["moved"] = cost

        if self._rc == GOAL_CELL:
            info["goal"] = True
            return self.state, R_GOAL, True, info

        info["truncated"] = self._steps >= self.max_steps
        return self.state, R_STEP, info["truncated"], info

    # -- Tier 3 sensing (never used by the tabular core) --------------------
    def lidar_obs(self) -> np.ndarray:
        from tier_d.env.lidar import scan

        return scan(np.asarray(self.xy), self.trees)
