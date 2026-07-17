"""Wind-augmented grid MDP.

Same as env.GridWorld, but with stochastic action perturbations simulating wind gusts.
With probability WIND_PROB, the intended action is replaced with a uniformly random action.

This breaks the determinism and forces the agent to learn *robust* policies, not just
memorize the optimal path. Learning curves and efficiency metrics will differ from the
baseline — that's the point.

A* oracle is updated to account for wind: L* is recomputed with transition probabilities.
"""

from __future__ import annotations

import numpy as np

from tier_d.env_wind.constants import (
    ACTIONS,
    GRID_N,
    N_ACTIONS,
    R_COLLISION,
    R_GOAL,
    R_STEP,
    WIND_PROB,
)
from tier_d.env_wind.forest import make_forest
from tier_d.env_wind.occupancy import (
    GOAL_CELL,
    START_CELL,
    cell_center,
    diagonal_blocked,
    occupancy_grid,
)


class GridWorldWind:
    """8-action stochastic grid MDP over a seeded forest with wind disturbances."""

    def __init__(self, n_trees: int = 16, seed: int = 0, max_steps: int = 400, wind_prob: float = WIND_PROB) -> None:
        self.seed = seed
        self.n_trees = n_trees
        self.max_steps = max_steps
        self.wind_prob = wind_prob  # Can override WIND_PROB
        self.trees = make_forest(n_trees, seed)
        self.blocked = occupancy_grid(self.trees)
        self.n_states = GRID_N * GRID_N
        self.n_actions = N_ACTIONS
        self._rc = START_CELL
        self._steps = 0
        self._rng = np.random.default_rng(seed)

    # -- state encoding (identical to env.GridWorld) ---------------------------
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

    # -- dynamics (with wind) --------------------------------------------------
    def reset(self) -> int:
        self._rc = START_CELL
        self._steps = 0
        return self.state

    def step(self, a: int) -> tuple[int, float, bool, dict]:
        """Return (next_state, reward, done, info).

        With probability wind_prob, the intended action is replaced with a
        uniformly random action. This simulates a gust that changes the drone's
        intended direction.

        Leaving the domain is a no-op move that still costs a step. Entering an
        occupied cell -- or cutting a blocked corner -- is a terminal collision.
        """
        # Wind gust: replace intended action
        if self._rng.random() < self.wind_prob:
            a = int(self._rng.integers(N_ACTIONS))

        dr, dc, cost = ACTIONS[a]
        r, c = self._rc
        self._steps += 1
        nr, nc = r + dr, c + dc

        info = {"collision": False, "goal": False, "truncated": False, "moved": 0.0, "wind": self._rng.random() < self.wind_prob}

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
