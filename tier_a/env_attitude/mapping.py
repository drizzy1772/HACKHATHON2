"""Incremental occupancy belief built purely from 2D LIDAR range returns
(``env_attitude/lidar2d.py``) -- never from ground-truth tree coordinates.

A simple tri-state grid, not a probabilistic/log-odds occupancy map: this
sensor is analytic and noiseless (no false-positive/false-negative rate to
model), so a Bayesian layer would model an uncertainty that doesn't exist in
this simulator. A tri-state array with a ray-march update is this repo's
tabular-first philosophy applied to mapping, not a simplification of
something more "correct."

Grid geometry (``CELL``, ``GRID_N``) is imported from ``env.occupancy`` --
never redefined -- so this belief lives on exactly the same cells
``oracle.astar`` plans over (INVARIANT 3's spirit: one shared geometry, no
drift between what's sensed and what's planned).
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import CELL, GRID_N
from tier_d.env.occupancy import xy_to_cell
from tier_a.env_attitude.lidar2d import LIDAR_RANGE_2D, N_RAYS_2D, ray_dirs2d

UNKNOWN, FREE, BLOCKED = 0, 1, 2

MARCH_STEP: float = CELL / 2.0  # finer than one cell -- a ray can't skip a cell
HIT_MARGIN: float = CELL / 2.0  # push the "blocked" mark past the surface return,
# toward the tree's own cell. TREE_R == CELL by construction (env/constants.py),
# so a surface hit sits almost exactly one cell short of the tree's true
# occupied cell; pushing half a cell further lands close to it -- close, not
# exact, for off-centre (chord) rays. This quantization error is real and
# acknowledged, not hidden (see env_attitude/README.md's online-planning section).


class BeliefMap:
    """What the drone believes about the forest so far, built entirely from
    its own LIDAR returns. Never holds or derives from ground-truth trees."""

    def __init__(self) -> None:
        self.state = np.full((GRID_N, GRID_N), UNKNOWN, dtype=np.int8)

    def is_blocked(self, r: int, c: int) -> bool:
        return bool(self.state[r, c] == BLOCKED)

    def update_from_scan(self, origin_xy, dists: np.ndarray, psi: float,
                          n_rays: int = N_RAYS_2D, max_range: float = LIDAR_RANGE_2D) -> None:
        """Ray-march every return: cells crossed before a hit become FREE
        (unless already BLOCKED -- that mark is sticky, never downgraded: a
        false-positive obstacle is preferred over a false negative). A cell
        just past a genuine hit (``dist < max_range``) becomes BLOCKED."""
        origin = np.asarray(origin_xy, dtype=np.float64)
        dirs = ray_dirs2d(psi, len(dists))
        for d, dist in zip(dirs, dists):
            hit = dist < max_range - 1e-9
            n_steps = max(1, int(dist / MARCH_STEP))
            for k in range(n_steps + 1):
                t = min(k * MARCH_STEP, dist)
                p = origin + t * d
                r, c = xy_to_cell(p[0], p[1])
                if self.state[r, c] != BLOCKED:
                    self.state[r, c] = FREE
            if hit:
                p_hit = origin + (dist + HIT_MARGIN) * d
                r, c = xy_to_cell(p_hit[0], p_hit[1])
                self.state[r, c] = BLOCKED

    def planning_grid(self) -> np.ndarray:
        """Boolean mask for ``oracle.astar.astar()``: True only where BLOCKED.
        UNKNOWN cells count as passable (optimistic-unknown) -- this is what
        lets the drone move toward the goal before it has scanned everything,
        the core D*-Lite idea, without D*-Lite's incremental-repair machinery."""
        return self.state == BLOCKED

    def explored_fraction(self) -> float:
        return float(np.mean(self.state != UNKNOWN))
