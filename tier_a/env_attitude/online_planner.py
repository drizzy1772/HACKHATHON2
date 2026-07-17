"""Online, LIDAR-driven path replanning: a periodic full A* replan over the
drone's own accumulated occupancy belief (``env_attitude/mapping.py``) --
not a hand-rolled D*-Lite.

Why a full periodic replan, not true incremental D*-Lite: this repo already
has a precedent for choosing the simplest mechanism that is still correct
over the maximally sophisticated one (a synced target network instead of
double-DQN in ``agent/qnet_attitude.py``; potential-based shaping instead of
ad hoc reward hacking everywhere else). ``oracle.astar.astar()`` over this
kit's ~625-cell grid is trivially cheap even every control tick, so there is
no performance case for D*-Lite's incremental repair -- the budget is better
spent on getting the mapping, anchoring, and tests right. See
``env_attitude/README.md``'s "Online LIDAR-based replanning" section for the
honest write-up of this trade-off and its measured cost.

Never takes a ``trees`` argument anywhere in this module: only the belief
map (built from LIDAR returns) and the drone's own pose, which it is always
entitled to know about itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tier_d.env.occupancy import cell_center, xy_to_cell
from tier_a.env_attitude.constants import LEAD_IN, LOOKAHEAD_CELLS, REPLAN_EVERY_STEPS
from tier_a.env_attitude.mapping import BeliefMap
from tier_d.oracle.astar import astar


@dataclass
class PlannerConfig:
    replan_every_steps: int = REPLAN_EVERY_STEPS
    lookahead_cells: int = LOOKAHEAD_CELLS
    lead_in: float = LEAD_IN


class OnlinePlanner:
    """Wraps ``oracle.astar.astar()`` over a belief that only ever grows from
    the drone's own LIDAR scans."""

    def __init__(self, goal_xy: tuple[float, float], config: PlannerConfig | None = None) -> None:
        self.goal_xy = goal_xy
        self.config = config or PlannerConfig()
        self._steps_since_replan = 0
        self._last_path_cells: np.ndarray | None = None

    def _path_invalidated(self, belief: BeliefMap) -> bool:
        """Cheap 'D*-Lite flavor' early trigger: has any of the next
        ``lookahead_cells`` cells of the last planned path just become
        known BLOCKED?"""
        if self._last_path_cells is None:
            return False
        for r, c in self._last_path_cells[: self.config.lookahead_cells]:
            if belief.is_blocked(int(r), int(c)):
                return True
        return False

    def maybe_replan(self, belief: BeliefMap, xy: tuple[float, float],
                      psi: float) -> np.ndarray | None:
        """Returns new anchored waypoints if a replan fires this tick, else None."""
        due = self._steps_since_replan >= self.config.replan_every_steps
        invalidated = self._path_invalidated(belief)
        no_plan_yet = self._last_path_cells is None  # always plan once, even before the first timer tick
        self._steps_since_replan += 1
        if not (due or invalidated or no_plan_yet):
            return None

        start_cell = xy_to_cell(*xy)
        goal_cell = xy_to_cell(*self.goal_xy)
        blocked = belief.planning_grid()
        if blocked[start_cell] or blocked[goal_cell]:
            return None  # believed-blocked start/goal -- keep flying, don't replan into nonsense

        result = astar(blocked, start=start_cell, goal=goal_cell)
        if result is None:
            return None  # no known-free path yet -- keep flying the current reference

        self._steps_since_replan = 0
        _, path_cells = result
        self._last_path_cells = path_cells
        return self._anchor_waypoints(path_cells, xy, psi)

    def _anchor_waypoints(self, path_cells: np.ndarray, xy: tuple[float, float],
                          psi: float) -> np.ndarray:
        """waypoint[0] = the drone's exact continuous position, then a short
        lead-in point along its current heading, then the A* cell-centre path
        -- skipping the path's own first cell (the drone's current cell): it
        sits behind the lead-in point almost as often as ahead of it, which
        would otherwise carve a spurious backward notch into the reference
        right at the anchor and corrupt the heading profile there. Keeps the
        new reference's initial cross-track and heading error small at the
        instant it replaces the old one -- see ``env_attitude/env.py``'s
        discontinuity mitigation."""
        lead_in_xy = (xy[0] + self.config.lead_in * math.cos(psi),
                      xy[1] + self.config.lead_in * math.sin(psi))
        cells = path_cells[1:] if len(path_cells) > 1 else path_cells
        rest = np.array([cell_center(int(r), int(c)) for r, c in cells])
        return np.vstack([np.asarray(xy, dtype=float), np.asarray(lead_in_xy), rest])
