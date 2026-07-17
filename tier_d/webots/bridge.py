"""Policy -> waypoint bridge for the Webots 3D demo (Phase 7).

Webots is NOT installed on the build machine, so this module deliberately
contains **no Webots import**. It is pure geometry: it turns a learned greedy
policy into the horizontal target-velocity / waypoint stream a Webots controller
would consume, letting an existing PID hold altitude and attitude.

The Webots side (not written here, since it cannot be run or verified) only has
to call ``next_waypoint`` each control tick and feed the result to the PID.
Training never happens here -- the grid MDP stays the substrate.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import ACTIONS, CELL, GOAL_XY
from tier_d.env.occupancy import cell_center, xy_to_cell

CRUISE_SPEED = 1.2  # m/s, horizontal
ARRIVE_TOL = 0.5 * CELL


def policy_waypoints(path_xy: np.ndarray) -> np.ndarray:
    """Collapse a cell-by-cell path into turn-point waypoints.

    Consecutive steps in the same direction are merged, so the PID gets a few
    long legs instead of one waypoint per 0.4 m cell.
    """
    if len(path_xy) < 2:
        return np.asarray(path_xy)
    out = [path_xy[0]]
    prev_dir = None
    for a, b in zip(path_xy, path_xy[1:]):
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            continue
        d = d / n
        if prev_dir is not None and not np.allclose(d, prev_dir, atol=1e-9):
            out.append(a)
        prev_dir = d
    out.append(path_xy[-1])
    return np.array(out)


def greedy_action_at(Q: np.ndarray, xy: np.ndarray) -> int:
    """Which of the 8 actions the learned policy takes from world point ``xy``."""
    from tier_d.env.constants import GRID_N

    r, c = xy_to_cell(float(xy[0]), float(xy[1]))
    return int(np.argmax(Q[r * GRID_N + c]))


def next_waypoint(Q: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Centre of the cell the greedy policy moves into from ``xy``.

    Once inside the goal cell the policy is meaningless -- the goal is terminal,
    so Q-learning never updates its row and ``argmax`` would return action 0 and
    fly the drone straight back out. Home on the goal point instead.
    """
    from tier_d.env.constants import GRID_N
    from tier_d.env.occupancy import GOAL_CELL

    r, c = xy_to_cell(float(xy[0]), float(xy[1]))
    if (r, c) == GOAL_CELL:
        return np.asarray(GOAL_XY, dtype=float)
    dr, dc, _ = ACTIONS[greedy_action_at(Q, xy)]
    nr = int(np.clip(r + dr, 0, GRID_N - 1))
    nc = int(np.clip(c + dc, 0, GRID_N - 1))
    return np.asarray(cell_center(nr, nc))


def target_velocity(Q: np.ndarray, xy: np.ndarray, speed: float = CRUISE_SPEED) -> np.ndarray:
    """Horizontal velocity command: steer toward the next waypoint.

    Following the waypoint rather than the raw 8-way action vector is what makes
    the continuous trajectory converge onto cell centres instead of drifting
    diagonally past them.
    """
    d = next_waypoint(Q, xy) - np.asarray(xy, dtype=float)
    n = float(np.linalg.norm(d))
    return d / n * speed if n > 1e-12 else np.zeros(2)


def arrived(xy: np.ndarray, goal=np.asarray(GOAL_XY), tol: float = ARRIVE_TOL) -> bool:
    return bool(np.linalg.norm(np.asarray(xy) - goal) <= tol)


def simulate_bridge(Q: np.ndarray, start: np.ndarray, max_ticks: int = 500,
                    dt: float = 0.1) -> np.ndarray:
    """Kinematic stand-in for the PID: integrate the commanded velocity.

    Not a physics model. It exists so the bridge's *logic* can be tested and so
    the 3D animation has a continuous trajectory rather than cell hops.
    """
    xy = np.asarray(start, dtype=float).copy()
    traj = [xy.copy()]
    for _ in range(max_ticks):
        if arrived(xy):
            break
        wp = next_waypoint(Q, xy)
        d = wp - xy
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            break
        # Never overshoot the waypoint: a fixed-speed step past it would chatter.
        xy = xy + d / n * min(CRUISE_SPEED * dt, n)
        traj.append(xy.copy())
    return np.array(traj)
