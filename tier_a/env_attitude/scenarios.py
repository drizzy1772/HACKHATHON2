"""Hand-built stress-test scenarios -- never used for training/scoring
(those always draw from the seeded forest via ``build_reference``), only for
demonstrating that a learned policy can thread a genuinely tight gap it can
only sense via LIDAR.

Two variants:

``build_gauntlet_env`` -- the full A-to-B diagonal (~11 m, ~560 steps) with
two trees straddling it. Same long horizon as the default track; useful for
showing the gate *within* a normal flight, but inherits the same long-horizon
credit-assignment difficulty documented in ``env_attitude/README.md``.

``build_local_gauntlet_env`` -- a short (~3 m, tens of steps) local segment
that starts just before the gate and ends just past it, via
``build_local_reference`` (bypasses the global START_XY/GOAL_XY entirely).
This turns "react to a nearby obstacle without tumbling" into a genuinely
short-horizon RL problem, which is what actually converges in a same-day
budget -- see the training curves in the progressive-training filmstrip.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import GOAL_XY, START_XY
from tier_a.env_attitude.constants import CRUISE_SPEED
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.reference_trajectory import build_local_reference

GAUNTLET_CLEARANCE = 0.6  # metres from the centreline path to each trunk


def _gate_trees(mid: np.ndarray, along: np.ndarray, clearance: float) -> np.ndarray:
    perp = np.array([-along[1], along[0]])
    return np.array([mid + clearance * perp, mid - clearance * perp])


def _gauntlet_trees(clearance: float = GAUNTLET_CLEARANCE) -> np.ndarray:
    a, b = np.asarray(START_XY, dtype=float), np.asarray(GOAL_XY, dtype=float)
    mid = 0.5 * (a + b)
    along = (b - a) / np.linalg.norm(b - a)
    return _gate_trees(mid, along, clearance)


GAUNTLET_TREES = _gauntlet_trees()


def build_gauntlet_env(seed: int = 0, clearance: float = GAUNTLET_CLEARANCE, **kwargs) -> AttitudeEnv:
    """AttitudeEnv over the full A-to-B diagonal, with a hand-built two-tree
    gate at the midpoint instead of a seeded forest."""
    trees = _gauntlet_trees(clearance)
    return AttitudeEnv(n_trees=len(trees), seed=seed, custom_trees=trees, **kwargs)


def build_local_gauntlet_env(
    seed: int = 0,
    clearance: float = GAUNTLET_CLEARANCE,
    approach: float = 1.5,
    departure: float = 1.5,
    **kwargs,
) -> AttitudeEnv:
    """A short local episode: start ``approach`` metres before the gate,
    finish ``departure`` metres past it, straight along the A-to-B direction.

    This is the scenario actually used for training the reactive-avoidance
    behaviour: a ~(approach+departure) metre flight instead of the full
    ~11 m lap, so the RL problem's horizon matches the *local* skill being
    demonstrated (react to a sensed obstacle, don't tumble) rather than
    inheriting the full trajectory-tracking task's long-horizon difficulty.

    The segment represents the *middle* of an already-cruising flight: the
    drone enters at ``CRUISE_SPEED`` (``v0``) and the reference holds a
    constant speed throughout (``v_min_frac=1.0`` disables the accel/decel
    ease, which is only meaningful for a launch-to-landing leg).
    """
    a, b = np.asarray(START_XY, dtype=float), np.asarray(GOAL_XY, dtype=float)
    along = (b - a) / np.linalg.norm(b - a)
    mid = 0.5 * (a + b)
    trees = _gate_trees(mid, along, clearance)

    local_start = mid - approach * along
    local_goal = mid + departure * along
    ref = build_local_reference(tuple(local_start), tuple(local_goal), trees, seed=seed, v_min_frac=1.0)
    if ref is None:
        raise ValueError("local gauntlet segment is not clear of the gate trees -- widen clearance")
    # Drag is not compensated by the constant-speed (zero-pitch) reference, so
    # the real flight is slower than the idealised T_total; give it real room.
    kwargs.setdefault("max_time", 2.4 * ref.T_total)
    return AttitudeEnv(n_trees=len(trees), seed=seed, custom_reference=ref, v0=CRUISE_SPEED, **kwargs)
