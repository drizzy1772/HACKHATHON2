"""Curriculum gates for the pirouette guidance layer (agent/pirouette_attitude.py).

Extends env_attitude/scenarios.py's own build_local_gauntlet_env rather than
reinventing scenario-building: only the clearance changes across stages,
each keyed to the two thresholds env_attitude.occupancy.presented_half_width
already defines --

    TREE_R + R_BODY = 0.65 m  (level-flight footprint -- passable unrolled)
    TREE_R + H_BODY = 0.48 m  (knife-edge footprint -- passable only near
                               PHI_MAX_PIROUETTE)

-- so "how hard do I need to bank" is read off real physics, not asserted.
"""

from __future__ import annotations

import math

from tier_d.env.constants import TREE_R
from tier_a.env_attitude.constants import H_BODY, PHI_MAX_PIROUETTE, R_BODY
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.occupancy import presented_half_width
from tier_a.env_attitude.scenarios import build_local_gauntlet_env

PIROUETTE_CLEARANCE_STAGES: dict[str, float] = {
    "straight":  0.75,  # > TREE_R+R_BODY=0.65 -- passable level, no roll needed
    "angled":    0.58,  # between TREE_R+H_BODY=0.48 and TREE_R+R_BODY=0.65
    "pirouette": 0.50,  # just above TREE_R+H_BODY=0.48 -- near-full knife-edge
}


def build_pirouette_gate_env(seed: int = 0, stage: str = "pirouette", **kwargs) -> AttitudeEnv:
    """A local gate scenario (see build_local_gauntlet_env) with
    body_aware_collision=True by default -- the whole point of this track is
    the ellipsoid collision model, not the original point-mass one."""
    clearance = PIROUETTE_CLEARANCE_STAGES[stage]
    kwargs.setdefault("body_aware_collision", True)
    return build_local_gauntlet_env(seed=seed, clearance=clearance, **kwargs)


def gap_required_roll(clearance: float, tree_r: float = TREE_R, r_body: float = R_BODY,
                      h_body: float = H_BODY, phi_max: float = PHI_MAX_PIROUETTE) -> float | None:
    """Minimum roll magnitude (radians) needed for presented_half_width(phi)
    to fit within this gate's available clearance (clearance - tree_r).

    Returns 0.0 if the gap is already passable level (target >= r_body),
    None if it's impossible even fully rolled to phi_max (target is below
    presented_half_width(phi_max) -- the smallest width the envelope allows,
    since phi_max=85 deg stops just short of the true h_body floor at 90 deg),
    otherwise bisects presented_half_width (monotonically decreasing in phi
    -- see its own docstring) to find the smallest sufficient roll."""
    target = clearance - tree_r
    if target >= r_body:
        return 0.0
    if target < presented_half_width(phi_max, r_body, h_body):
        return None
    lo, hi = 0.0, phi_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if presented_half_width(mid, r_body, h_body) > target:
            lo = mid
        else:
            hi = mid
    return hi
