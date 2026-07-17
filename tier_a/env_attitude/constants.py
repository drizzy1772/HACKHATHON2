"""Physics + action + reward constants for the Tier-A attitude-control track.

Domain/collision geometry is re-imported (never redefined) from
``env.constants`` so ``TREE_R`` can never drift from the base kit's single
source of truth (INVARIANT 3's adapted form here: equality by construction,
since this value literally *is* the imported float, not a re-typed literal).

Everything below this point is new: rigid-body physical parameters, the
discretised torque action set, wind-disturbance magnitude, termination
thresholds and reward weights. See the plan's "time-boxing mitigations" for
why thrust is fixed (not an action) and the action set is kept to 27 entries.
"""

from __future__ import annotations

import itertools
import math

from tier_d.env.constants import TREE_R  # noqa: F401 -- re-exported, single source of truth

# --- forest sparsity for this track: a short lap, not the full forest ------
N_TREES_ATTITUDE: int = 8
TREE_H: float = 4.0  # matches viz/forest3d.py's TREE_H (trunk height)
FLY_Z: float = 1.6   # matches viz/forest3d.py's FLY_Z (reference cruise altitude)

# --- rigid-body physical parameters -----------------------------------------
MASS: float = 0.5  # kg
INERTIA: tuple[float, float, float] = (3.0e-3, 3.0e-3, 5.0e-3)  # (Ixx, Iyy, Izz), kg*m^2
GRAVITY: float = 9.81  # m/s^2
DRAG_COEFF: float = 0.15  # linear drag coefficient (c_d in dv/dt = ... - (c_d/m)*v)

# --- control: thrust fixed at hover, three torque channels ------------------
D_TAU: float = 0.01  # N*m, one discretised torque step per axis
ACTIONS_ATTITUDE: tuple[tuple[float, float, float], ...] = tuple(
    itertools.product((-D_TAU, 0.0, D_TAU), repeat=3)
)
N_ACTIONS_ATTITUDE: int = len(ACTIONS_ATTITUDE)  # 27
assert N_ACTIONS_ATTITUDE == 27

CRUISE_SPEED: float = 1.2  # m/s, matches webots/bridge.py's CRUISE_SPEED convention
DT: float = 0.02  # s, 50 Hz control tick

# --- wind disturbance (mirrors env_wind.constants.WIND_PROB) ----------------
WIND_PROB: float = 0.05
TAU_WIND_MAX: float = 0.02  # N*m, comparable to control authority -- genuinely disruptive

# --- termination thresholds --------------------------------------------------
MAX_DEVIATION: float = 1.0  # m, lateral or altitude deviation before "departed" terminal
PHI_MAX: float = math.radians(60.0)  # roll/pitch loss-of-control threshold
ARRIVE_TOL: float = 0.3  # m, arrival tolerance around the final waypoint
TIME_MARGIN: float = 1.5  # episode timeout = TIME_MARGIN * T_ref

# --- reward -------------------------------------------------------------
R_STEP: float = -1.0
R_COLLISION: float = -100.0
R_DEPARTED: float = -100.0
R_GOAL: float = 100.0

# --- shaping weights (agent/qnet_attitude.py's potential()) -----------------
W_POS: float = 1.0
W_ATT: float = 0.3

assert 0.0 <= WIND_PROB < 1.0, "WIND_PROB must be in [0, 1)"

# --- online LIDAR-based replanning (env_attitude/online_planner.py) --------
# A discovered path is longer than the offline-optimal L* by construction, so
# a fair timeout needs more slack than TIME_MARGIN gives the ground-truth case.
ONLINE_TIME_MARGIN: float = 2.5
REPLAN_EVERY_STEPS: int = 25   # ~0.5s at DT=0.02s
LOOKAHEAD_CELLS: int = 6       # invalidate-and-replan-early window
LEAD_IN: float = 0.3           # m, heading-matched anchor point ahead of the drone

# --- physical body envelope, for orientation-aware collision (the "pirouette") --
# A quadcopter is a flat disc, not a point: wide across the rotor span, thin
# top-to-bottom. Rolling it swings the thin edge into the direction of travel,
# shrinking the width it presents to a gap -- the actual physical mechanism a
# knife-edge/pirouette manoeuvre exploits, not just a coordinated-turn nicety.
# "Bigger, more stable" airframe dimensions, chosen deliberately over a small
# racing-quad footprint: R_BODY < TREE_R keeps the effect meaningful without
# implying the drone could physically stack up against a trunk.
R_BODY: float = 0.25  # m, half the rotor-to-rotor span (level-flight presented half-width)
H_BODY: float = 0.08  # m, half the body thickness (knife-edge presented half-width)

# PHI_MAX (above) is the *cruise* loss-of-control limit and stays untouched --
# a bank steep enough for the full knife-edge benefit (approaching 90 deg)
# would itself trip it. The pirouette manoeuvre is a deliberately different,
# opt-in operating envelope, not a relaxation of the default one.
PHI_MAX_PIROUETTE: float = math.radians(85.0)
