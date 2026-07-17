"""Wind environment constants. Identical to tier_d/env/constants.py, plus WIND_PROB."""

from __future__ import annotations

import math

# --- Domain -----------------------------------------------------------------
DOMAIN: float = 10.0
START_XY: tuple[float, float] = (1.0, 1.0)
GOAL_XY: tuple[float, float] = (9.0, 9.0)

# --- Discretisation ---------------------------------------------------------
GRID_N: int = 25
CELL: float = DOMAIN / GRID_N

# --- Forest generation (identical to tier_d/env/constants.py) ----------------------
BORDER_MARGIN: float = 0.9
AB_CLEARANCE: float = 1.1
MIN_TREE_SEP: float = 0.85

# --- Collision ---------------------------------------------------------------
TREE_R: float = 0.40

# --- MDP (identical to tier_d/env/constants.py) ------------------------------------
N_ACTIONS: int = 8
ACTIONS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, CELL),
    (1, 0, CELL),
    (0, -1, CELL),
    (0, 1, CELL),
    (-1, -1, CELL * math.sqrt(2)),
    (-1, 1, CELL * math.sqrt(2)),
    (1, -1, CELL * math.sqrt(2)),
    (1, 1, CELL * math.sqrt(2)),
)

R_STEP: float = -1.0
R_COLLISION: float = -100.0
R_GOAL: float = +100.0

# --- Wind disturbance (new) --------------------------------------------------
# With probability WIND_PROB, the intended action is replaced with a random one.
# This makes the environment stochastic and forces the agent to learn robustness.
# A* oracle sees this probability and adjusts L* accordingly.
WIND_PROB: float = 0.05  # 5% chance of wind gust per step

# --- LIDAR (Tier 3 state / rendering) ----------------------------------------
LIDAR_RAYS: int = 16
LIDAR_RANGE: float = 3.0

assert 2 * TREE_R < MIN_TREE_SEP, "trees could overlap"
assert AB_CLEARANCE > TREE_R, "start/goal could be inside a tree"
assert 0.0 <= WIND_PROB < 1.0, "WIND_PROB must be in [0, 1)"
