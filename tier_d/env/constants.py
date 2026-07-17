"""Single source of truth for every geometric constant in the kit.

INVARIANT 3 lives here. ``TREE_R`` is imported by both the environment
(collision test) and the oracle (occupancy test). They must never diverge:
if they do, ``efficiency = L*/L`` can exceed 1 and the scoreboard becomes
dishonest. ``tier_d/admin/tests/test_invariants.py`` asserts both sides import *this*
symbol rather than defining their own.
"""

from __future__ import annotations

import math

# --- Domain -----------------------------------------------------------------
DOMAIN: float = 10.0  # world is [0, DOMAIN]^2
START_XY: tuple[float, float] = (1.0, 1.0)  # A
GOAL_XY: tuple[float, float] = (9.0, 9.0)  # B

# --- Discretisation ---------------------------------------------------------
GRID_N: int = 25  # 25 x 25 cells
CELL: float = DOMAIN / GRID_N  # 0.4

# --- Forest generation (frozen reference; see tier_d/env/forest.py) ----------------
# The build plan names three constants "0.9 / 1.1 / 0.85". Their roles, fixed
# here as the reference implementation:
BORDER_MARGIN: float = 0.9  # tree centre keeps >= this from the domain edge
AB_CLEARANCE: float = 1.1  # tree centre keeps >= this from A and from B
MIN_TREE_SEP: float = 0.85  # tree centres keep >= this from each other

# --- Collision --------------------------------------------------------------
# A cell is occupied iff its CENTRE lies within TREE_R of a tree centre.
# 2*TREE_R < MIN_TREE_SEP guarantees trees never overlap.
TREE_R: float = 0.40

# --- MDP --------------------------------------------------------------------
N_ACTIONS: int = 8
# (dr, dc, step cost in world units). Order is stable and part of the contract.
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

# --- LIDAR (Tier 3 state / rendering) --------------------------------------
LIDAR_RAYS: int = 16
LIDAR_RANGE: float = 3.0

assert 2 * TREE_R < MIN_TREE_SEP, "trees could overlap"
assert AB_CLEARANCE > TREE_R, "start/goal could be inside a tree"
