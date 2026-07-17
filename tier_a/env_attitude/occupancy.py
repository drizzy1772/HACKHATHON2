"""Continuous-domain collision check for Tier-A.

INVARIANT 3 (adapted): ``TREE_R`` is *imported*, never re-declared, from
``env.constants`` -- the single source of truth the base kit and oracle share.
This module's collision test operates on continuous ``(x, y)`` positions
rather than a discretised grid (there is no grid in this track), but it is
built on the exact same radius.

``collides()`` is the original point-mass test (drone has zero width) and
stays the default -- every existing Tier-A result was measured against it.
``collides_oriented()`` is a new, opt-in alternative that treats the drone as
a flat, rollable ellipsoid instead of a point: see ``presented_half_width()``.
"""

from __future__ import annotations

import math

import numpy as np

from tier_d.env.constants import TREE_R
from tier_a.env_attitude.constants import H_BODY, R_BODY, TREE_H

__all__ = [
    "TREE_R", "TREE_H", "horizontal_distance_to_nearest_tree", "collides",
    "presented_half_width", "collides_oriented",
]


def horizontal_distance_to_nearest_tree(xy: tuple[float, float], trees: np.ndarray) -> float:
    """Distance from a world point to the nearest tree centre, ignoring z."""
    if len(trees) == 0:
        return float("inf")
    d = trees - np.asarray(xy)
    return float(np.min(np.hypot(d[:, 0], d[:, 1])))


def collides(xy: tuple[float, float], z: float, trees: np.ndarray,
             tree_r: float = TREE_R, tree_h: float = TREE_H) -> bool:
    """True iff the point (xy, z) is inside a tree's trunk cylinder."""
    if not (0.0 <= z <= tree_h):
        return False
    return horizontal_distance_to_nearest_tree(xy, trees) < tree_r


def presented_half_width(phi: float, r_body: float = R_BODY, h_body: float = H_BODY) -> float:
    """Half-width the drone's body presents perpendicular to its direction of
    travel, as a function of roll ``phi``. Models the body as an ellipse
    (semi-axes ``r_body`` across the rotor span, ``h_body`` through the
    body's thickness) and projects its boundary onto the gap direction:
    parametrising the ellipse as ``(r cos t, h sin t)`` in the body frame and
    rotating by ``phi``, the projected half-width is the amplitude of
    ``r cos(t) cos(phi) - h sin(t) sin(phi)`` over ``t``, i.e.
    ``sqrt((r cos phi)**2 + (h sin phi)**2)``.

    NOT the simpler ``r|cos phi| + h|sin phi|`` (that is the bounding-box
    width of a rotated *rectangle*, not an ellipse -- it has a spurious local
    maximum a few degrees off level before decreasing). The ellipse formula
    used here decreases monotonically from ``r_body`` at ``phi=0`` (level) to
    ``h_body`` at ``|phi|=90 deg`` (knife-edge) -- the physical mechanism a
    pirouette roll exploits to fit through a gap narrower than the
    level-flight footprint (real quadcopters and fixed-wing aircraft both use
    this), not merely a coordinated-turn side effect."""
    return math.hypot(r_body * math.cos(phi), h_body * math.sin(phi))


def collides_oriented(xy: tuple[float, float], z: float, phi: float, trees: np.ndarray,
                       tree_r: float = TREE_R, tree_h: float = TREE_H,
                       r_body: float = R_BODY, h_body: float = H_BODY) -> bool:
    """Like ``collides()``, but the drone is a rolled ellipsoid, not a point:
    collision iff centre-to-centre clearance is less than
    ``tree_r + presented_half_width(phi)``."""
    if not (0.0 <= z <= tree_h):
        return False
    threshold = tree_r + presented_half_width(phi, r_body, h_body)
    return horizontal_distance_to_nearest_tree(xy, trees) < threshold
