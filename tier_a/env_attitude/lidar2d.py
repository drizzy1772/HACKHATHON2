"""2D, yaw-rotating LIDAR for the online path-discovery track.

This is a *different* sensor from ``lidar3d.py``'s body-frame-rotating 3D
scan: the drone flies at a fixed altitude ``FLY_Z`` inside the canopy, so a
flat, horizontal-plane ranging sensor needs no cylinder-height test -- trees
reduce to plain 2D circles, exactly the geometry ``env/lidar.py`` (the base
kit's Tier-3 sensor) already solves. This module reuses that closed-form
ray-vs-circle math directly rather than reimplementing it; ``env/lidar.py``
itself is left untouched.

Ray directions rotate with **yaw only** (``psi``), not full 3D attitude. A
real horizontally-scanning LIDAR ring (e.g. an RPLIDAR mounted flat on a
multirotor) is driven by heading, not momentary bank/pitch -- and coupling
this sensor's map quality to instantaneous roll/pitch would tangle the
online-vs-offline path comparison (see ``experiments/compare_online_planning.py``)
with a second, unrelated variable. That question -- how attitude jitter
degrades sensing -- is already the subject of ``lidar3d.py`` and the
hybrid/tabular controllers that consume it; this module stays orthogonal to it.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import LIDAR_RANGE, LIDAR_RAYS, TREE_R
from tier_d.env.lidar import _ray_box, _ray_circle

N_RAYS_2D = LIDAR_RAYS
LIDAR_RANGE_2D = LIDAR_RANGE


def ray_dirs2d(psi: float, n_rays: int = N_RAYS_2D) -> np.ndarray:
    """(n_rays, 2) unit directions in the WORLD frame: ray 0 points along
    the current heading ``psi``, the rest evenly spaced around it."""
    a = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False) + psi
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def scan2d(origin_xy, psi: float, trees: np.ndarray, n_rays: int = N_RAYS_2D,
           max_range: float = LIDAR_RANGE_2D, include_walls: bool = True) -> np.ndarray:
    """(n_rays,) distances in world units, clipped to ``max_range``."""
    origin = np.asarray(origin_xy, dtype=np.float64)
    dirs = ray_dirs2d(psi, n_rays)
    out = np.full(n_rays, max_range, dtype=np.float64)
    for i, d in enumerate(dirs):
        best = _ray_box(origin, d) if include_walls else np.inf
        for t_xy in trees:
            best = min(best, _ray_circle(origin, d, np.asarray(t_xy), TREE_R))
        out[i] = min(best, max_range)
    return out


def endpoints2d(origin_xy, psi: float, dists: np.ndarray) -> np.ndarray:
    """(n_rays, 2) world-space ray endpoints, for drawing the scan fan."""
    return np.asarray(origin_xy) + ray_dirs2d(psi, len(dists)) * dists[:, None]
