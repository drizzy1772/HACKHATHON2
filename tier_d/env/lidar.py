"""Analytic LIDAR: closed-form ray vs. circle. No physics engine.

Used for (a) rendering scan fans along a trajectory in the visual report and
(b) the optional Tier 3 continuous state. The tabular core never calls this.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import DOMAIN, LIDAR_RANGE, LIDAR_RAYS, TREE_R


def ray_dirs(n_rays: int = LIDAR_RAYS) -> np.ndarray:
    """(n_rays, 2) unit directions, evenly spaced over 360 degrees."""
    a = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def _ray_circle(origin: np.ndarray, d: np.ndarray, centre: np.ndarray, radius: float) -> float:
    """Smallest positive t with |origin + t*d - centre| = radius, else inf."""
    f = origin - centre
    b = float(f @ d)  # d is unit, so a == 1
    c = float(f @ f) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return np.inf
    sq = float(np.sqrt(disc))
    for t in (-b - sq, -b + sq):
        if t > 1e-9:
            return t
    return np.inf


def _ray_box(origin: np.ndarray, d: np.ndarray) -> float:
    """Distance to the domain wall along d (slab method); always finite."""
    ts = []
    for axis in (0, 1):
        if abs(d[axis]) < 1e-12:
            continue
        for bound in (0.0, DOMAIN):
            t = (bound - origin[axis]) / d[axis]
            if t > 1e-9:
                ts.append(t)
    return min(ts) if ts else np.inf


def scan(
    origin: np.ndarray,
    trees: np.ndarray,
    n_rays: int = LIDAR_RAYS,
    max_range: float = LIDAR_RANGE,
    include_walls: bool = True,
) -> np.ndarray:
    """Return (n_rays,) distances in world units, clipped to ``max_range``.

    A ray that hits nothing reports exactly ``max_range``.
    """
    origin = np.asarray(origin, dtype=np.float64)
    out = np.full(n_rays, max_range, dtype=np.float64)
    dirs = ray_dirs(n_rays)

    for i, d in enumerate(dirs):
        best = _ray_box(origin, d) if include_walls else np.inf
        for t_xy in trees:
            best = min(best, _ray_circle(origin, d, np.asarray(t_xy), TREE_R))
        out[i] = min(best, max_range)
    return out


def scan_path(path_xy: np.ndarray, trees: np.ndarray, **kw) -> np.ndarray:
    """(len(path), n_rays) scans along a trajectory -- the report's LIDAR fans."""
    return np.stack([scan(p, trees, **kw) for p in np.asarray(path_xy)])


def endpoints(origin: np.ndarray, dists: np.ndarray) -> np.ndarray:
    """(n_rays, 2) world-space ray endpoints, for drawing."""
    return np.asarray(origin) + ray_dirs(len(dists)) * dists[:, None]
