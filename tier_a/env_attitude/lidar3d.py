"""Analytic 3D LIDAR: closed-form ray vs. finite vertical cylinder (tree trunk).

Rays are generated in the BODY frame and rotated into the world frame by the
drone's *current* attitude (phi, theta, psi) on every scan -- a body-fixed
ring, exactly like a real drone-mounted LIDAR: bank the body and the whole
sensor cone tilts with it. This is new, Tier-A-only geometry; the base kit's
``env/lidar.py`` stays a pure 2D, world-frame-fixed sensor used only for
Tier 3's rendering/state and is untouched by this module.

Sensor readings are the ONLY way obstacle geometry reaches the tabular agent
in ``agent/qtable_attitude.py`` -- it never reads ``env.trees`` directly. This
mirrors the base kit's own precedent: Tier 3's 2D LIDAR is an allowed sensor
read, not "map access," because a range return is indirect (it costs the
agent effort to sense and act on it) where a raw coordinate is not.
"""

from __future__ import annotations

import math

import numpy as np

from tier_d.env.constants import TREE_R
from tier_a.env_attitude.constants import TREE_H
from tier_a.env_attitude.dynamics import rotation_matrix

N_RAYS_3D = 12
LIDAR_RANGE_3D = 2.5


def body_ray_dirs(n_rays: int = N_RAYS_3D) -> np.ndarray:
    """(n_rays, 3) unit directions in the BODY frame: a horizontal ring
    around the body x-axis (nose/forward), evenly spaced in azimuth.
    Ray 0 points straight along body-forward (matches the gizmo's red axis).
    """
    a = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1), a


def _ray_cylinder(origin: np.ndarray, d: np.ndarray, centre_xy: np.ndarray,
                   radius: float, height: float) -> float:
    """Smallest positive t hitting the finite vertical cylinder (radius,
    0<=z<=height) centred at ``centre_xy``, else inf. The ray may approach
    the trunk from any tilt (d need not be horizontal): a horizontal circle
    solve for t, then a height check on the resulting z, exactly as a real
    3D LIDAR beam would be occluded by a physical tree trunk of finite height.
    """
    fx, fy = origin[0] - centre_xy[0], origin[1] - centre_xy[1]
    a = d[0] * d[0] + d[1] * d[1]
    if a < 1e-12:
        return np.inf  # ray is vertical: never crosses a vertical trunk's wall
    b = 2.0 * (fx * d[0] + fy * d[1])
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return np.inf
    sq = float(np.sqrt(disc))
    for t in sorted(((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a))):
        if t > 1e-9:
            z = origin[2] + t * d[2]
            if 0.0 <= z <= height:
                return t
    return np.inf


def scan3d(origin_xyz: np.ndarray, phi: float, theta: float, psi: float,
           trees: np.ndarray, n_rays: int = N_RAYS_3D,
           max_range: float = LIDAR_RANGE_3D) -> np.ndarray:
    """Return (n_rays,) distances in world units, clipped to ``max_range``.

    Rays are defined in the body frame and rotated into world space by the
    body's current attitude before intersection -- the sensor cone banks and
    pitches with the vehicle.
    """
    origin = np.asarray(origin_xyz, dtype=np.float64)
    body_dirs, _ = body_ray_dirs(n_rays)
    R = rotation_matrix(phi, theta, psi)
    world_dirs = body_dirs @ R.T  # world_dir_i = R @ body_dir_i

    out = np.full(n_rays, max_range, dtype=np.float64)
    for i, d in enumerate(world_dirs):
        best = np.inf
        for t_xy in trees:
            best = min(best, _ray_cylinder(origin, d, np.asarray(t_xy), TREE_R, TREE_H))
        out[i] = min(best, max_range)
    return out


def body_bearings(n_rays: int = N_RAYS_3D) -> np.ndarray:
    """(n_rays,) body-frame azimuth of each ray, radians, 0 = straight ahead."""
    return body_ray_dirs(n_rays)[1]


def endpoints3d(origin_xyz: np.ndarray, phi: float, theta: float, psi: float,
                dists: np.ndarray) -> np.ndarray:
    """(n_rays, 3) world-space ray endpoints, for drawing the scan fan."""
    body_dirs, _ = body_ray_dirs(len(dists))
    R = rotation_matrix(phi, theta, psi)
    world_dirs = body_dirs @ R.T
    return np.asarray(origin_xyz) + world_dirs * dists[:, None]


# --- true spherical variant: 3 great circles instead of one flat ring ------
# ``body_ray_dirs`` above is a single ring in the body's XY plane -- it can
# never sense straight up/down or anything off that one plane, no matter how
# the body tilts, because the ring tilts as a whole rigid disc with it. This
# is the richer sensor sketched on the whiteboard: 16 rays split across three
# mutually orthogonal great circles (XY, XZ, YZ), giving genuine coverage in
# every direction, not just "around" the body. Kept separate from
# ``body_ray_dirs``/``scan3d`` rather than replacing them: the trained
# tabular/hybrid agents (agent/qtable_attitude.py, agent/hybrid_attitude.py)
# were tuned against the flat-ring sensor and its horizontal "bearing"
# semantics, which don't carry over cleanly to rays pointing up or down.
N_RAYS_3D_SPHERICAL = 16


def body_ray_dirs_spherical(n_rays: int = N_RAYS_3D_SPHERICAL) -> np.ndarray:
    """(n_rays, 3) unit directions in the BODY frame, split as evenly as
    possible across the XY (horizontal ring, as before), XZ (forward-up) and
    YZ (right-up) great circles."""
    base, extra = divmod(n_rays, 3)
    counts = [base + (1 if i < extra else 0) for i in range(3)]
    dirs = []
    for i in range(counts[0]):
        a = 2.0 * np.pi * i / counts[0]
        dirs.append([np.cos(a), np.sin(a), 0.0])
    for i in range(counts[1]):
        a = 2.0 * np.pi * i / counts[1]
        dirs.append([np.cos(a), 0.0, np.sin(a)])
    for i in range(counts[2]):
        a = 2.0 * np.pi * i / counts[2]
        dirs.append([0.0, np.cos(a), np.sin(a)])
    return np.array(dirs, dtype=np.float64)


def scan3d_spherical(origin_xyz: np.ndarray, phi: float, theta: float, psi: float,
                     trees: np.ndarray, n_rays: int = N_RAYS_3D_SPHERICAL,
                     max_range: float = LIDAR_RANGE_3D) -> np.ndarray:
    """Like ``scan3d``, but over the 3-great-circle spherical ray set."""
    origin = np.asarray(origin_xyz, dtype=np.float64)
    body_dirs = body_ray_dirs_spherical(n_rays)
    R = rotation_matrix(phi, theta, psi)
    world_dirs = body_dirs @ R.T

    out = np.full(len(body_dirs), max_range, dtype=np.float64)
    for i, d in enumerate(world_dirs):
        best = np.inf
        for t_xy in trees:
            best = min(best, _ray_cylinder(origin, d, np.asarray(t_xy), TREE_R, TREE_H))
        out[i] = min(best, max_range)
    return out


def endpoints3d_spherical(origin_xyz: np.ndarray, phi: float, theta: float, psi: float,
                          dists: np.ndarray) -> np.ndarray:
    """(n_rays, 3) world-space endpoints for the spherical ray set."""
    body_dirs = body_ray_dirs_spherical(len(dists))
    R = rotation_matrix(phi, theta, psi)
    world_dirs = body_dirs @ R.T
    return np.asarray(origin_xyz) + world_dirs * dists[:, None]


# --- dense variant: 8 great circles x 8 rays = 64, for gap-shape judgment --
# ``body_ray_dirs_spherical`` above is 3 FIXED axis-aligned circles (XY, XZ,
# YZ) -- a coarse, if genuinely 3D, sampling. This is a denser generalisation:
# ``n_circles`` great circles, all sharing the body-X (forward) axis as a
# common diameter -- like meridians of a globe with body-X as the polar axis
# -- swept through evenly-spaced rotations about that axis, each carrying
# ``rays_per_circle`` points. Measured on this machine: even 64 rays costs
# ~417us/scan (~2% of one CPU core at this env's 50Hz tick) -- real-time cost
# is not the constraint. What IS a real constraint (see agent/pirouette_attitude.py):
# dense sensing must never translate into a large LEARNED state -- naively
# binning even 16 rays at 3 bins each would be 3**16 ~= 43M states, a direct
# regression to the state-space blowup agent/qtable_attitude.py's README
# section already documents as a failure mode. The guidance layer built on
# top of this sensor extracts a small, fixed number of scalars from the scan
# (lidar.min() and the agent's own attitude) regardless of n_rays.
N_CIRCLES_DENSE = 8
RAYS_PER_CIRCLE_DENSE = 8
N_RAYS_3D_DENSE = N_CIRCLES_DENSE * RAYS_PER_CIRCLE_DENSE  # 64


def body_ray_dirs_dense(n_circles: int = N_CIRCLES_DENSE,
                        rays_per_circle: int = RAYS_PER_CIRCLE_DENSE) -> np.ndarray:
    """(n_circles*rays_per_circle, 3) unit directions in the BODY frame.

    Circle ``k`` is the XZ great circle (``[cos a, 0, sin a]``) rotated about
    the body-X axis by ``beta_k = pi*k/n_circles`` -- sweeping ``k`` from 0 to
    ``n_circles-1`` covers betas in ``[0, pi)``; a further +pi would just
    retrace the same set of great circles in the opposite direction, so this
    range alone already gives every distinct circle through the X axis.

    ``k=0`` (beta=0) recovers ``body_ray_dirs_spherical``'s XZ circle exactly.
    For ``n_circles=8``, ``k=4`` (beta=pi/2) recovers its XY (horizontal) ring
    (up to a y-sign convention, since this construction parametrises the
    swept circles oppositely -- still the identical *set* of directions).
    """
    dirs = []
    for k in range(n_circles):
        beta = math.pi * k / n_circles
        cb, sb = math.cos(beta), math.sin(beta)
        for i in range(rays_per_circle):
            a = 2.0 * math.pi * i / rays_per_circle
            ca, sa = math.cos(a), math.sin(a)
            dirs.append([ca, -sa * sb, sa * cb])
    return np.array(dirs, dtype=np.float64)


def scan3d_dense(origin_xyz: np.ndarray, phi: float, theta: float, psi: float,
                 trees: np.ndarray, n_circles: int = N_CIRCLES_DENSE,
                 rays_per_circle: int = RAYS_PER_CIRCLE_DENSE,
                 max_range: float = LIDAR_RANGE_3D) -> np.ndarray:
    """Like ``scan3d``/``scan3d_spherical``, but over the dense multi-great-circle
    ray set (64 rays by default)."""
    origin = np.asarray(origin_xyz, dtype=np.float64)
    body_dirs = body_ray_dirs_dense(n_circles, rays_per_circle)
    R = rotation_matrix(phi, theta, psi)
    world_dirs = body_dirs @ R.T

    out = np.full(len(body_dirs), max_range, dtype=np.float64)
    for i, d in enumerate(world_dirs):
        best = np.inf
        for t_xy in trees:
            best = min(best, _ray_cylinder(origin, d, np.asarray(t_xy), TREE_R, TREE_H))
        out[i] = min(best, max_range)
    return out


def endpoints3d_dense(origin_xyz: np.ndarray, phi: float, theta: float, psi: float,
                      dists: np.ndarray, n_circles: int = N_CIRCLES_DENSE) -> np.ndarray:
    """(n_rays, 3) world-space endpoints for the dense ray set."""
    rays_per_circle = len(dists) // n_circles
    body_dirs = body_ray_dirs_dense(n_circles, rays_per_circle)
    R = rotation_matrix(phi, theta, psi)
    world_dirs = body_dirs @ R.T
    return np.asarray(origin_xyz) + world_dirs * dists[:, None]
