"""Builds the "pre-baked trajectory between trees" from the base kit's own A* oracle.

Pipeline, reusing ``oracle.astar`` verbatim (the oracle module is already
allowed to see tree coordinates -- the same trust boundary the base tier uses
to compute ``L*``):

    make_forest -> optimal_path_xy (A*) -> string_pull (smoothing)
    -> dense arc-length resample -> heading/curvature -> bank/pitch profile
    -> arc-length -> time via a trapezoidal cruise-speed schedule

The result is a fixed-altitude (``FLY_Z``) reference the agent must *track*
(never plan): a nominal yaw from the path tangent, a nominal roll from the
path's curvature via a coordinated-turn approximation, and a nominal pitch
from a simple accelerate/cruise/decelerate speed schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tier_a.env_attitude.constants import CRUISE_SPEED, FLY_Z, GRAVITY, N_TREES_ATTITUDE
from tier_a.env_attitude.forest import make_forest
from tier_d.oracle.astar import is_feasible, optimal_path_xy, string_pull

# --- trajectory-shape constants (pedagogical simplification, not physics) --
DS: float = 0.05  # m, dense-resampling step
PHI_REF_MAX: float = math.radians(25.0)  # coordinated-turn bank angle clip
THETA_REF_MAX: float = math.radians(15.0)  # pitch-from-acceleration clip
K_PITCH: float = 1.0  # pitch-per-g scale for the accel/decel ease schedule
V_MIN_FRAC: float = 0.2  # accel/decel floor, as a fraction of CRUISE_SPEED


def _wrap(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def _signed_curvature(p_prev: np.ndarray, p: np.ndarray, p_next: np.ndarray) -> float:
    """4*signed_area(triangle) / (product of side lengths) -- 1/circumradius,
    signed positive for a left (CCW) turn, matching atan2's orientation convention."""
    a = p - p_prev
    b = p_next - p
    c = p_next - p_prev
    lab, lbc, lac = np.hypot(*a), np.hypot(*b), np.hypot(*c)
    denom = lab * lbc * lac
    if denom < 1e-12:
        return 0.0
    signed_area = 0.5 * (a[0] * b[1] - a[1] * b[0])
    return 4.0 * signed_area / denom


@dataclass
class ReferenceTrajectory:
    """Dense, time-parameterised reference. ``trees`` is oracle/viz-only --
    the agent's observation must never read it (see AttitudeBlindEnv)."""

    seed: int
    n_trees: int
    trees: np.ndarray
    waypoints_xy: np.ndarray
    s: np.ndarray
    t: np.ndarray
    xy: np.ndarray
    psi_ref: np.ndarray
    phi_ref: np.ndarray
    theta_ref: np.ndarray
    kappa: np.ndarray
    z_ref: float
    S_total: float
    T_total: float

    def at(self, t: float) -> dict:
        """Reference state at time ``t`` (clamped to the trajectory's span)."""
        tc = float(np.clip(t, 0.0, self.T_total))
        x = float(np.interp(tc, self.t, self.xy[:, 0]))
        y = float(np.interp(tc, self.t, self.xy[:, 1]))
        psi = _wrap(float(np.interp(tc, self.t, self.psi_ref)))
        phi = float(np.interp(tc, self.t, self.phi_ref))
        theta = float(np.interp(tc, self.t, self.theta_ref))
        kappa = float(np.interp(tc, self.t, self.kappa))
        return {"xy": np.array([x, y]), "z": self.z_ref, "psi": psi,
                "phi": phi, "theta": theta, "kappa": kappa}

    def nearest(self, xy: tuple[float, float]) -> tuple[float, float, float]:
        """Nearest-sample projection: (progress_s, signed cross-track e_lat, tangent psi).

        ``e_lat`` is positive when ``xy`` sits to the left of the reference
        tangent direction -- this is the sign convention the agent's ``e_lat``
        observation uses.
        """
        p = np.asarray(xy, dtype=float)
        d = self.xy - p
        i = int(np.argmin(d[:, 0] ** 2 + d[:, 1] ** 2))
        psi_i = self.psi_ref[i]
        tangent = np.array([math.cos(psi_i), math.sin(psi_i)])
        v = p - self.xy[i]
        e_lat = float(tangent[0] * v[1] - tangent[1] * v[0])
        return float(self.s[i]), e_lat, _wrap(float(psi_i))


def build_reference(
    n_trees: int = N_TREES_ATTITUDE,
    seed: int = 0,
    speed: float = CRUISE_SPEED,
    ds: float = DS,
    trees: np.ndarray | None = None,
) -> ReferenceTrajectory | None:
    """Deterministic reference trajectory for (n_trees, seed), or None if infeasible.

    ``trees`` overrides the seeded forest with an explicit ``(m,2)`` array of
    tree centres -- used only by hand-built stress-test scenarios (e.g. a tight
    gauntlet between two trees); training/scoring always uses the seeded path.
    """
    if trees is None:
        trees = make_forest(n_trees, seed)
    if not is_feasible(trees):
        return None

    cells_path_xy = optimal_path_xy(trees)
    waypoints_xy = string_pull(cells_path_xy, trees)
    if len(waypoints_xy) < 2:
        return None

    return _trajectory_from_waypoints(waypoints_xy, trees, seed, n_trees, speed, ds)


def _segment_clear(p: np.ndarray, q: np.ndarray, trees: np.ndarray, tree_r: float) -> bool:
    """True if the straight segment p->q keeps >= tree_r from every tree centre."""
    if len(trees) == 0:
        return True
    d = q - p
    L2 = float(d @ d)
    for c in trees:
        if L2 < 1e-12:
            closest = p
        else:
            u = float(np.clip((c - p) @ d / L2, 0.0, 1.0))
            closest = p + u * d
        if float(np.hypot(*(c - closest))) < tree_r:
            return False
    return True


def build_local_reference(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    trees: np.ndarray,
    seed: int = 0,
    speed: float = CRUISE_SPEED,
    ds: float = DS,
    v_min_frac: float = V_MIN_FRAC,
) -> ReferenceTrajectory | None:
    """A short, hand-placed straight-line reference between two arbitrary
    points, skipping the A*/forest/START_XY-GOAL_XY pipeline entirely.

    Used only for focused local stress tests (e.g. "fly through this one
    gap"), never for training/scoring -- those always go through
    ``build_reference``'s seeded-forest + oracle path. Returns None if the
    straight segment itself would clip a tree.

    ``v_min_frac=1.0`` disables the accel/decel ease entirely (constant
    cruise speed, zero pitch throughout) -- appropriate for a short segment
    meant to represent the *middle* of a longer flight (the drone enters
    already at cruise speed), which is how ``env_attitude.scenarios``'s local
    gauntlet uses this function.
    """
    from tier_d.env.constants import TREE_R

    waypoints_xy = np.array([start_xy, goal_xy], dtype=float)
    if not _segment_clear(waypoints_xy[0], waypoints_xy[1], trees, TREE_R):
        return None
    return _trajectory_from_waypoints(waypoints_xy, trees, seed, len(trees), speed, ds, v_min_frac)


def _trajectory_from_waypoints(
    waypoints_xy: np.ndarray,
    trees: np.ndarray,
    seed: int,
    n_trees: int,
    speed: float,
    ds: float,
    v_min_frac: float = V_MIN_FRAC,
) -> ReferenceTrajectory:
    """Shared second half of the pipeline: waypoints -> dense, time-parameterised
    reference (arc-length resample, heading/curvature, bank/pitch profile,
    cruise-speed timing schedule). Used by both ``build_reference`` (A*-derived
    waypoints) and ``build_local_reference`` (hand-placed waypoints)."""
    seg_vec = np.diff(waypoints_xy, axis=0)
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
    seg_len_safe = np.where(seg_len < 1e-9, 1e-9, seg_len)
    cum = np.concatenate([[0.0], np.cumsum(seg_len_safe)])
    S_total = float(cum[-1])

    n_samples = max(int(S_total / ds) + 1, 2)
    s = np.linspace(0.0, S_total, n_samples)
    seg_idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_len) - 1)
    local = (s - cum[seg_idx]) / seg_len_safe[seg_idx]
    xy = waypoints_xy[seg_idx] + local[:, None] * seg_vec[seg_idx]

    d = np.gradient(xy, axis=0)
    psi_ref = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))

    kappa = np.zeros(n_samples)
    for i in range(1, n_samples - 1):
        kappa[i] = _signed_curvature(xy[i - 1], xy[i], xy[i + 1])
    if n_samples > 2:
        kappa[0], kappa[-1] = kappa[1], kappa[-2]

    accel_dist = min(1.0, max(S_total / 4.0, 1e-6))
    v_min = v_min_frac * speed
    v = np.where(
        s < accel_dist, v_min + (speed - v_min) * (s / accel_dist),
        np.where(s > S_total - accel_dist,
                 v_min + (speed - v_min) * ((S_total - s) / accel_dist), speed),
    )
    slope = np.where(
        s < accel_dist, (speed - v_min) / accel_dist,
        np.where(s > S_total - accel_dist, -(speed - v_min) / accel_dist, 0.0),
    )
    a_forward = slope * v
    # Positive theta tilts body-z (thrust) toward +body-x (forward) under this
    # module's R = Rz(psi)Ry(theta)Rx(phi) convention (verified directly against
    # env_attitude.dynamics.rotation_matrix) -- so accelerating forward needs a
    # POSITIVE theta_ref, not negative. Getting this sign backwards means
    # "tracking theta_ref perfectly" pitches away from the direction of travel
    # during every accel/decel leg, which was silently fighting every
    # controller (hand-tuned PD, DQN, tabular) tested against this reference.
    theta_ref = np.clip(K_PITCH * a_forward / GRAVITY, -THETA_REF_MAX, THETA_REF_MAX)
    phi_ref = np.clip(np.arctan2(speed ** 2 * kappa, GRAVITY), -PHI_REF_MAX, PHI_REF_MAX)

    ds_actual = s[1] - s[0] if n_samples > 1 else ds
    dt = ds_actual / ((v[:-1] + v[1:]) / 2.0)
    t = np.concatenate([[0.0], np.cumsum(dt)])

    return ReferenceTrajectory(
        seed=seed, n_trees=n_trees, trees=trees, waypoints_xy=waypoints_xy,
        s=s, t=t, xy=xy, psi_ref=psi_ref, phi_ref=phi_ref, theta_ref=theta_ref,
        kappa=kappa, z_ref=FLY_Z, S_total=S_total, T_total=float(t[-1]),
    )
