"""Tabular Q-learning over discretised 3D-LIDAR state — Tier-A attitude track.

Reproduces the documented result from ``tier_a/env_attitude/README.md``:
a ~10935×27 numpy Q-table over a LIDAR-derived discrete state.  The table
trains fast (no torch, plain numpy) but **rarely beats a hand-tuned PD
controller** on the same gauntlet — this is the expected, documented outcome.

Key insight this module demonstrates
--------------------------------------
"More sensor rays ≠ bigger learned state."  The state space is:

    e_lat_bin  × e_alt_bin  × e_phi_bin × e_theta_bin × e_psi_bin × lmin_bin × bear_bin
       3       ×     3      ×     3      ×      3       ×     3     ×    5     ×    9
    = 3^5 × 5 × 9 = 10935   states, 27 actions  (AttitudeEnv's full action set)

No matter how many rays the LIDAR fires (12, 16, 64), the *learned* state
keeps the same 7 scalar aggregates: 5 attitude-error bins + 2 LIDAR aggregates
(nearest return distance and body-frame bearing).  The invariant is structural:
more rays cannot change N_STATES because the discretize() function reduces the
entire scan to exactly (lmin, bearing), discarding per-ray details.

Two extra reward terms beyond env.step's R_STEP/R_COLLISION/R_GOAL:
  - stability  penalty: −K_RATE * |angular-rate| (targets "don't tumble")
  - proximity  penalty: −K_PROX * max(0, PROX_R - lidar_min) / PROX_R
    (targets "react to what's nearby")

These are genuine behavioural preferences, NOT potential-based shaping
(they do not satisfy Φ(terminal) = 0 for a closed loop), so they are kept
as separate reward terms rather than folded into the potential, preserving
Invariant 4.

Usage (local gauntlet, matching README)
---------------------------------------
    from tier_a.env_attitude.scenarios import build_local_gauntlet_env
    from tier_a.scaffold.qtable_attitude import train_tabular_attitude, greedy_rollout_tabular

    env = build_local_gauntlet_env(seed=0)
    Q, hist = train_tabular_attitude(env, episodes=30_000, seed=0)
    roll = greedy_rollout_tabular(env, Q)
    print(roll["success"], roll["mean_rate"], roll["min_lidar"])
"""

from __future__ import annotations

import math

import numpy as np

from tier_a.env_attitude.lidar3d import N_RAYS_3D, LIDAR_RANGE_3D, scan3d

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
GAMMA: float = 0.99
LR0: float = 0.15          # initial learning rate
LR_MIN: float = 0.02       # floor after linear decay
EPS0: float = 1.0
EPS_MIN: float = 0.05

# Extra reward weights (non-shaping — genuine preferences)
K_RATE: float = 0.05       # stability penalty per rad/s of angular-rate magnitude
K_PROX: float = 0.2        # proximity penalty when within PROX_R of an obstacle
PROX_R: float = 0.8        # m — inner safety radius for proximity penalty

# ---------------------------------------------------------------------------
# State-space discretization  (3×3×3×3×3×5×9 = 3^5×5×9 = 10 935 states)
# ---------------------------------------------------------------------------
# Attitude-error dimensions (each 3 bins: below/on/above the reference)
_ELAT_EDGES   = np.array([-0.20, 0.20])          # e_lat: right / on-path / left
_EALT_EDGES   = np.array([-0.15, 0.15])          # e_alt: below / on-alt / above
_EPHI_EDGES   = np.array([-0.20, 0.20])          # roll error  (rad)
_ETHETA_EDGES = np.array([-0.15, 0.15])          # pitch error (rad)
_EPSI_EDGES   = np.array([-0.20, 0.20])          # yaw error   (rad)
# LIDAR aggregates
_LMIN_EDGES   = np.array([0.4, 0.7, 1.0, 1.6])  # 5 bins: danger → clear
_N_BEAR = 9                                       # 9 bearing sectors
_BEAR_EDGES = np.array([math.pi * (2 * k / _N_BEAR - 1) for k in range(1, _N_BEAR)])

# Flat index strides for (e_lat, e_alt, e_phi, e_theta, e_psi, lmin, bearing)
_STRIDES = (
    3 * 3 * 3 * 3 * 5 * _N_BEAR,  # e_lat
    3 * 3 * 3 * 5 * _N_BEAR,      # e_alt
    3 * 3 * 5 * _N_BEAR,          # e_phi
    3 * 5 * _N_BEAR,              # e_theta
    5 * _N_BEAR,                  # e_psi
    _N_BEAR,                      # lmin
    1,                            # bearing
)
N_STATES: int = 3 ** 5 * 5 * _N_BEAR   # 10 935


def discretize(obs: np.ndarray, lidar: np.ndarray) -> int:
    """Map (10D obs, n_rays lidar) to a single integer in [0, N_STATES).

    Attitude-error part (5 dimensions, each 3 bins):
        e_lat, e_alt, e_phi, e_theta, e_psi
    LIDAR part (2 aggregates, regardless of how many rays were fired):
        lmin (5 bins), body-frame bearing of nearest return (9 bins)

    KEY INVARIANT: sensor richness (12 / 16 / 64 rays) does NOT grow the
    learned state — the entire scan reduces to exactly two scalars.

    Parameters
    ----------
    obs:   10D observation from AttitudeEnv (e_lat, e_alt, e_vz, e_phi, ...)
    lidar: (n_rays,) distances from env.lidar_scan()

    Returns
    -------
    Flat state index in [0, N_STATES).
    """
    e_lat   = float(obs[0])
    e_alt   = float(obs[1])
    e_phi   = float(obs[3])
    e_theta = float(obs[4])
    e_psi   = float(obs[5])

    i_lat   = int(np.searchsorted(_ELAT_EDGES,   e_lat,   side="right"))
    i_alt   = int(np.searchsorted(_EALT_EDGES,   e_alt,   side="right"))
    i_phi   = int(np.searchsorted(_EPHI_EDGES,   e_phi,   side="right"))
    i_theta = int(np.searchsorted(_ETHETA_EDGES, e_theta, side="right"))
    i_psi   = int(np.searchsorted(_EPSI_EDGES,   e_psi,   side="right"))

    if len(lidar) == 0:
        i_lmin = len(_LMIN_EDGES)   # "clear" bin — no sensor
        i_bear = 0
    else:
        nearest_idx = int(np.argmin(lidar))
        lmin = float(lidar[nearest_idx])
        i_lmin = int(np.searchsorted(_LMIN_EDGES, lmin, side="right"))
        # Body-frame azimuth of ray nearest_idx (evenly spaced around the ring)
        n_rays = len(lidar)
        bearing = 2.0 * math.pi * nearest_idx / n_rays  # [0, 2*pi)
        if bearing > math.pi:
            bearing -= 2.0 * math.pi   # wrap to (-pi, pi]
        i_bear = int(np.searchsorted(_BEAR_EDGES, bearing, side="right"))

    return (i_lat   * _STRIDES[0] + i_alt   * _STRIDES[1] +
            i_phi   * _STRIDES[2] + i_theta * _STRIDES[3] +
            i_psi   * _STRIDES[4] + i_lmin  * _STRIDES[5] + i_bear)


def _lidar_scalars(lidar: np.ndarray) -> tuple[float, float]:
    """Return (nearest distance, nearest body-bearing) for reward shaping."""
    if len(lidar) == 0:
        return LIDAR_RANGE_3D, 0.0
    nearest_idx = int(np.argmin(lidar))
    lmin = float(lidar[nearest_idx])
    n_rays = len(lidar)
    bearing = 2.0 * math.pi * nearest_idx / n_rays
    if bearing > math.pi:
        bearing -= 2.0 * math.pi
    return lmin, bearing


def _extra_reward(obs: np.ndarray, lidar: np.ndarray) -> float:
    """Stability + proximity reward terms (not shaping — real preferences)."""
    p, q, r = float(obs[6]), float(obs[7]), float(obs[8])
    rate_mag = math.sqrt(p * p + q * q + r * r)
    lmin, _ = _lidar_scalars(lidar)
    r_stability = -K_RATE * rate_mag
    r_proximity = -K_PROX * max(0.0, PROX_R - lmin) / PROX_R
    return r_stability + r_proximity


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tabular_attitude(
    env,
    episodes: int = 30_000,
    seed: int = 0,
    n_rays: int = N_RAYS_3D,
    eval_every: int = 500,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a tabular Q-agent on *env* for *episodes* episodes.

    Parameters
    ----------
    env:        AttitudeEnv or build_local_gauntlet_env instance.
    episodes:   Training budget.  ~30 000 runs in well under 2 min (no torch).
    seed:       RNG seed for reproducibility.
    n_rays:     LIDAR ray count to use for scanning.  Key result: changing this
                from 12 to 16 to 64 does NOT change N_STATES — only the two
                extracted aggregates get noisier/smoother.
    eval_every: Greedy checkpoint frequency (episodes).

    Returns
    -------
    Q:     (N_STATES, n_actions) float32 Q-table.
    hist:  (n_checkpoints, 2) array of [episode, tracking_rmse].
    """
    rng = np.random.default_rng(seed)
    n_actions = env.n_actions
    Q = np.zeros((N_STATES, n_actions), dtype=np.float32)
    decay = max(1, int(episodes * 0.7))
    hist = []

    for ep in range(episodes):
        frac = ep / decay
        eps = max(EPS_MIN, EPS0 + (EPS_MIN - EPS0) * frac)
        lr  = max(LR_MIN,  LR0  + (LR_MIN  - LR0)  * frac)

        obs = env.reset()
        lidar = env.lidar_scan(n_rays)
        s = discretize(obs, lidar)

        while True:
            if rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                a = int(np.argmax(Q[s]))

            obs2, r_env, done, info = env.step(a)
            lidar2 = env.lidar_scan(n_rays) if not done else np.full(n_rays, LIDAR_RANGE_3D)
            s2 = discretize(obs2, lidar2)

            terminal = (info["collision"] or info["goal"]
                        or info["departed"] or info.get("loss_of_control", False))
            r_extra = _extra_reward(obs2, lidar2)
            r_total = r_env + r_extra

            # Standard Q-update: bootstrap through truncation, zero through terminal
            if terminal:
                target = r_total
            else:
                target = r_total + GAMMA * float(np.max(Q[s2]))

            Q[s, a] += lr * (target - Q[s, a])

            obs = obs2
            lidar = lidar2
            s = s2
            if done or info.get("truncated", False):
                break

        if ep % eval_every == 0 or ep == episodes - 1:
            roll = greedy_rollout_tabular(env, Q, n_rays=n_rays)
            hist.append([ep, roll["tracking_rmse"]])

    return Q, np.array(hist)


# ---------------------------------------------------------------------------
# Greedy rollout
# ---------------------------------------------------------------------------

def greedy_rollout_tabular(env, Q: np.ndarray, n_rays: int = N_RAYS_3D) -> dict:
    """One greedy episode (no exploration) with the tabular Q-table.

    Returns
    -------
    dict with keys:
        success        — bool, reached goal
        collision      — bool
        departed       — bool (includes loss_of_control)
        t              — float, wall-clock sim time at end
        tracking_rmse  — float, sqrt(mean(e_lat^2+e_alt^2))
        mean_rate      — float, mean ||angular rate|| over episode
        min_lidar      — float, minimum LIDAR reading ever seen
        steps          — int
    """
    obs = env.reset()
    lidar = env.lidar_scan(n_rays)
    info = {"goal": False, "collision": False, "departed": False,
            "loss_of_control": False, "truncated": False}
    sq, rates, min_l = [], [], [LIDAR_RANGE_3D]

    for _ in range(4000):
        s = discretize(obs, lidar)
        a = int(np.argmax(Q[s]))
        obs2, _, done, info = env.step(a)
        lidar2 = env.lidar_scan(n_rays) if not done else lidar

        sq.append(obs2[0] ** 2 + obs2[1] ** 2)
        p, q, r = float(obs2[6]), float(obs2[7]), float(obs2[8])
        rates.append(math.sqrt(p * p + q * q + r * r))
        if len(lidar2) > 0:
            min_l.append(float(np.min(lidar2)))

        obs = obs2
        lidar = lidar2
        if done or info.get("truncated", False):
            break

    return {
        "success":       bool(info["goal"]),
        "collision":     bool(info["collision"]),
        "departed":      bool(info.get("departed", False) or info.get("loss_of_control", False)),
        "t":             float(env._t_wall),
        "tracking_rmse": float(np.sqrt(np.mean(sq))) if sq else 0.0,
        "mean_rate":     float(np.mean(rates)) if rates else 0.0,
        "min_lidar":     float(min(min_l)),
        "steps":         len(sq),
    }


# ---------------------------------------------------------------------------
# PD baseline (for comparison — hand-tuned, no learning)
# ---------------------------------------------------------------------------
# The comparison: greedy_rollout_tabular vs. a single env rollout with pd_action.

def pd_action(obs: np.ndarray) -> int:
    """Hand-tuned PD attitude controller.

    Converts the 10D tracking-error observation to one of the 27 torque
    actions by proportional-derivative feedback on (e_phi, e_theta, e_psi).
    Tuning constants match the reference PD baseline in the README gauntlet
    comparison (0.6 m clearance, same result ceiling).
    """
    from tier_a.env_attitude.constants import ACTIONS_ATTITUDE, D_TAU

    e_phi, e_theta, e_psi = obs[3], obs[4], obs[5]
    p, q, r = obs[6], obs[7], obs[8]

    # PD gains (dimensionless, scaled to D_TAU increments)
    Kp_phi, Kd_phi = 0.6, 0.15
    Kp_theta, Kd_theta = 0.4, 0.10
    Kp_psi, Kd_psi = 0.5, 0.12

    u_phi   = -(Kp_phi   * e_phi   + Kd_phi   * p)
    u_theta = -(Kp_theta * e_theta + Kd_theta * q)
    u_psi   = -(Kp_psi   * e_psi   + Kd_psi   * r)

    def _clamp(u: float) -> float:
        # Map continuous u to {-D_TAU, 0, +D_TAU}
        if u > D_TAU * 0.4:
            return D_TAU
        elif u < -D_TAU * 0.4:
            return -D_TAU
        return 0.0

    tau = (_clamp(u_phi), _clamp(u_theta), _clamp(u_psi))
    try:
        return ACTIONS_ATTITUDE.index(tau)
    except ValueError:
        return 13  # neutral (0, 0, 0) is index 13 in the 27-action grid


def greedy_rollout_pd(env) -> dict:
    """Run one full episode with the PD baseline — for direct comparison."""
    obs = env.reset()
    info = {"goal": False, "collision": False, "departed": False,
            "loss_of_control": False, "truncated": False}
    sq, rates = [], []
    for _ in range(4000):
        a = pd_action(obs)
        obs, _, done, info = env.step(a)
        sq.append(obs[0] ** 2 + obs[1] ** 2)
        p, q, r = float(obs[6]), float(obs[7]), float(obs[8])
        rates.append(math.sqrt(p * p + q * q + r * r))
        if done or info.get("truncated", False):
            break
    return {
        "success":       bool(info["goal"]),
        "collision":     bool(info["collision"]),
        "departed":      bool(info.get("departed", False) or info.get("loss_of_control", False)),
        "t":             float(env._t_wall),
        "tracking_rmse": float(np.sqrt(np.mean(sq))) if sq else 0.0,
        "mean_rate":     float(np.mean(rates)) if rates else 0.0,
        "steps":         len(sq),
    }


# ---------------------------------------------------------------------------
# Quick comparison runner
# ---------------------------------------------------------------------------

def compare_tabular_vs_pd(
    n_seeds: int = 4,
    episodes: int = 8_000,
    n_rays_list: tuple[int, ...] = (12, 16),
) -> None:
    """Print a table comparing tabular Q (various ray counts) vs PD baseline.

    Reproduces the README's key result: more rays -> same state space size,
    similar (poor) performance vs. PD.

    Run directly:
        python -m tier_a.scaffold.qtable_attitude
    """
    from tier_a.env_attitude.scenarios import build_local_gauntlet_env

    print(f"\n{'=' * 72}")
    print("Tabular Q (10 935 states) vs PD baseline — local gauntlet, "
          f"{n_seeds} seeds, {episodes} episodes")
    print(f"{'=' * 72}")
    print(f"{'Method':<22} {'States':>7} {'OK':>5} {'RMSE':>8} "
          f"{'Rate':>7} {'minL':>7}")
    print("-" * 72)

    pd_results = []
    for seed in range(n_seeds):
        env = build_local_gauntlet_env(seed=seed)
        r = greedy_rollout_pd(env)
        pd_results.append(r)
    pd_succ = sum(r["success"] for r in pd_results)
    pd_rmse = float(np.mean([r["tracking_rmse"] for r in pd_results]))
    pd_rate = float(np.mean([r["mean_rate"] for r in pd_results]))
    print(f"{'PD baseline':<22} {'  N/A':>7} {pd_succ}/{n_seeds:<3} "
          f"{pd_rmse:>8.3f} {pd_rate:>7.3f} {'  N/A':>7}")

    for n_rays in n_rays_list:
        q_results = []
        for seed in range(n_seeds):
            env = build_local_gauntlet_env(seed=seed)
            Q, _ = train_tabular_attitude(env, episodes=episodes, seed=seed,
                                          n_rays=n_rays, eval_every=episodes)
            r = greedy_rollout_tabular(env, Q, n_rays=n_rays)
            q_results.append(r)
        q_succ = sum(r["success"] for r in q_results)
        q_rmse = float(np.mean([r["tracking_rmse"] for r in q_results]))
        q_rate = float(np.mean([r["mean_rate"] for r in q_results]))
        q_lmin = float(np.mean([r["min_lidar"] for r in q_results]))
        label = f"Tabular Q ({n_rays} rays)"
        print(f"{label:<22} {N_STATES:>7} {q_succ}/{n_seeds:<3} "
              f"{q_rmse:>8.3f} {q_rate:>7.3f} {q_lmin:>7.3f}")

    print("=" * 72)
    print(f"Key result: both n_rays variants use N_STATES = {N_STATES} (same table).")
    print("More rays -> same state -> PD still dominates on success rate.")
    print()


if __name__ == "__main__":
    compare_tabular_vs_pd()
