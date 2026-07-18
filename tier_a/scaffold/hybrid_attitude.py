# -*- coding: utf-8 -*-
"""Hybrid attitude controller: hand-tuned PD stabilises orientation, a TINY
9x3 Q-table makes the high-level *direction* decision.

Architecture (the whole point of the achievement)
-------------------------------------------------
    PD  (from qtable_attitude.pd_action) — low level: holds whatever roll/pitch/
        yaw reference it is given. Fast, hand-tuned, no learning.
    9x3 table — high level: every tick it looks at a COARSE state (where am I
        off the path x where is the nearest tree) and picks one of three
        decisions — bank LEFT / HOLD / bank RIGHT. That decision is injected as
        a roll command into the PD's error, so the PD banks the drone that way.

State (9 = 3 x 3):
    e_lat sign          : {left of path, on path, right of path}
    nearest-tree bearing: {tree on left, ahead, tree on right}
Action (3):
    bank left (-) / hold (0) / bank right (+)

Documented result: ~1/10 seeds. The VALUE is the diagnosis, not the number:
even a clean split "decide vs stabilise" hits the SAME fixed-thrust veto — when
the gap needs a real bank, banking sinks the drone (lift ~ cos phi) and the PD
has no thrust channel to compensate. The table can *decide* to bank; physics
still vetoes the pass.

Run directly:
    python -m tier_a.scaffold.hybrid_attitude
"""

from __future__ import annotations

import math

import numpy as np

from tier_a.scaffold.qtable_attitude import (
    N_RAYS_3D, LIDAR_RANGE_3D, _lidar_scalars, pd_action, greedy_rollout_pd,
)

# ── high-level state/action space ────────────────────────────────────────────
N_HI_STATES = 9          # 3 (e_lat) x 3 (tree bearing)
N_HI_ACTIONS = 3         # bank left / hold / bank right
BANK_CMD = 0.30          # roll command injected per high-level decision, rad (~17°)

_ELAT_THRESH = 0.15      # m: |e_lat| below this = "on path"
_BEAR_THRESH = 0.35      # rad: |bearing| below this = tree "ahead"
_PROX_RANGE = 1.6        # m: ignore trees farther than this (bearing = "ahead")


def hi_state(obs: np.ndarray, lidar: np.ndarray) -> int:
    """Coarse 9-state: (e_lat sign) x (nearest-tree bearing sign)."""
    e_lat = float(obs[0])
    if e_lat < -_ELAT_THRESH:
        lat = 0
    elif e_lat > _ELAT_THRESH:
        lat = 2
    else:
        lat = 1

    lmin, bearing = _lidar_scalars(lidar)
    if lmin > _PROX_RANGE:          # no tree close enough to matter
        bear = 1
    elif bearing < -_BEAR_THRESH:
        bear = 0
    elif bearing > _BEAR_THRESH:
        bear = 2
    else:
        bear = 1
    return lat * 3 + bear


def hi_to_low(obs: np.ndarray, hi_action: int) -> int:
    """Turn a high-level decision into a 27-torque PD action.

    The decision is a roll command; we inject it into the PD's roll error so the
    PD banks the drone that way, then stabilises around it."""
    bank = (hi_action - 1) * BANK_CMD          # -BANK / 0 / +BANK
    obs_cmd = obs.copy()
    obs_cmd[3] = obs_cmd[3] + bank             # obs[3] = e_phi (roll error) -> command a bank
    return pd_action(obs_cmd)


# ── training the 9x3 table (plain numpy Q-learning) ─────────────────────────
def train_hybrid_attitude(env_fn, episodes: int = 4000, seed: int = 0,
                          alpha: float = 0.2, gamma: float = 0.99,
                          eps0: float = 1.0, eps_min: float = 0.05,
                          n_rays: int = N_RAYS_3D):
    """Learn Q[9,3]: which bank decision to make in each coarse state."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_HI_STATES, N_HI_ACTIONS), dtype=np.float32)
    decay = max(1, int(episodes * 0.6))

    for ep in range(episodes):
        eps = max(eps_min, eps0 + (eps_min - eps0) * ep / decay)
        env = env_fn(seed=ep % 32)             # vary maps across training
        obs = env.reset()
        lidar = env.lidar_scan(n_rays)
        while True:
            s = hi_state(obs, lidar)
            a = int(rng.integers(N_HI_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            obs2, r, done, info = env.step(hi_to_low(obs, a))
            lidar2 = env.lidar_scan(n_rays) if not done else lidar
            terminal = info["collision"] or info["goal"] or info["departed"] or info["loss_of_control"]

            s2 = hi_state(obs2, lidar2)
            target = r if terminal else r + gamma * float(np.max(Q[s2]))
            Q[s, a] += alpha * (target - Q[s, a])

            obs, lidar = obs2, lidar2
            if done or info.get("truncated", False):
                break
    return Q


def greedy_rollout_hybrid(env, Q: np.ndarray, n_rays: int = N_RAYS_3D) -> dict:
    """One greedy episode: table decides bank, PD stabilises."""
    obs = env.reset()
    lidar = env.lidar_scan(n_rays)
    info = {"goal": False, "collision": False, "departed": False,
            "loss_of_control": False, "truncated": False}
    sq, rates = [], []
    for _ in range(4000):
        s = hi_state(obs, lidar)
        a = int(np.argmax(Q[s]))
        obs2, _, done, info = env.step(hi_to_low(obs, a))
        lidar = env.lidar_scan(n_rays) if not done else lidar
        sq.append(obs2[0] ** 2 + obs2[1] ** 2)
        p, q, r = float(obs2[6]), float(obs2[7]), float(obs2[8])
        rates.append(math.sqrt(p * p + q * q + r * r))
        obs = obs2
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


# ── comparison runner ────────────────────────────────────────────────────────
def compare_hybrid_vs_pd(n_seeds: int = 10, episodes: int = 4000) -> None:
    """Reproduce the documented ~1/10 result and print the diagnosis."""
    from tier_a.env_attitude.scenarios import build_local_gauntlet_env

    Q = train_hybrid_attitude(build_local_gauntlet_env, episodes=episodes)

    print(f"\n{'=' * 68}")
    print(f"Hybrid (PD + 9x3 table) vs PD baseline — local gauntlet, {n_seeds} seeds")
    print(f"{'=' * 68}")
    print(f"{'Method':<26} {'states':>7} {'OK':>6} {'RMSE':>8}")
    print("-" * 68)

    pd = [greedy_rollout_pd(build_local_gauntlet_env(seed=s)) for s in range(n_seeds)]
    hy = [greedy_rollout_hybrid(build_local_gauntlet_env(seed=s), Q) for s in range(n_seeds)]
    pd_ok = sum(r["success"] for r in pd)
    hy_ok = sum(r["success"] for r in hy)
    print(f"{'PD baseline':<26} {'N/A':>7} {pd_ok:>4}/{n_seeds:<2} "
          f"{np.mean([r['tracking_rmse'] for r in pd]):>8.3f}")
    print(f"{'Hybrid PD + 9x3 table':<26} {N_HI_STATES:>7} {hy_ok:>4}/{n_seeds:<2} "
          f"{np.mean([r['tracking_rmse'] for r in hy]):>8.3f}")
    print("-" * 68)
    print(f"Result: hybrid {hy_ok}/{n_seeds}. The table CAN pick a bank direction, but")
    print("banking under fixed thrust sinks the drone (lift ~ cos phi) and the PD")
    print("has no thrust channel to hold altitude -> same physical veto as pure RL.")


if __name__ == "__main__":
    compare_hybrid_vs_pd()
