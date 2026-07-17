"""Файл вашої команди (Tier-A). Сюди експортуєте свій тренер з ноутбука.

Коли обидві функції заповнені, ``python tier_a/scaffold/validate_team_attitude.py``
прожене їх на PUBLIC_SEEDS.

Контракт (сигнатури мають лишитися саме такими):

    train_attitude(env, episodes=1200, seed=0, eval_every=0, **kwargs) -> (qnet, history)
    greedy_rollout_attitude(env, qnet) -> dict
        повертає принаймні {"success": bool, "collision": bool,
                            "departed": bool, "t": float, "tracking_rmse": float}

Tier-A — дослідницький трек: можна змінювати будь-що в tier_a/ (агент, env,
планувальник), КРІМ tier_a/admin/. Інваріанти лишаються: стан = досвід (без
координат дерев), shaping — лише потенціал-орієнтований з Φ(термінал)=0.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# --- гіперпараметри (як у starter_attitude.ipynb) ---------------------------
GAMMA = 0.99
LR = 1e-3
EPS0, EPS_MIN = 1.0, 0.05
TRAIN_EVERY = 4
TARGET_SYNC_EVERY = 200
BATCH_SIZE = 64

W_POS, W_ATT = 1.0, 0.3


# --- потенціал-орієнтований shaping (Φ(термінал)=0) -------------------------
def potential(obs):
    e_lat, e_alt = obs[0], obs[1]
    e_phi, e_theta, e_psi = obs[3], obs[4], obs[5]
    return -(W_POS * math.hypot(e_lat, e_alt) + W_ATT * math.hypot(e_phi, e_theta, e_psi))


def shaping_F(obs, obs2, gamma, terminal=False):
    phi_next = 0.0 if terminal else potential(obs2)
    return gamma * phi_next - potential(obs)


# --- мережа й буфер ----------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim=10, n_actions=27, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000, seed=0):
        from collections import deque
        import random
        self.buf = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, s, a, r, s2, terminal):
        self.buf.append((s, a, r, s2, terminal))

    def sample(self, batch_size):
        batch = self._rng.sample(self.buf, min(batch_size, len(self.buf)))
        s, a, r, s2, terminal = zip(*batch)
        return (torch.as_tensor(np.array(s), dtype=torch.float32),
                torch.as_tensor(a, dtype=torch.long),
                torch.as_tensor(r, dtype=torch.float32),
                torch.as_tensor(np.array(s2), dtype=torch.float32),
                torch.as_tensor(terminal, dtype=torch.bool))

    def __len__(self):
        return len(self.buf)


# --- крок навчання DQN (ваша дірка з ноутбука) ------------------------------
def dqn_update(qnet, target_net, optimizer, batch, gamma):
    s, a, r, s2, terminal = batch
    q_sa = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # термінал → r; інакше → r + γ·max Q_target(s2)
        target = r + gamma * target_net(s2).max(dim=1).values * (~terminal).float()
    loss = ((q_sa - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


# --- контракт ----------------------------------------------------------------
def greedy_rollout_attitude(env, qnet) -> dict:
    """argmax_a Q(s,a), без дослідження. Так вас оцінюватимуть."""
    obs = env.reset()
    sq = []
    info = {"goal": False, "collision": False, "departed": False, "loss_of_control": False}
    for _ in range(2000):
        with torch.no_grad():
            a = int(qnet(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())
        obs2, r, done, info = env.step(a)
        sq.append(obs2[0] ** 2 + obs2[1] ** 2)
        obs = obs2
        if done or info["truncated"]:
            break
    rmse = float(np.sqrt(np.mean(sq))) if sq else 0.0
    return {
        "success": bool(info["goal"]),
        "collision": bool(info["collision"]),
        "departed": bool(info["departed"] or info["loss_of_control"]),
        "t": float(env._t_wall),
        "tracking_rmse": rmse,
        "steps": len(sq),
    }


def train_attitude(env, episodes: int = 1200, seed: int = 0, eval_every: int = 0, **kwargs):
    """Тренер DQN з target-мережею (перенесено з starter_attitude.ipynb)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    qnet = QNet(env.obs_dim, env.n_actions)
    target_net = QNet(env.obs_dim, env.n_actions)
    target_net.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=LR)
    buffer = ReplayBuffer(seed=seed)
    decay = max(1, int(episodes * 0.6))
    eval_interval = eval_every if eval_every > 0 else 25
    curve = []
    global_step, update_count = 0, 0

    for ep in range(episodes):
        eps = max(EPS_MIN, EPS0 + (EPS_MIN - EPS0) * ep / decay)
        obs = env.reset()
        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                with torch.no_grad():
                    a = int(qnet(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())
            obs2, r, done, info = env.step(a)
            global_step += 1
            terminal = info["collision"] or info["goal"] or info["departed"] or info["loss_of_control"]
            r_train = r + shaping_F(obs, obs2, GAMMA, terminal)
            buffer.push(obs, a, r_train, obs2, terminal)

            if len(buffer) >= BATCH_SIZE and global_step % TRAIN_EVERY == 0:
                dqn_update(qnet, target_net, optimizer, buffer.sample(BATCH_SIZE), GAMMA)
                update_count += 1
                if update_count % TARGET_SYNC_EVERY == 0:
                    target_net.load_state_dict(qnet.state_dict())

            obs = obs2
            if done or info["truncated"]:
                break

        if ep % eval_interval == 0 or ep == episodes - 1:
            roll = greedy_rollout_attitude(env, qnet)
            curve.append((ep, roll["tracking_rmse"]))
    return qnet, np.array(curve)
