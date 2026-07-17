"""Файл вашої команди. Сюди експортуєте train/greedy_rollout зі свого ноутбука.

Коли обидві функції заповнені, ``python tier_d/scaffold/validate_team.py``
прожене їх на PUBLIC_SEEDS і покаже таблицю ефективності.

Контракт (сигнатури мають лишитися саме такими):

    train(env, episodes=3000, seed=0, **kwargs) -> (Q, history)
        Q       : np.ndarray форми (env.n_states, N_ACTIONS)
        history : будь-який ваш об'єкт логів (валідатор його не читає)

    greedy_rollout(env, Q, max_steps=400) -> dict
        повертає {"success": bool, "collision": bool,
                  "length": float, "path": np.ndarray}

Нагадування про інваріанти: стан — лише індекс клітинки (без координат дерев),
shaping — тільки потенціал-орієнтований з Φ(термінал)=0.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import CELL, GRID_N, N_ACTIONS
from tier_d.env.occupancy import GOAL_CELL

# --- гіперпараметри (набір A з starter.ipynb) --------------------------------
ALPHA = 0.05      # менший темп → точніший шлях
GAMMA = 0.99      # далекоглядний агент
EPS0 = 1.0
EPS_MIN = 0.01    # менше випадкових детурів наприкінці
USE_SHAPING = True


# --- потенціал-орієнтований shaping (Φ(термінал) = 0) ------------------------
def potential(s):
    """Φ(s) = −відстань від клітинки s до цілі (у світових одиницях)."""
    r, c = divmod(s, GRID_N)
    gr, gc = GOAL_CELL
    return -np.hypot(r - gr, c - gc) * CELL


def shaping_F(s, s2, gamma, terminal):
    if not USE_SHAPING:
        return 0.0
    phi_next = 0.0 if terminal else potential(s2)   # Φ(термінал) = 0
    return gamma * phi_next - potential(s)


# --- правило Беллмана --------------------------------------------------------
def q_update(Q, s, a, r, s2, terminal, alpha, gamma):
    """Оновити Q[s, a] на місці. Повернути TD-помилку."""
    if terminal:
        target = r                                  # кінець гри — майбутнього немає
    else:
        target = r + gamma * max(Q[s2])             # гра триває — плюс обіцянка майбутнього

    td = target - Q[s, a]
    Q[s, a] = Q[s, a] + alpha * td
    return td


# --- контракт ----------------------------------------------------------------
def greedy_rollout(env, Q, max_steps: int = 400) -> dict:
    """Політ без дослідження: суто argmax. Так вас оцінюватимуть."""
    s = env.reset()
    path = [env.xy]
    length = 0.0
    for _ in range(max_steps):
        s, _, done, info = env.step(int(np.argmax(Q[s])))
        path.append(env.xy)
        length += info["moved"]
        if info["collision"]:
            return dict(success=False, collision=True, length=length, path=np.array(path))
        if info["goal"]:
            return dict(success=True, collision=False, length=length, path=np.array(path))
    return dict(success=False, collision=False, length=length, path=np.array(path))


def train(env, episodes: int = 3000, seed: int = 0, **kwargs):
    """Табличний Q-learning із potential-based shaping."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, N_ACTIONS))
    curve = []
    decay = max(1, int(episodes * 0.6))

    for ep in range(episodes):
        eps = max(EPS_MIN, EPS0 + (EPS_MIN - EPS0) * ep / decay)
        s = env.reset()
        while True:
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done, info = env.step(a)
            terminal = info["collision"] or info["goal"]

            r_train = r + shaping_F(s, s2, GAMMA, terminal)
            q_update(Q, s, a, r_train, s2, terminal, ALPHA, GAMMA)

            s = s2
            if done:
                break

        if ep % 25 == 0:
            roll = greedy_rollout(env, Q)
            curve.append((ep, roll["length"] if roll["success"] else np.nan))
    return Q, np.array(curve)
