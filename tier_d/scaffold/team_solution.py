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

import numpy as np  # noqa: F401  (знадобиться вашій реалізації)


def train(env, episodes: int = 3000, seed: int = 0, **kwargs):
    """Скопіюйте сюди вашу навчену версію train() з ноутбука."""
    raise NotImplementedError("Вставте сюди train зі свого ноутбука (starter.ipynb)")


def greedy_rollout(env, Q, max_steps: int = 400) -> dict:
    """Скопіюйте сюди вашу версію greedy_rollout() з ноутбука."""
    raise NotImplementedError("Вставте сюди greedy_rollout зі свого ноутбука (starter.ipynb)")
