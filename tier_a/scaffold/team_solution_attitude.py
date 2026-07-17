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


def train_attitude(env, episodes: int = 1200, seed: int = 0, eval_every: int = 0, **kwargs):
    """Скопіюйте сюди ваш тренер з ноутбука (starter_attitude.ipynb)."""
    raise NotImplementedError(
        "Вставте сюди train_attitude зі свого ноутбука (starter_attitude.ipynb)"
    )


def greedy_rollout_attitude(env, qnet) -> dict:
    """Скопіюйте сюди ваш greedy-прогін з ноутбука."""
    raise NotImplementedError(
        "Вставте сюди greedy_rollout_attitude зі свого ноутбука (starter_attitude.ipynb)"
    )
