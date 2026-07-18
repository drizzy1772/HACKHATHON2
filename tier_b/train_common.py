"""Спільна тренувальна сантехніка PPO і SAC: фабрика env, векторизація, сіди.

Урок RunSky: Monitor обгортає env ДО векторизації (інакше губляться
per-episode infos). Однакові master-сіди у PPO і SAC за конструкцією —
порівняння чесне на рівні генерації лісів."""

from __future__ import annotations

import pathlib


def require_filled_hparams(hp: dict, algo: str) -> None:
    """Дружня перевірка учнівського конфіга: null-гіперпараметри = незаповнені TODO."""
    missing = [k for k, v in hp.items() if v is None]
    if missing:
        raise SystemExit(
            f"У вашому конфізі незаповнені гіперпараметри {algo}: "
            f"{', '.join(sorted(missing))}.\n"
            f"Відкрийте tier_b/scaffold/student_config.yaml, замініть null на "
            f"свої значення (і будьте готові обґрунтувати вибір журі)."
        )

INFO_KEYWORDS = ("success", "is_success", "collision", "jam", "out_of_bounds",
                 "gates_passed", "level")


def worker_master_seed(master_seed: int, rank: int) -> int:
    """Незбіжні простори сідів епізодів між воркерами (episode_seed =
    master·10_000 + ep, епізодів на воркера < 10_000·100)."""
    return master_seed * 100 + rank


def make_env(rank: int, level: int, master_seed: int, monitor_dir: str | None = None):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        from tier_b.envs.pirouette_aviary import PirouetteAviary

        env = PirouetteAviary(level=level,
                              master_seed=worker_master_seed(master_seed, rank))
        fname = None
        if monitor_dir is not None:
            pathlib.Path(monitor_dir).mkdir(parents=True, exist_ok=True)
            fname = str(pathlib.Path(monitor_dir) / f"monitor_{rank}")
        return Monitor(env, filename=fname, info_keywords=INFO_KEYWORDS)

    return _init


def build_vec_env(n_envs: int, level: int, master_seed: int, run_dir: str):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    monitor_dir = str(pathlib.Path(run_dir) / "monitor")
    return SubprocVecEnv([make_env(i, level, master_seed, monitor_dir)
                          for i in range(n_envs)], start_method="spawn")
