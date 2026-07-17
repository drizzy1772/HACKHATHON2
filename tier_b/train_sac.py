#!/usr/bin/env python3
"""Тренування SAC. Гіперпараметри — config.yaml["sac"].

Чесна примітка щодо n_envs=4: SAC обмежений градієнтами, а не збором
досвіду; gradient_steps=-1 (= n_envs кроків градієнта на крок вектора)
зберігає канонічне співвідношення 1 update : 1 transition.

    conda run -n drones python tier_b/train_sac.py --steps 50000  # smoke
    conda run -n drones python tier_b/train_sac.py                # повні 2M
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from tier_b import CONFIG_PATH, load_config  # noqa: E402
from tier_b.callbacks import CurriculumCallback  # noqa: E402
from tier_b.train_common import build_vec_env, require_filled_hparams  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH),
                    help="шлях до config.yaml (команди: tier_b/scaffold/student_config.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hp = cfg["sac"]
    require_filled_hparams(hp, "sac")
    steps = args.steps or hp["total_steps"]
    run_dir = args.run_dir or f"tier_b/runs/sac/seed{args.seed}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    vec = build_vec_env(hp["n_envs"], args.level, args.seed, run_dir)
    model = SAC(
        "MlpPolicy", vec,
        buffer_size=hp["buffer_size"], batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"], tau=hp["tau"],
        learning_starts=hp["learning_starts"], train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        policy_kwargs=dict(net_arch=list(hp["net_arch"])),
        device=args.device or hp["device"], seed=args.seed,
        tensorboard_log=str(pathlib.Path(run_dir) / "tb"), verbose=1,
    )
    callbacks = [
        CurriculumCallback(start_level=args.level,
                           advance_window=cfg["curriculum"]["advance_window"],
                           advance_rate=cfg["curriculum"]["advance_rate"],
                           n_levels=len(cfg["curriculum"]["gaps"])),
        CheckpointCallback(save_freq=max(hp["checkpoint_every"] // hp["n_envs"], 1),
                           save_path=str(pathlib.Path(run_dir) / "ckpt"),
                           name_prefix="sac"),
    ]
    model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
    model.save(str(pathlib.Path(run_dir) / "final_model"))
    vec.close()
    print(f"збережено: {run_dir}/final_model.zip")


if __name__ == "__main__":
    main()
