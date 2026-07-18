#!/usr/bin/env python3
"""GUI-візуалізатор LIDAR — порт патерну RunSky/scripts/demo.py:
живий PyBullet GUI, 34 промені через addUserDebugLine (червоний hit /
сірий miss, повторне використання через replaceItemUniqueId), chase-камера.

    conda run -n drones python tier_b/viz_lidar.py --level 2 --seconds 30
    conda run -n drones python tier_b/viz_lidar.py --model runs/ppo/seed0/final_model.zip
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from tier_b.envs.lidar import MAX_RANGE, N_RAYS, RAY_DIRS  # noqa: E402
from tier_b.envs.pirouette_aviary import PirouetteAviary  # noqa: E402

COLOR_HIT = [1.0, 0.15, 0.15]
COLOR_MISS = [0.55, 0.55, 0.55]


def _load_policy(path: str):
    """Автовизначення алгоритму за назвою шляху; повертає predict-функцію."""
    from stable_baselines3 import PPO, SAC

    cls = SAC if "sac" in path.lower() else PPO
    model = cls.load(path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--model", type=str, default=None,
                    help="шлях до SB3 .zip; без нього — завис на c=g")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    import pybullet as p

    env = PirouetteAviary(level=args.level, master_seed=args.master_seed, gui=True)
    obs, _ = env.reset()

    # без прев'ю-панелей (патерн demo.py) — лише головний в'юпорт
    for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                 p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                 p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
        p.configureDebugVisualizer(flag, 0, physicsClientId=env.CLIENT)

    predict = _load_policy(args.model) if args.model else \
        (lambda _obs: np.zeros((1, 4), dtype=np.float32))

    line_ids = [-1] * N_RAYS
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        action = np.asarray(predict(obs), dtype=np.float32).reshape(1, 4)
        obs, _r, term, trunc, info = env.step(action)

        pos = env.pos[0]
        rot = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
        world = RAY_DIRS @ rot.T
        lidar = obs[13:13 + N_RAYS]
        for i in range(N_RAYS):
            frac = float(lidar[i])
            end = pos + world[i] * MAX_RANGE * frac
            hit = frac < 1.0
            kwargs = dict(lineColorRGB=COLOR_HIT if hit else COLOR_MISS,
                          lineWidth=2 if hit else 1, physicsClientId=env.CLIENT)
            if line_ids[i] >= 0:
                kwargs["replaceItemUniqueId"] = line_ids[i]
            line_ids[i] = p.addUserDebugLine(pos, end, **kwargs)

        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=-45,
                                     cameraPitch=-25, cameraTargetPosition=pos,
                                     physicsClientId=env.CLIENT)
        time.sleep(1.0 / env.CTRL_FREQ)
        if term or trunc:
            print(f"епізод: success={info['success']} collision={info['collision']} "
                  f"jam={info['jam']} gates={info['gates_passed']}")
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
