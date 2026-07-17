#!/usr/bin/env python3
"""Експорт rollout-у в JSON для Blender: позиція + кватерніон (XYZW pybullet →
WXYZ Blender, документовано в обох файлах) + 34 кінцеві точки LIDAR на кадр.

    conda run -n drones python tier_b/export_blender.py \
        --model tier_b/runs/ppo/seed0/final_model.zip --level 2 \
        --out tier_b/exports/tier_b_traj.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from tier_b.envs.lidar import MAX_RANGE, N_RAYS, RAY_DIRS  # noqa: E402


def export(model_path: str | None, level: int, master_seed: int,
           max_steps: int = 600) -> dict:
    import pybullet as p

    from tier_b.envs.pirouette_aviary import PirouetteAviary

    if model_path:
        from stable_baselines3 import PPO, SAC

        cls = SAC if "sac" in model_path.lower() else PPO
        model = cls.load(model_path, device="cpu")
        predict = lambda o: model.predict(o, deterministic=True)[0]  # noqa: E731
    else:
        predict = lambda o: np.zeros((1, 4), dtype=np.float32)  # noqa: E731

    env = PirouetteAviary(level=level, master_seed=master_seed)
    obs, _ = env.reset()

    frames = []
    info = {}
    for i in range(max_steps):
        action = np.asarray(predict(obs), dtype=np.float32).reshape(1, 4)
        obs, _r, term, trunc, info = env.step(action)

        pos = env.pos[0]
        qx, qy, qz, qw = env.quat[0]  # pybullet: XYZW
        rot = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
        lidar = obs[13:13 + N_RAYS]
        world = RAY_DIRS @ rot.T
        rays = [{"to": (pos + world[k] * MAX_RANGE * float(lidar[k])).round(4).tolist(),
                 "hit": bool(lidar[k] < 1.0)} for k in range(N_RAYS)]
        frames.append({"t": round(i / env.CTRL_FREQ, 4),
                       "pos": pos.round(4).tolist(),
                       "quat_wxyz": [round(float(v), 6) for v in (qw, qx, qy, qz)],
                       "lidar": rays})
        if term or trunc:
            break

    data = {
        "fps": int(env.CTRL_FREQ),
        "ellipsoid": list(env.semi_axes),
        "lidar_range": MAX_RANGE,
        "trees": [{"x": float(x), "y": float(y), "r": float(r),
                   "h": float(env.forest.tree_height)}
                  for x, y, r in env.forest.trees],
        "waypoints": env.tracker.waypoints.round(4).tolist(),
        "outcome": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else int(v))
                    for k, v in info.items()},
        "frames": frames,
    }
    env.close()
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--master-seed", type=int, default=900001)
    ap.add_argument("--out", type=str, default="tier_b/exports/tier_b_traj.json")
    args = ap.parse_args()

    data = export(args.model, args.level, args.master_seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"записано {out}: {len(data['frames'])} кадрів, "
          f"outcome={data['outcome']}")


if __name__ == "__main__":
    main()
