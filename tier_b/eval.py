#!/usr/bin/env python3
"""Детермінована оцінка чекпоінта на HELD-OUT сідах (дисципліна інваріанта 5:
оцінні ліси ніколи не бачені у тренуванні — master_seed зсунуто на 900000).

    conda run -n drones python tier_b/eval.py \
        --model tier_b/runs/ppo/seed0/final_model.zip --level 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from tier_b import load_config  # noqa: E402


def rollout_episode(env, predict, max_steps: int = 600) -> dict:
    obs, _ = env.reset()
    total_r, steps = 0.0, 0
    info = {}
    for _ in range(max_steps):
        action = np.asarray(predict(obs), dtype=np.float32).reshape(1, 4)
        obs, r, term, trunc, info = env.step(action)
        total_r += float(r)
        steps += 1
        if term or trunc:
            break
    return {"return": total_r, "steps": steps,
            "success": bool(info.get("success")),
            "collision": bool(info.get("collision")),
            "jam": bool(info.get("jam")),
            "out_of_bounds": bool(info.get("out_of_bounds")),
            "gates_passed": int(info.get("gates_passed", 0))}


def evaluate(model_path: str, level: int, n_episodes: int, seed_offset: int,
             algo: str | None = None) -> list[dict]:
    from stable_baselines3 import PPO, SAC

    from tier_b.envs.pirouette_aviary import PirouetteAviary

    algo = algo or ("sac" if "sac" in model_path.lower() else "ppo")
    model = (SAC if algo == "sac" else PPO).load(model_path, device="cpu")

    env = PirouetteAviary(level=level, master_seed=seed_offset)
    predict = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
    results = [rollout_episode(env, predict) for _ in range(n_episodes)]
    env.close()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config()
    n = args.episodes or cfg["eval"]["n_episodes"]
    offset = cfg["eval"]["master_seed_offset"]

    results = evaluate(args.model, args.level, n, offset)
    n_s = sum(r["success"] for r in results)
    n_c = sum(r["collision"] for r in results)
    n_j = sum(r["jam"] for r in results)
    print(f"L{args.level}: success {n_s}/{n}  collision {n_c}  jam {n_j}  "
          f"oob {sum(r['out_of_bounds'] for r in results)}  "
          f"mean_gates {np.mean([r['gates_passed'] for r in results]):.2f}")

    out = args.out or (str(pathlib.Path(args.model).parent /
                           f"eval_L{args.level}.json"))
    with open(out, "w") as f:
        json.dump({"model": args.model, "level": args.level,
                   "episodes": results}, f, indent=2)
    print(f"записано {out}")


if __name__ == "__main__":
    main()
