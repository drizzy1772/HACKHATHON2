#!/usr/bin/env python3
"""Самоперевірка команди Tier-B перед здачею (conda-only).

Проганяє ваш чекпоінт на PUBLIC_SEEDS — тих самих held-out сідах, за
схемою яких (але з іншими значеннями) відбудеться фінальний залік.

    conda run -n drones python tier_b/scaffold/validate_team_b.py \
        --model tier_b/runs/sac/seed0/final_model.zip --level 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

try:
    import stable_baselines3  # noqa: F401
except ImportError:
    print("Потрібне conda-середовище `drones` (stable_baselines3 не знайдено).\n"
          "Запуск:  conda run -n drones python tier_b/scaffold/validate_team_b.py --model ...")
    sys.exit(1)

import numpy as np  # noqa: E402

from tier_b.eval import rollout_episode  # noqa: E402
from tier_b.scoring.seeds import PUBLIC_SEEDS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="шлях до вашого final_model.zip")
    ap.add_argument("--algo", choices=("ppo", "sac"), default=None)
    ap.add_argument("--level", type=int, default=5)
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC

    from tier_b.envs.pirouette_aviary import PirouetteAviary

    algo = args.algo or ("sac" if "sac" in args.model.lower() else "ppo")
    model = (SAC if algo == "sac" else PPO).load(args.model, device="cpu")
    predict = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731

    print("\n" + "=" * 66)
    print(f"AI SkyRun · Tier-B validation on PUBLIC_SEEDS  [{algo}, L{args.level}]")
    print("=" * 66)
    print(f"{'Seed':>8} {'OK':>3} {'Gates':>5} {'Steps':>6} {'Return':>9}  Причина")
    print("-" * 66)

    results = []
    for seed in PUBLIC_SEEDS:
        env = PirouetteAviary(level=args.level, master_seed=seed)
        r = rollout_episode(env, predict)
        env.close()
        results.append(r)
        reason = ("фініш" if r["success"] else
                  "колізія" if r["collision"] else
                  "jam" if r["jam"] else
                  "out-of-bounds" if r["out_of_bounds"] else "таймаут")
        ok = "✓" if r["success"] else "✗"
        print(f"{seed:>8} {ok:>3} {r['gates_passed']:>5} {r['steps']:>6} "
              f"{r['return']:>9.1f}  {reason}")

    n = len(results)
    n_ok = sum(r["success"] for r in results)
    print("-" * 66)
    print(f"success-rate {n_ok}/{n}   mean gates "
          f"{np.mean([r['gates_passed'] for r in results]):.2f}   mean return "
          f"{np.mean([r['return'] for r in results]):.1f}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
