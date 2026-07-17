#!/usr/bin/env python3
"""
Self-check for Tier-A teams before final submission.
Runs your DQN attitude controller on PUBLIC_SEEDS and shows where you stand.

    python tier_a/scaffold/validate_team_attitude.py           # full config (1200 episodes)
    python tier_a/scaffold/validate_team_attitude.py --smoke   # fast sanity check (150 episodes)

Output is a table: seed, T*, tracking_rmse, success, collision/departed.
This is a genuinely hard from-scratch 6-DOF control problem -- do not expect
every seed to reach the goal even with a well-trained agent; a falling
tracking_rmse and a rising survival time are meaningful progress signals in
their own right (see env_attitude/README.md).
"""

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np

from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.constants import N_TREES_ATTITUDE
from tier_d.scoring.seeds import PUBLIC_SEEDS

try:
    from tier_a.scaffold.team_solution_attitude import train_attitude, greedy_rollout_attitude
    import inspect
    if "NotImplementedError" in inspect.getsource(train_attitude):
        raise ImportError("team_solution_attitude.py is still a stub")
    print("Imported trainer from tier_a.scaffold.team_solution_attitude (your code)")
except ImportError:
    try:
        from tier_a.admin.agent.qnet_attitude import train_attitude, greedy_rollout_attitude
        print("Imported trainer from tier_a.admin.agent.qnet_attitude (reference)")
    except ImportError:
        print("Заповніть tier_a/scaffold/team_solution_attitude.py: перенесіть туди "
              "train_attitude і greedy_rollout_attitude зі свого ноутбука "
              "(starter_attitude.ipynb), потім запустіть цю перевірку ще раз.")
        sys.exit(1)


def validate(seed: int, n_trees: int, episodes: int) -> dict:
    env = AttitudeEnv(n_trees=n_trees, seed=seed)
    qnet, _ = train_attitude(env, episodes=episodes, seed=seed, eval_every=0)
    roll = greedy_rollout_attitude(env, qnet)
    return {"seed": seed, "T_ref": env.reference.T_total, **roll}


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-A team validation on PUBLIC_SEEDS")
    p.add_argument("--smoke", action="store_true", help="fast sanity check (150 episodes, 3 trees)")
    p.add_argument("--n-trees", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    args = p.parse_args()

    n_trees = args.n_trees or (3 if args.smoke else N_TREES_ATTITUDE)
    episodes = args.episodes or (150 if args.smoke else 1200)

    print("\n" + "=" * 72)
    print(f"AI SkyRun Tier-A · Team Validation on PUBLIC_SEEDS "
          f"({'smoke' if args.smoke else 'full'} config: {episodes} episodes, {n_trees} trees)")
    print("=" * 72 + "\n")

    results = [validate(s, n_trees, episodes) for s in PUBLIC_SEEDS]

    print(f"{'Seed':>5} {'T*':>8} {'RMSE':>8} {'Steps':>7}  {'OK':>3}  {'Coll':>5} {'Dep':>4}")
    print("-" * 60)
    success_count = 0
    rmses = []
    for r in results:
        ok_str = "OK" if r["success"] else "--"
        print(f"{r['seed']:>5} {r['T_ref']:>8.3f} {r['tracking_rmse']:>8.3f} {r['steps']:>7}  "
              f"{ok_str:>3}  {int(r['collision']):>5} {int(r['departed']):>4}")
        rmses.append(r["tracking_rmse"])
        success_count += r["success"]

    print("-" * 60)
    print(f"Mean tracking RMSE: {np.mean(rmses):.3f}")
    print(f"Success rate: {success_count}/{len(PUBLIC_SEEDS)} = {100 * success_count / len(PUBLIC_SEEDS):.0f}%")
    print()
    print("=" * 72)
    if args.smoke:
        print("Smoke check complete -- this budget is NOT expected to converge.")
        print("Run without --smoke for the full (1200-episode) config before submitting.")
    elif success_count == len(PUBLIC_SEEDS):
        print("ALL PUBLIC SEEDS PASSED! Ready for final on HIDDEN_SEEDS.")
    else:
        print(f"{success_count}/{len(PUBLIC_SEEDS)} passed. A falling tracking_rmse and longer "
              f"survival time are real progress even without full-lap success.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
