#!/usr/bin/env python3
"""
Self-check for teams before final submission.
Runs your Q-learning on PUBLIC_SEEDS and shows where you stand.

    python tier_d/scaffold/validate_team.py

Output is a simple table: seed, L*, your_L, efficiency, success, collisions.
If all greens, you're ready for final (which runs on HIDDEN_SEEDS).
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
from tier_d.env.gridworld import GridWorld
from tier_d.env.constants import N_ACTIONS
from tier_d.oracle.astar import optimal_length
from tier_d.scoring.seeds import PUBLIC_SEEDS

# Import your trainer: first the team's exported solution, then (in the
# organizer copy only) the reference agent from admin/.
try:
    from tier_d.scaffold.team_solution import train, greedy_rollout
    # Fail fast if the stubs are still unfilled (they raise NotImplementedError).
    import inspect
    if "NotImplementedError" in inspect.getsource(train):
        raise ImportError("team_solution.py is still a stub")
    print("✓ Imported trainer from tier_d.scaffold.team_solution (your code)")
    IS_REFERENCE = False
except ImportError:
    try:
        from tier_d.admin.agent.qlearning import train, greedy_rollout
        print("✓ Imported trainer from tier_d.admin.agent.qlearning (reference)")
        IS_REFERENCE = True
    except ImportError:
        print("⚠ Заповніть tier_d/scaffold/team_solution.py: перенесіть туди "
              "train і greedy_rollout зі свого ноутбука (starter.ipynb), "
              "потім запустіть цю перевірку ще раз.")
        sys.exit(1)

def validate(seed, n_trees=16, alpha=0.15, gamma=0.99, episodes=3000):
    """Train on one seed and return (L_star, L, success, collisions, efficiency)."""
    env = GridWorld(n_trees, seed)
    lstar = optimal_length(env.trees)

    Q, _ = train(env, episodes=episodes, seed=seed)
    roll = greedy_rollout(GridWorld(n_trees, seed), Q)

    if roll["success"]:
        L = roll["length"]
        eff = lstar / L
        collisions = 0  # greedy_rollout doesn't crash in final policy
    else:
        L = np.nan
        eff = 0.0
        collisions = 1

    return lstar, L, roll["success"], collisions, eff

def main():
    n_trees = 16
    print("\n" + "="*70)
    print("AI SkyRun · Team Validation on PUBLIC_SEEDS")
    print("="*70)
    print()

    results = []
    for seed in PUBLIC_SEEDS:
        lstar, L, success, collisions, eff = validate(seed, n_trees=n_trees)
        results.append({
            "seed": seed,
            "lstar": lstar,
            "L": L,
            "success": success,
            "collisions": collisions,
            "eff": eff,
        })

    # Print table
    print(f"{'Seed':>5} {'L*':>8} {'L':>8} {'L*/L':>8}  {'OK':>3}  {'Coll':>4}")
    print("-" * 50)

    eff_list = []
    success_count = 0
    for r in results:
        eff_str = f"{r['eff']:.3f}" if r['success'] else "—"
        ok_str = "✓" if r["success"] else "✗"
        print(f"{r['seed']:>5} {r['lstar']:>8.3f} {r['L']:>8.3f} {eff_str:>8}  {ok_str:>3}  {r['collisions']:>4}")
        if r["success"]:
            eff_list.append(r["eff"])
            success_count += 1

    print("-" * 50)
    if eff_list:
        avg_eff = np.mean(eff_list)
        print(f"Average efficiency (successes only): {avg_eff:.3f}")
        print(f"Success rate: {success_count}/{len(PUBLIC_SEEDS)} = {100*success_count/len(PUBLIC_SEEDS):.0f}%")
    else:
        print("⚠ No successes yet. Check GAMEPLAY_GUIDE.md for debugging.")

    print()
    print("="*70)
    if success_count == len(PUBLIC_SEEDS):
        print("✓ ALL PUBLIC SEEDS PASSED! Ready for final on HIDDEN_SEEDS.")
    elif success_count >= len(PUBLIC_SEEDS) * 0.7:
        print(f"✓ {success_count}/{len(PUBLIC_SEEDS)} seeds passed. Good progress!")
    else:
        print(f"⚠ Only {success_count}/{len(PUBLIC_SEEDS)} passed. Review GAMEPLAY_GUIDE.md.")
    print("="*70)
    print()

if __name__ == "__main__":
    main()
