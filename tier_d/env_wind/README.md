# env_wind — Stochastic Environment

Wind-augmented version of the base environment. Identical to `tier_d/env/` except:

- **5% of actions are replaced** with uniformly random actions per step
- This simulates wind gusts or motor noise
- **Not deterministic anymore** — same training seed can produce different episodes

## Usage

```python
from env_wind.gridworld import GridWorldWind

env = GridWorldWind(n_trees=16, seed=0, wind_prob=0.05)
s = env.reset()
s_next, r, done, info = env.step(action)  # 5% chance of random action

if info["wind"]:
    print("Wind gust occurred!")
```

## Comparison with baseline

| Aspect | `env.GridWorld` | `env_wind.GridWorldWind` |
|--------|---|---|
| Transitions | Deterministic | 95% as requested, 5% random |
| State space | Same | Same |
| Reward structure | Same | Same |
| Learning curve | Smooth, fast | Noisy, slower |
| Learned policy | Memorizes path | Robust to perturbations |
| Real-world relevance | Idealized | More realistic |

## For experiments

See `tier_d/experiments/compare_wind.py` for a head-to-head comparison:

```bash
python tier_d/experiments/compare_wind.py --seed 0 --wind-prob 0.05
```

Output: side-by-side learning curves + statistics showing the trade-off between
deterministic memorization and stochastic robustness.

## Caveats

- **The six invariants still hold** in wind version (state = cell index, no tree access)
- **A* oracle sees the stochasticity** — if you recompute L* for wind, it will increase
- **Action override is uniform random**, not biased (no preferred direction)
- Training longer generally helps — more episodes to learn robust policies

## For students (Tier 2+)

This is an optional, harder track. Start with baseline (`env.GridWorld`), then try wind:

1. Train on baseline, note the learning curve
2. Train on wind with same hyperparameters
3. Run `compare_wind.py` to see the difference
4. Adjust hyperparameters for wind (e.g., increase EPISODES)
5. Journal your observations for the writeup
