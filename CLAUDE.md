# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A kit for an RL hackathon. Teams write a tabular Q-learning agent that
flies a drone from A to B through a seeded forest. The kit supplies the
environment, an A* oracle to score against, visualisers, and a student scaffold.

Pure Python 3.11 + NumPy + Matplotlib. The core is **tabular** — deliberately no
neural nets, no GPU. `torch`/`gymnasium` belong only to optional Tier 3 and must
never become core dependencies.

Build status: phases 0–6 complete and tested. Phase 7 (Webots) partial — the
bridge logic is tested, the simulator is not installed. Phase 8 (packaging) not
started. See `README.md` for commands.

## Tier layout

| folder | track | stack |
|---|---|---|
| `tier_d/` | the base **2D** kit this file mostly describes: grid MDP, A* oracle, tabular Q-learning, APF foil, 2D LIDAR viz, scoring, scaffold | main venv (numpy/matplotlib) |
| `tier_a/` | 6DOF attitude control along a reference path; ends in the measured fixed-thrust physics veto | main venv (+optional torch for DQN) |
| `tier_b/` | CTBR pirouette in gym-pybullet-drones, PPO vs SAC | conda env `drones` only |
| `tier_c/` | realistic FPV drone in Blender: A* + APF + kinematic autopilot; students edit only `solution.py` | Blender (bundled python) |

Each tier folder has a `CONCEPT.md` with the task concept and open work.

AI-assistant prompts for the event live in `prompts/` (universal protocol +
per-tier cards).

## Commands

```bash
source .venv/bin/activate
python tier_d/scaffold/validate_team.py    # self-check on PUBLIC seeds (fill team_solution.py first)
```

## The six invariants

Violating one yields a kit that **still runs, still scores, and is silently
wrong**.

1. **Reproducibility.** `tier_d/env/forest.py` ports `mulberry32` exactly. Same
   `(n, seed)` → identical forest, in Python *and* in `tier_d/viz/field_scanner.html`.
   Changing the RNG invalidates every cached `L*`.

2. **State = experience, not map.** The agent's state is a bare cell index.
   The agent never touches `env.trees` or `env.blocked` — enforced by
   a `BlindEnv` wrapper that raises. If the agent can
   see where trees are, it is doing cartography, not RL.

3. **One collision radius.** `tier_d/env/occupancy.py` is the *only* place `TREE_R` is
   applied. `tier_d/env/gridworld.py` and `tier_d/oracle/astar.py` both call
   `occupancy_grid()`; neither reimplements it, and both apply the same
   no-corner-cutting rule. This is what makes `efficiency = L*/L ≤ 1` hold.

4. **Shaping preserves the optimum.** Potential-based only:
   `F = γ·Φ(s′) − Φ(s)`, `Φ(s) = −dist(s, goal)`, **and `Φ(terminal) = 0`**. That
   last clause is not cosmetic — without it the Ng–Harada–Russell guarantee is
   void at the collision terminal. Tested two ways: `Σ F` around a closed loop is
   zero, and shaped/unshaped agents both converge to exactly `L*`.

5. **Hidden ≠ public.** `HIDDEN_SEEDS` are never used in training or tuning.
   In the handout, `tier_d/scoring/seeds.py` exposes an empty `HIDDEN_SEEDS`
   tuple; only `PUBLIC_SEEDS` are available.

6. **`V(s)` is a navigation function.** One minimum, at the goal, no local traps.
   Verified against `tier_d/oracle/value_iteration.dijkstra_cost_to_go` via
   `has_single_minimum`, not asserted.

## Things that are easy to get wrong here

**Terminal vs truncated.** `env.step` returns `info["collision"]`,
`info["goal"]`, `info["truncated"]`. Bootstrap through truncation; do *not*
bootstrap through a real terminal, and zero `Φ` only at real terminals.
Conflating the two is the classic silent value-leak.

**The goal's Q row is never updated** (it is terminal), so it stays at 0.
Anything reading `argmax Q[goal]` gets action 0. This bit `tier_d/webots/bridge.py`,
which now homes on the goal point once inside the goal cell.

**`max_a Q` cannot see the collision penalty**, because the best action never
walks into a tree. Obstacle cells therefore stay at 0 in the value surface.
`tier_d.scaffold.qtools.experienced_obstacle_value` reads the pain back out of
`Q(neighbour, action-into-tree)` so peaks reflect collisions actually suffered —
untouched trees stay flat. Do not "fix" this by stamping `R_COLLISION` on every
tree; that draws a pretty surface that owes nothing to learning.

**Sample efficiency must be measured on the greedy policy**, not on exploration
episodes. With `ε` floored at 0.05 the exploring agent keeps taking random
detours long after it has learned. The harness right-censors at the budget; an
uncensored mean is `inf` and erases all differences.

**`string_pull` is cosmetic.** `L*` is the grid A* cost. Scoring against a
smoothed path would make `η = 1.0` unreachable by construction.

## Experiments: wind ablation

`tier_d/env_wind/` (+ `tier_d/scaffold/starter_wind.ipynb`) form a controlled
comparison: same agent, same hyperparameters, but deterministic
baseline vs 5% stochastic action replacement (simulated wind gusts).

Expected outcome: baseline learns faster (memorizes the path), wind learns slower
but reaches a *robust* policy — the cost of resilience in RL. This teaches why
real robotics (with uncertainty) needs different strategies than our idealized
deterministic grid.

## Environment constants

All in `tier_d/env/constants.py`.

| | |
|---|---|
| Domain | 10×10, continuous |
| Grid | 25×25, `CELL = 0.4` |
| Start A / Goal B | `(1,1)` → cell `(2,2)` / `(9,9)` → cell `(22,22)` |
| Actions | 8 directions, no corner-cutting |
| Rewards | `−1` step, `−100` collision (terminal), `+100` goal (terminal) |
| `TREE_R` | `0.40` — a cell is occupied iff its **centre** is within `TREE_R` of a tree |
| Forest | `BORDER_MARGIN 0.9`, `AB_CLEARANCE 1.1`, `MIN_TREE_SEP 0.85` |

Two consequences worth holding in mind. With `γ = 0.99`, wandering forever is
worth `−1/(1−γ) = −100` — exactly the collision penalty, so suicide is only
barely irrational; the `+100` goal breaks the tie. And because `TREE_R = CELL`
with `MIN_TREE_SEP = 0.85`, the sampler cannot generate a sealed wall, so no
generated seed is ever infeasible (`is_feasible` still correctly rejects a
hand-built palisade).

## Phase gating

The build plan's nine phases are strictly ordered and each ends with a passing
check. Don't start phase N until N−1 is green.

```
0 toolchain ─▶ 1 env ─▶ 2 oracle ─┐
                                  ▼
        3 Q-learning + shaping ───┤
                                  ▼
   4 viz     5 scoring   ◀────────┤
      │          │                │
      └────┬─────┘                │
           ▼                      │
    6 scaffold ◀──────────────────┘
           ▼
    7 Webots bridge   (partial: no simulator here)
           ▼
    8 packaging       (not started)
```

## Related

`/Users/drl-6/Desktop/RunSky/` is a predecessor project (PyBullet + PPO, a
different concept). `tier_d/oracle/astar.py` is ported from its `src/utils.py`. Nothing
else there is reusable; it has no mulberry32, no APF, no tabular Q-learning.

---

## Стисло (українською)

Тут лише те, що не можна порушувати.

**Шість інваріантів:**

1. **Відтворюваність.** Seeded `mulberry32`; однакові `seed`+`N` → байт-у-байт
   однаковий ліс. Однаково в Python і в JS (`tier_d/viz/field_scanner.html`).
2. **Стан = досвід, не мапа.** Координати дерев **ніколи** не в стані агента.
   Лише клітинка. Інакше це картографування, а не RL.
3. **Однаковий радіус колізії.** Одна константа `TREE_R` у `tier_d/env/occupancy.py`,
   яку викликають і середовище, і оракул. Тоді `efficiency = L*/L ≤ 1`.
4. **Shaping зберігає оптимум.** Лише потенціал-орієнтований:
   `F = γ·Φ(s′) − Φ(s)`, `Φ = −відстань до цілі`, **і `Φ(термінал) = 0`**.
5. **Hidden ≠ public.** Приховані seed-и ніколи не в тренуванні чи тюнінгу.
6. **`V(s)` — navigation function.** Один мінімум, у цілі, без локальних пасток.

**Порядок фаз суворий.** Кожна фаза завершується перевіркою. Не переходь до фази
N, доки тести фази N−1 не зелені.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%). Format flags (-c, -l, -L, -o, -Z) run raw.
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->