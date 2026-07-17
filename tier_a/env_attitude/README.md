# env_attitude — Tier-A: Flight Attitude Control

A separate, self-contained track. The 2D forest and its A* oracle path are
reused as-is; the new problem is *flying* that fixed path with a 6-DOF rigid
body by controlling roll/pitch/yaw torques -- not planning which cell to
visit next. A stochastic wind disturbance is available as an opt-in harder
variant (see "Wind: on or off?" below), off by default.

This is the flight-control gap the base kit already flags but never fills:
`tier_d/webots/bridge.py` (Phase 7) converts a policy into 2D waypoints and a target
velocity, then explicitly leaves attitude/altitude to *"an existing PID"*
that doesn't exist in this repo. Tier-A is that missing controller, built as
an RL problem instead of a hand-tuned PID.

## What's different from the base kit

| | base kit (`tier_d/env/`) | Tier-A (`env_attitude/`) |
|---|---|---|
| State | discrete cell index | continuous 10D tracking-error vector |
| Action | 8 grid directions | 27 discretized torque combinations |
| Dynamics | teleport between cells | 6-DOF rigid-body integration (`dynamics.py`) |
| Path | agent must find it | pre-baked by `oracle.astar`, agent must *track* it |
| Learner | tabular Q-table | tiny torch DQN (`agent/qnet_attitude.py`) or tabular Q over LIDAR (`agent/qtable_attitude.py`) |
| Disturbance | none by default (`tier_d/env_wind`'s 5% action override is opt-in) | none by default; pass `wind_prob=WIND_PROB` (5%) for the harder variant |

`AttitudeEnv` is deterministic by default (`wind_prob=0.0`), the same relationship
`env.GridWorld` has to `env_wind.GridWorldWind`: get the deterministic case
training reliably first, then opt into gusts as a harder variant, not the
default. Wind was dropped to the default's confound list on purpose --
diagnosing a long-horizon control problem is hard enough without also
averaging over stochastic transitions; see "Wind: on or off?" below.

## Usage

```python
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.admin.agent.qnet_attitude import train_attitude, greedy_rollout_attitude

env = AttitudeEnv(n_trees=8, seed=0)
qnet, hist = train_attitude(env, episodes=1200, seed=0)
roll = greedy_rollout_attitude(env, qnet)
print(roll["success"], roll["tracking_rmse"])
```

Scoring: `python -m tier_a.admin.scoring.precompute_attitude` then
`python -m tier_a.admin.scoring.harness_attitude --public`. Self-check:
`python tier_a/scaffold/validate_team_attitude.py [--smoke]`.

The tabular + 3D-LIDAR variant (see below) runs on a hand-built local
scenario instead:

```python
from tier_a.env_attitude.scenarios import build_local_gauntlet_env
from tier_a.admin.agent.qtable_attitude import train_tabular_attitude, greedy_rollout_tabular

env = build_local_gauntlet_env(seed=0)   # a tight two-tree gate, ~3m corridor
Q, hist = train_tabular_attitude(env, episodes=8000, seed=0)
roll = greedy_rollout_tabular(env, Q)
print(roll["success"], roll["mean_rate"], roll["min_lidar"])
```

## The six invariants, adapted

Same discipline as the base kit's `tests/test_invariants.py`, applied to a
continuous track -- see `tests/test_invariants_attitude.py`:

1. **Reproducibility.** The reference trajectory and the wind-gust sequence
   are both deterministic from `(n_trees, seed)`.
2. **State = experience, not map.** The agent's observation is *only*
   tracking error relative to the reference (`e_lat, e_alt, e_φ, e_θ, e_ψ, p,
   q, r, κ`) -- never the raw `(x, y, z)` position or tree coordinates.
3. **One collision radius.** `env_attitude.constants.TREE_R` is *imported*
   from `tier_d.env.constants`, never re-declared.
4. **Shaping preserves the optimum.** Potential-based only, `Φ(terminal)=0`,
   adapted to a continuous tracking-error potential.
5. **Hidden ≠ public.** Reuses `tier_d.scoring.seeds.PUBLIC_SEEDS`/`HIDDEN_SEEDS`
   directly -- no new seed lists to keep in sync.
6. **The potential is a true error surface.** Zero only at zero tracking
   error, strictly negative elsewhere.

Plus one Tier-A-specific rule: **`torch` never appears under `env_attitude/`**
-- it is confined to `agent/qnet_attitude.py`. The base kit's `requirements.txt`
is untouched (torch stays commented out, "Tier 3+ only, never core"); if you
work on this track, add `torch` to your own `requirements_team.txt`, exactly
as Tier 3's LIDAR net already asks students to do.

## An honest note on difficulty

This is a genuinely hard control problem attempted from scratch: full 6-DOF
rigid-body dynamics, a discretized-action DQN, and a wind disturbance, all in
one day. Don't expect the *smoke* config (150 episodes) to converge --
its only job is to catch crashes and shape errors in seconds, the same way
the base kit's test suite stays fast. The *full* config (1200 episodes, run
interactively, never inside the automated test suite) does show real
learning: episode survival time roughly doubles over training and the
lexicographic scoreboard (`tier_a/admin/scoring/harness_attitude.py`) reliably
discriminates a trained agent from an untrained one on collision/departure
rate -- but full-lap success on every seed is not guaranteed at this budget.
Treat a falling `tracking_rmse` and a longer survival time as real progress in
their own right, the same way the base kit's APF track (`tier_d/apf/`) reports its
34% stall rate honestly rather than dressing it up.

If you want to push this further: a target network was already necessary
(a vanilla online DQN measurably underperformed a random baseline here --
see `agent/qnet_attitude.py`'s `dqn_update` docstring); double DQN, n-step
returns, or a curriculum over `n_trees`/`CRUISE_SPEED` are the next levers
worth trying before reaching for a heavier algorithm.

## 3D LIDAR + tabular Q-learning (`lidar3d.py`, `agent/qtable_attitude.py`)

A second, independent learner: real tabular Q-learning (a numpy Q-table, no
neural net at all) over a state discretized from a **body-frame 3D LIDAR**
(`env_attitude/lidar3d.py`) instead of the continuous DQN's analytic error
vector. The ray ring is fixed in the *body* frame and rotated into world
space by the current `(φ,θ,ψ)` every scan (`scan3d`), so banking the drone
banks the whole sensor cone with it -- a real ray/finite-cylinder
intersection against the tree trunks, not a 2D circle test. This is what the
first review's LIDAR section said hadn't been attempted; it now has.

The discretized state (`agent/qtable_attitude.discretize`, ~11k entries)
folds in attitude error, angular-rate magnitude, and the nearest LIDAR
return's distance + bearing. Two behavioural objectives are added as direct
reward terms (not potential-based shaping, since they're real preferences,
not a policy-invariant speedup): a **stability** penalty on angular-rate
magnitude (targets "don't tumble") and a **proximity** penalty when the
nearest LIDAR return is inside a safety radius (targets "react to what's
nearby"). Being tabular, training is dramatically cheaper than the DQN: ~30k
episodes run in well under two minutes with no `torch` involved anywhere in
`env_attitude/` or `agent/qtable_attitude.py`.

### A real bug this surfaced

Building a short, focused training scenario (`env_attitude.scenarios`, a
"gauntlet": two hand-placed trees straddling a corridor with only
`GAUNTLET_CLEARANCE=0.6m` of clearance to each trunk) surfaced a genuine
sign error in `reference_trajectory.py`: `theta_ref` (the reference pitch
during acceleration/deceleration) had the wrong sign relative to
`dynamics.rotation_matrix`'s actual convention. "Perfectly tracking
`theta_ref`" was silently pitching *away* from the direction of travel on
every accel/decel leg -- confirmed by a hand-tuned PD controller flying
*backwards* before the fix, and correctly forwards after. This was
presumably also suppressing full-lap success on the main DQN track above,
though its effect there is entangled with the long-horizon difficulty
already documented. Fixed now; see the comment at `theta_ref`'s definition.

### What the tabular gauntlet run actually shows

`env_attitude/scenarios.build_local_gauntlet_env` carves out just the ~3 m
segment around the gate (not the full ~11 m lap) so the RL horizon matches
the *local* skill being tested, at a constant cruise speed (the segment
represents mid-flight, not launch). Training this tabular agent is **noisy,
not monotonic** -- of 61 checkpoints sampled every 500 episodes across a
30,000-episode run, 3 succeeded cleanly, 16 collided, 42 drifted off and
timed out, with no clean trend line (see
`tier_a/admin/reports/tier_a_lidar_training_progress.png`). That noise is reported
honestly, not hidden.

But the checkpoints that *do* succeed are real and worth seeing:
`tier_a/admin/reports/tier_a_lidar_progressive_training.png` shows four flights through
the same gate -- an early checkpoint tumbling and departing in 36 steps, a
mid-training checkpoint flying purposefully but clipping a trunk, a later
checkpoint (episode 14,000) banking cleanly between both trees with smooth,
low-rate gizmo orientation the whole way through, and the hand-tuned PD
controller (no learning at all) as the reference ceiling -- proof the
gap is physically threadable given a precise enough controller. The gap
between "sometimes works" (tabular) and "always works" (hand-tuned) is, as
with the DQN, a training-budget/algorithm gap, not a physics one: 3-bin
discretization of a continuous attitude error is a much cruder controller
than continuous PD feedback, and that expressivity loss is the main suspect
for why convergence isn't reliable yet at this budget.

### Wind: on or off?

Tested directly rather than assumed: 8 independent training seeds at 6,000
episodes each, deterministic vs `wind_prob=0.05`, same gauntlet, same greedy
evaluation.

| | success | collision | departed | mean angular rate |
|---|---|---|---|---|
| no wind | 1 / 8 | 4 | 3 | 0.92 rad/s |
| wind 5% | 0 / 8 | 2 | 6 | 1.15 rad/s |

Removing wind gives a modestly more stable, slightly more successful policy
(lower mean rate, one clean success where wind produced none) -- consistent
with the hypothesis that stochastic transitions add variance a coarse
discretized Q-table already struggles to average out. But it is not a fix:
1/8 vs 0/8 is still mostly failing either way. The dominant bottleneck is the
3-bin attitude discretization's expressivity gap against continuous PD
feedback (above), not wind -- removing wind helps marginally at the margin,
it doesn't change the fundamental picture. Train the deterministic case
first (now the default); add `wind_prob=WIND_PROB` back only once that
converges reliably, exactly as `env_wind` is positioned relative to `env`
in the base kit.

## Hybrid: PD stabilization + a 9-state learned guidance layer (`agent/hybrid_attitude.py`)

The end-to-end tabular Q-learner above (~11k states) rarely beats a
hand-tuned PD controller on the same gauntlet -- the working theory was that
3-bin attitude discretization is just too coarse to reproduce continuous PD
precision. The direct test of that theory: keep PD for what it's already
provably good at (moment-to-moment stabilization), and shrink the *learned*
part down to a single decision a PD-only controller cannot make on its own
-- which way to bias the roll target when the 3D LIDAR senses a nearby
trunk. State is `(e_lat bin, LIDAR-min bin)` -- 3x3 = 9 states, 3 actions
(swerve left / none / swerve right), small enough that a few thousand
episodes should be more than sufficient to converge.

```python
from tier_a.env_attitude.scenarios import build_local_gauntlet_env
from tier_a.admin.agent.hybrid_attitude import train_hybrid_guidance, greedy_rollout_hybrid

env = build_local_gauntlet_env(seed=0)
Q, hist = train_hybrid_guidance(env, episodes=4000, seed=0)
roll = greedy_rollout_hybrid(env, Q)
```

**Result, tested honestly across 10 independent training seeds:** 1 clean
success, 7 collisions, 2 near-misses that drifted off course --
`tier_a/admin/reports/tier_a_hybrid_seeds.png`. When it works (seed 2), it matches the
hand-tuned PD almost exactly, banking cleanly through the same 0.6 m gate in
the same 189 steps -- `tier_a/admin/reports/tier_a_hybrid_three_approaches.png` puts all
three controllers (PD / end-to-end tabular / hybrid) side by side on the
identical gate.

So a 9-state table is *not* automatically reliable just because it's small.
Root cause, found by inspecting the "danger" bin's Q-values directly: the
0.6 m gap only stays inside any reasonable LIDAR danger threshold for a
handful of steps out of ~55-190 per episode, so even a tiny table gets very
few samples of exactly the states where the decision matters most --
Q-learning's usual answer to sparse-but-critical states (more episodes) is
undercut here by the *episode* itself ending (collision) before those states
accumulate enough visits to converge. Widening the danger-detection radius
(`_LMIN_EDGES`, triggers earlier in the approach) helped some (0/10 -> 1/10
in one comparison); it did not fix it outright. This is the same family of
finding as the DQN and pure-tabular sections above, now demonstrated on a
problem two orders of magnitude smaller than the original ~11k-state table
-- state-space size was a real contributor, but not the whole story.

## Online LIDAR-based replanning (`env_attitude/online_planner.py`)

Everything above still flies a trajectory planned once, offline, from full
ground-truth tree coordinates (`oracle.astar` over the seeded forest) before
the flight ever starts -- the 3D LIDAR only ever nudges *how carefully* the
drone tracks a path it already knows. This section closes that gap: the
drone no longer knows the forest in advance. It discovers it incrementally
via a new 2D, yaw-rotating LIDAR (`env_attitude/lidar2d.py`, reusing
`tier_d/env/lidar.py`'s ray-vs-circle math -- flying at fixed `FLY_Z`, height
doesn't matter for horizontal ranging), builds an occupancy belief
(`env_attitude/mapping.py`'s tri-state `BeliefMap`: unknown cells count as
optimistically passable, so the drone can start moving before it has scanned
everything), and periodically replans its route with the *same*
`oracle.astar.astar()` used everywhere else in this kit -- now given two
additive, defaulted `start`/`goal` parameters so it can plan from the
drone's current cell, not just from A.

```python
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.admin.agent.hybrid_attitude import pd_action

env = AttitudeEnv(online_lidar_planning=True, seed=0)
obs = env.reset()
for _ in range(1500):
    obs, r, done, info = env.step(pd_action(obs))
    if info["replanned"]:
        print("replanned at step", info)
    if done:
        break
```

This is a periodic **full** A* replan (every ~0.5s, or immediately if a
newly-discovered obstacle falls on the next few cells of the current path),
not a hand-rolled D*-Lite -- re-running A* over this kit's ~625-cell grid is
trivially cheap, so there is no performance case for incremental repair, and
the budget was better spent on the mapping/anchoring/tests below. This
mirrors this repo's own precedent for simplest-mechanism-that's-correct over
maximally sophisticated (the DQN target network, potential-based shaping):
see the module docstring in `online_planner.py` for the fuller argument.

**The discontinuity problem, and how far the fix actually goes.** Swapping
the reference mid-flight can jump `e_lat`/`e_psi` sharply, since the
tracking-error controllers only see the instantaneous error, not its
history. The mitigation is purely geometric (no dual-reference cross-fade --
that would be overengineering here): each new plan is anchored to the
drone's *exact* continuous position, then a short lead-in point 0.3 m ahead
along its *current heading*, then the A* path -- skipping the path's own
first cell (the drone's current cell), which sat behind the lead-in point
almost as often as ahead of it and was carving a spurious backward notch
into the very first version of this anchoring. That fix (see
`OnlinePlanner._anchor_waypoints`) brought the worst-case heading jump down
from ~180 deg to roughly 40-130 deg across test seeds -- better, but **not**
the tight <15 deg originally hoped for. `tests/test_env_attitude.py::test_replan_does_not_itself_cause_an_immediate_terminal`
confirms the swap itself never *immediately* reads as a collision or
departure, across several seeds -- but the tracking-error cost below shows
the mitigation is partial, not complete.

**Honest comparison, `experiments/compare_online_planning.py`, 6 seeds, plain
PD (no guidance bias) on the full 8-tree lap:**

| | offline (ground truth) | online (LIDAR-discovered) |
|---|---|---|
| success | 0 / 6 | 0 / 6 |
| mean tracking RMSE (e_lat, m) | 0.081 | 0.250 |
| mean steps survived | 722 | 1300 |

Two honest caveats, not buried: first, **plain PD alone does not reliably
complete the full multi-turn lap even in the offline, ground-truth
condition** -- every earlier PD success in this README was on
`build_local_gauntlet_env`'s short, near-straight two-tree segment, never
the full lap, and this is the first time plain PD (no LIDAR bias, no
guidance table) was actually run against it end to end. Success rate is
therefore a floor-effect metric here and can't discriminate the two
conditions -- it would take the hybrid guidance layer (or a stronger
controller generally) to get either condition off the floor, which is a
separate question from what this experiment isolates. Tracking RMSE is more
informative: **online tracking error is about 3x higher**, which is exactly
the discontinuity cost predicted above. Second, "mean steps survived" is
*not* evidence the online condition does better -- `ONLINE_TIME_MARGIN=2.5`
is deliberately larger than the offline `TIME_MARGIN=1.5` (a discovered path
is longer than `L*` by construction, so a fair timeout needs more slack),
so the online runs simply get more ticks before timing out; several online
rollouts (seeds 2, 3, 5) spend most of that extra time hovering near the
start rather than making progress. Replan counts (15-53 over a rollout)
confirm the mechanism fires throughout flight, not just once.

The honest takeaway: the online-planning *mechanism* -- sense, build belief,
replan, re-anchor -- works and is invariant-2-compliant end to end (see
`tests/test_invariants_online_planner.py`), but it is strictly harder than
the offline case, exactly as it should be: not knowing the map costs
something real, measured here in tracking error, and this repo would rather
report that plainly than dress up a partial mitigation as a solved problem.

## Tier-A+: the pirouette that physics vetoed (`agent/pirouette_attitude.py`)

The most instructive negative result in this kit. The setup: the drone's
body is a rollable ellipsoid (`env_attitude/occupancy.py`'s
`presented_half_width`: full rotor span `R_BODY=0.25` level, thin edge
`H_BODY=0.08` at knife-edge), so rolling toward `PHI_MAX_PIROUETTE=85 deg`
shrinks the width it presents to a gap -- and a gate narrower than the
level-flight footprint (`TREE_R+R_BODY=0.65`) but wider than the knife-edge
one (`TREE_R+H_BODY=0.48`) *should* be threadable by banking hard. A dense
64-ray LIDAR (`lidar3d.scan3d_dense`, 8 great circles x 8 rays -- measured
~417us/scan, ~2% of one core at the 50Hz tick, so ray count is not a real
constraint) senses the gap; a tiny 9-state x 7-action Q-table
(`pirouette_state` = bin(lidar_min) x bin(own |phi|), roll targets
0/+-30/+-60/+-85 deg) decides how hard to bank, on top of the same proven PD
that stabilizes everything else in this kit. A three-stage curriculum
(`env_attitude/scenarios_pirouette.py`: clearances 0.75/0.58/0.50, each
stage's required roll derived from real geometry by `gap_required_roll` --
0 deg, 47.1 deg, 75.3 deg) trains it, transfer vs. from-scratch, 10 seeds
each: `experiments/report_pirouette_curriculum.py`.

**Result: 0/10 transfer, 0/10 scratch.** But unlike the hybrid section's
1/10 (a convergence problem), this zero has a *diagnosed physical root
cause*, established by scripted-oracle probes (no learning involved --
hand-coded "roll to X deg when lidar_min < Y" policies):

| probe | outcome |
|---|---|
| roll to 85 deg at lidar_min<0.9, pirouette gate | COLLISION (roll only reached ~46 deg by the gate) |
| roll to 80 deg, trigger at 2.4m (earliest possible) | DEPARTED -- banked drone slides off the corridor |
| same, with the 1.0m deviation envelope REMOVED | drifts 30+ metres sideways, never reaches the gate |
| straight stage (0.75m, no roll needed) | SUCCESS -- the plumbing itself is fine |

The physics: **a bank in straight flight is inherently unbalanced.** In a
coordinated turn, the tilted thrust's lateral component supplies centripetal
force; flying *straight* while banked leaves that same component
(`g*sin(phi)` -- ~7 m/s^2 at 45 deg, ~9.7 at 80 deg) with nothing to oppose
it, so the drone accelerates sideways into the very trunk it banked toward,
while its vertical support simultaneously collapses to `cos(phi)` of hover.
A real knife-edge pass survives this only by thrust vectoring -- `T =
mg/cos(phi)`, i.e. 1.5x hover for this kit's "angled" gate (47 deg) and 3.9x
for the "pirouette" gate (75 deg) -- and thrust is *fixed at hover* in this
track by explicit original design (the time-boxing decision that kept the
action space at 27 torque triples). With this kit's roll authority
(`D_TAU=0.01` N*m on `I=3e-3` kg*m^2, ~3.3 rad/s^2), reaching 75 deg takes
~1.2s, by which point the unopposed slide has already carried the vehicle
out of the corridor. No policy -- learned or scripted -- can pass; the 0/10
is the environment telling the truth, not the RL failing to find an answer.

What survives this result: the sensor (64-ray dense LIDAR), the
orientation-aware collision model, the 9x7 state/action design and its
invariant guards (`tests/test_invariants_pirouette.py` structurally enforces
that dense sensing never widens the learned state -- `pirouette_state` may
extract only `lidar.min()` and own roll, no per-ray indexing), and the
curriculum/report infrastructure are all built, tested, and reusable. The
one missing prerequisite is physical, not algorithmic: **thrust as a fourth
action channel** (even 2 discrete levels, hover and ~2x hover, would make
the 47-deg "angled" gate physically feasible at `T=1.47x`). That is the
honest cost of the original fixed-thrust simplification, now measured
precisely -- and the natural next experiment for anyone extending this
track.
