# AI SkyRun — hackathon kit

Grid-world RL hackathon kit: seeded forest MDP, A* oracle, reference tabular
Q-learning with potential-based shaping, an APF baseline, a `V(s)` visualiser, a
scoring harness on hidden seeds, and a student scaffold.

Pure Python + NumPy + Matplotlib. The core is **tabular** — no neural nets, no GPU.
A 3000-episode training run takes ~0.2 s; the full test suite ~1 min.

## Tiers

Each tier is self-contained (code, tests, reports, `CONCEPT.md`) and split into a
student part and an organizer part:

| Folder | Track | Stack |
|---|---|---|
| [`tier_d/`](tier_d/) | the base **2D** competition kit described below: grid MDP, A* oracle, tabular Q-learning, APF foil, scoring, scaffold | main venv |
| [`tier_a/`](tier_a/) | research: 6DOF attitude control along a reference path — ends in the measured fixed-thrust physics veto | main venv (+optional torch) |
| [`tier_b/`](tier_b/) | research: CTBR pirouette in gym-pybullet-drones, PPO vs SAC | conda env `drones` |
| [`tier_c/`](tier_c/) | realistic FPV drone in Blender: A* + APF + kinematic autopilot, manual & autonomous modes | Blender (bundled python) |

## AI-assistant prompts

AI-assistant system prompts for the event live in [`prompts/`](prompts/):
one universal protocol + one card per tier.

---

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Verify: `python -c "import numpy, matplotlib; print('ok')"`

## Run

```bash
python tier_d/scaffold/validate_team.py       # self-check on PUBLIC seeds (fill team_solution.py first)
```

Open `tier_d/scaffold/starter.ipynb` to start the Tier 1 task (two `# TODO` holes).

Open `tier_d/viz/field_scanner.html` in a browser for the Tier 0 APF demo. Its
forest is bit-identical to Python's.

## Layout (tier_d)

| Path | What |
|---|---|
| `tier_d/env/` | `constants.py` (all geometry), `forest.py` (mulberry32), `occupancy.py` (the one collision test), `gridworld.py` (the MDP), `lidar.py` |
| `tier_d/oracle/` | `astar.py` (L*, feasibility), `value_iteration.py` (navigation function, V*) |
| `tier_d/scoring/` | `seeds.py` — PUBLIC seeds |
| `tier_d/scaffold/` | `starter.ipynb` (two `# TODO` holes), `team_solution.py` (student export stub), `qtools.py` (plot helpers), `README.md`, `validate_team.py` |
| `tier_d/viz/` | student-visible: `style.py`, `forest3d.py`, `value_surface.py`, Field Scanner |
| `tier_d/webots/` | `bridge.py` — policy→waypoint, no simulator import |

## Read first

**For teams:**
- `tier_d/scaffold/README.md` — RL concepts, the invariants teams must respect
- `tier_d/scaffold/GAMEPLAY_GUIDE.md` — diagnostic checklist, Tier 1–3 approaches, FAQ
- `tier_d/scaffold/validate_team.py` — self-check on PUBLIC_SEEDS before final
- `prompts/README.md` — AI-assistant prompts: universal protocol + per-tier cards

## Not done

**Phase 7 (Webots) is not verified end-to-end** — Webots is not installed on this
machine. `tier_d/webots/bridge.py` contains the policy→waypoint logic and is unit-tested
without importing the simulator; the 3D flight is rendered in Matplotlib instead.
Writing the `.wbt` world and the controller is the remaining work.

**Phase 8 (packaging)** — versions are not frozen and no local pip mirror exists.
Run `pip freeze > requirements.lock` only after a successful dry-run.
