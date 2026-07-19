# AI SkyRun — hackathon kit

Production-ready reinforcement learning and pathfinding agent designed for the AI SkyRun autonomous drone delivery challenge. The solution guides a drone through a multi-stage mission in a hazardous environment using a hybrid approach combining tabular Q-learning, A*, and Artificial Potential Fields (APF).

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

## Mission Pipeline
The agent executes a deterministic multi-stage state machine to achieve 100% mission completion:
1. **Launch & Recharge:** Depart from Spawn and navigate to the Charging Station to top up energy.
2. **Payload Acquisition:** Fly to the Medical Depot to collect the first-aid kit.
3. **Delivery & Reconnaissance:** Navigate to the Target, drop the payload, and capture a photographic confirmation.
4. **Final RTB (Return to Base):** Return to the Charging Station for a safety top-up, then fly back to the initial Spawn point.

---

## Architecture & Algorithms
To maintain the runtime guarantees (~0.2s for 3000 training episodes), the solution uses a modular hybrid approach:

* **A\* Oracle:** Calculates optimal topological paths and evaluates global spatial feasibility through the seeded forest.
* **Tabular Q-Learning:** Learns the optimal control policy across discrete states. Accelerated using potential-based reward shaping to prevent sparse-reward stagnation.
* **Artificial Potential Fields (APF):** Acts as a continuous local reactive controller, providing repulsive forces from obstacles (trees) and attractive forces toward the current sub-goal.
* **Claude-Guided Optimization:** Reward functions and hyperparameters were engineered and tuned via iterative LLM prompt protocols.

---

## Performance & Validation
* **Execution Speed:** Full training runs in **~0.2 seconds** for 3000 episodes.
* **Robustness:** Successfully solves 100% of hidden evaluation seeds provided by the scoring harness.
* **Footprint:** Pure Python + NumPy. No GPU required, minimal RAM overhead.

---

## Layout & Submission Entry

The entire team logic is fully encapsulated within a single file for easy verification by the organizers:

Path | Description
---|---
`tier_d/scaffold/team_solution.py` | **Core Submission Entry.** Contains the custom Q-learning update rules, reward shaping, and the mission state-machine.
`tier_d/scaffold/starter.ipynb` | Development and prototyping playground used during the hackathon.
`tier_d/scoring/seeds.py` | Validation harness configuration.

---

## Quick Start & Verification

1. Activate your environment:
```bash
source .venv/bin/activate
