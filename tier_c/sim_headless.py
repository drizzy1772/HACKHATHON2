# -*- coding: utf-8 -*-
"""
Headless-оркестрація автономного польоту: мапа → find_path → (2D-лідар +
compute_desired_direction + step_autopilot) цикл → ТАБЛИЦЯ кадрів (JSON).
Чиста функція без bpy — увесь розрахунок відбувається тут, ОДИН РАЗ, наперед;
Blender (blender_manual.py) лише ЧИТАЄ результат і виставляє позу дрона
щокадру — жодної фізики/AI усередині Blender.

ХАКАТОН: саму навігаційну логіку (find_path/compute_desired_direction/
StuckDetector/step_autopilot) реалізують учасники в solution.py — цей файл
лише викликає їх у правильному порядку й НЕ потребує змін.

Можна запускати і як звичайний Python-скрипт (headless, без Blender: python3 —
потрібен numpy), і через Blender -b (bundled Python уже має numpy), і просто
ІМПОРТОМ функції simulate() з інтерактивного Blender (адмін-кнопки «Завантажити
тестову мапу» / «Випадкова мапа» у blender_manual.py викликають її напряму —
без bpy усередині цього файлу такий виклик безпечний із самого модального оператора).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game_env import (DEFAULT_CONFIG as CFG, STATUS_RUNNING, STATUS_COLLISION,
                      STATUS_DISQUALIFIED, STATUS_FINISHED, STATUS_TIMEOUT)
from game_env.generator import (MapGenerator, wreck_yaw, point_in_oriented_box,
                                OBJECT_FOOTPRINTS, TRUCK_FOOTPRINT)
from game_env.lidar2d import binned_lidar_2d, lidar_obstacles_xyr, bin_angles
from apf_controller import PathTracker              # дано: carrot-chasing утиліта
from kinematic_autopilot import AutopilotState        # дано: контракт стану дрона
from solution import (find_path, StuckDetector, APFParams, compute_desired_direction,
                      AutopilotParams, step_autopilot)

SIM_HZ = 30.0
MAX_SIM_SECONDS = 90.0

# ── Фізика збурень (вітер/пориви дощу) ───────────────────────────────────────
# Щотіка з імовірністю WIND_PROB дрон отримує боковий поштовх WIND_GUST_M метрів
# у випадковий бік — ПЕРЕД перевіркою колізій, тож вітер реально впливає на
# траєкторію й статус. Сіється від seed прогону → відтворювано (той самий сід =
# той самий вітер). Рівень нижче — «помірний шторм», який розв'язок витримує
# (виміряно: 10/10 аж до 50%×0.12м); підніми, щоб зробити політ важчим.
WIND_ENABLED = True
WIND_PROB = 0.40        # частка тіків із поривом
WIND_GUST_M = 0.10      # сила пориву, м/тік
OUT_DIR = Path(__file__).resolve().parent / "out"
HISTORY_PATH = OUT_DIR / "autonomous_runs.json"


def collision_and_bounds_status(pos, md, terrain, cfg):
    """Дослівно логіка _update_status коміту 2e2bc5d (та сама, що й у
    blender_manual.py, тут — БЕЗ bpy-залежностей, чиста геометрія): межі (стеля АБО
    горизонтальні межі арени) → DISQUALIFIED; дотик рельєфу/дерева/колізійної
    перешкоди/кузова фури → COLLISION."""
    x, y, z = pos
    if z > md.ceiling or abs(x) > cfg.bounds or abs(y) > cfg.bounds:
        return STATUS_DISQUALIFIED
    if z - terrain.height_at(x, y) < cfg.drone_radius:
        return STATUS_COLLISION
    for i, (tx, ty, z_base, r, h) in enumerate(md.trees):
        if i == md.wreck_index and md.wreck_kind in OBJECT_FOOTPRINTS:
            # Уламок техніки — окремий орієнтований хітбокс (не кругла
            # апроксимація WRECK_DIMS-радіуса) — та сама логіка, що й у
            # blender_manual.py.
            if point_in_oriented_box(x, y, tx, ty, wreck_yaw(tx, ty),
                                     OBJECT_FOOTPRINTS[md.wreck_kind], margin=cfg.drone_radius):
                if z_base - cfg.drone_radius <= z <= z_base + h + cfg.drone_radius:
                    return STATUS_COLLISION
            continue
        if math.hypot(x - tx, y - ty) < r + cfg.drone_radius:
            if z_base - cfg.drone_radius <= z <= z_base + h + cfg.drone_radius:
                return STATUS_COLLISION
    for _kind, ox, oy, oz, r, collidable in md.obstacles:
        if collidable and math.dist((x, y, z), (ox, oy, oz)) < r + cfg.drone_radius:
            return STATUS_COLLISION
    dr = cfg.drone_radius
    for cx, cy, _cz in md.checkpoints:
        gz = terrain.height_at(cx, cy)
        if (point_in_oriented_box(x, y, cx, cy, 0.0, TRUCK_FOOTPRINT, margin=dr)
                and gz - dr <= z <= gz + cfg.truck_height):
            return STATUS_COLLISION
    return STATUS_RUNNING


def simulate(seed: int, n_trees: int = None, sim_seconds: float = MAX_SIM_SECONDS,
            sim_hz: float = SIM_HZ) -> dict:
    """Повний автономний прогін. Повертає словник {meta, map, frames} — таблицю,
    придатну і для json.dump, і для прямого відтворення в Blender без серіалізації."""
    cfg = CFG if n_trees is None else dataclasses.replace(CFG, n_trees=int(n_trees))

    md = MapGenerator(cfg).build(seed=seed)
    terrain = md.terrain(cfg)
    path = find_path(md, cfg, cell_size=1.0)
    if path is None:                      # не мало б статись на розумній мапі — запобіжник
        path = [tuple(md.start[:2]), (md.checkpoints[0][0], md.checkpoints[0][1])]

    ox, oy, orr = lidar_obstacles_xyr(md)
    ba = bin_angles(cfg.lidar_n_az)

    ap_p = AutopilotParams()
    state = AutopilotState(x=md.start[0], y=md.start[1], z=md.start[2])
    tracker = PathTracker(path)
    stuck = StuckDetector()
    apf_p = APFParams()
    boost_state = {}

    import random
    wind_rng = random.Random(seed)          # відтворюваний вітер на кожен сід

    goal = md.checkpoints[0]
    goal_radius = cfg.cp_cyl_radius
    dt = 1.0 / sim_hz

    frames = []
    status = STATUS_RUNNING
    t = 0.0
    n_steps = int(sim_seconds * sim_hz)
    for i in range(n_steps):
        t += dt
        lidar = binned_lidar_2d(ox, oy, orr, state.x, state.y, cfg.lidar_n_az, cfg.lidar_range)
        direction, boosted, _carrot = compute_desired_direction(
            (state.x, state.y), lidar, ba, tracker, stuck, t, apf_p, boost_state)
        state = step_autopilot(state, direction, terrain, dt, ap_p, min_lidar=float(lidar.min()))

        # ВІТЕР: боковий порив ПЕРЕД перевіркою колізій — реально впливає на політ
        if WIND_ENABLED and wind_rng.random() < WIND_PROB:
            ang = wind_rng.uniform(0.0, 2.0 * math.pi)
            state.x += WIND_GUST_M * math.cos(ang)
            state.y += WIND_GUST_M * math.sin(ang)

        status = collision_and_bounds_status((state.x, state.y, state.z), md, terrain, cfg)
        if status == STATUS_RUNNING and math.hypot(state.x - goal[0], state.y - goal[1]) < goal_radius:
            status = STATUS_FINISHED

        frames.append({
            "t": round(t, 3), "x": round(state.x, 3), "y": round(state.y, 3), "z": round(state.z, 3),
            "yaw": round(state.yaw, 4), "pitch": round(state.pitch, 4), "roll": round(state.roll, 4),
            "speed": round(math.hypot(state.vx, state.vy), 3),
            "lidar": [round(float(v), 2) for v in lidar],
            "status": status, "boosted": bool(boosted),
        })
        if status != STATUS_RUNNING:
            break
    else:
        status = STATUS_TIMEOUT
        if frames:
            frames[-1]["status"] = STATUS_TIMEOUT

    return {
        "meta": {"seed": seed, "sim_hz": sim_hz, "n_frames": len(frames),
                "final_status": status, "n_trees": cfg.n_trees},
        "map": md.to_dict(),
        "frames": frames,
    }


def save_trajectory(result: dict, path=None) -> Path:
    out_path = Path(path) if path else (OUT_DIR / "trajectory.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return out_path


def load_trajectory(path=None) -> dict:
    in_path = Path(path) if path else (OUT_DIR / "trajectory.json")
    return json.loads(Path(in_path).read_text(encoding="utf-8"))


def compute_run_metrics(result: dict) -> dict:
    """Легкі підсумкові метрики якості ОДНОГО прогону (без heightmap/кадрів —
    придатні для персистентної історії, на відміну від save_trajectory, що
    зберігає весь важкий result). Рахує зі стану, який УЖЕ є в result:
      • path_efficiency — пряма відстань старт→ціль / реально пройдений шлях
        (1.0 = ідеально пряма траєкторія, менше — більше «блукання»);
      • min/avg_clearance — найближче й типове наближення до перешкод за
        весь прогін (мінімум/середнє по кожному кадру з binned_lidar_2d);
      • stuck_pct — частка кадрів, де APF-детектор застрягання увімкнув
        «буст» (frame["boosted"]) — показник того, як часто алгоритму
        доводилось «виборсуватися» замість плавної навігації."""
    meta = result["meta"]
    frames = result["frames"]
    map_d = result["map"]
    start = map_d["start"]
    goal = map_d["checkpoints"][0]
    n = len(frames)

    straight_line = math.hypot(goal[0] - start[0], goal[1] - start[1])
    path_length = 0.0
    speeds = []
    min_clearances = []
    stuck_frames = 0
    px, py = start[0], start[1]
    for fr in frames:
        path_length += math.hypot(fr["x"] - px, fr["y"] - py)
        px, py = fr["x"], fr["y"]
        speeds.append(fr["speed"])
        if fr["lidar"]:
            min_clearances.append(min(fr["lidar"]))
        if fr.get("boosted"):
            stuck_frames += 1

    return {
        "seed": meta["seed"],
        "final_status": meta["final_status"],
        "n_frames": n,
        "n_trees": meta["n_trees"],
        "duration_s": round(frames[-1]["t"], 2) if frames else 0.0,
        "path_length": round(path_length, 2),
        "straight_line_distance": round(straight_line, 2),
        "path_efficiency": round(straight_line / path_length, 4) if path_length > 1e-6 else None,
        "avg_speed": round(sum(speeds) / n, 2) if n else 0.0,
        "max_speed": round(max(speeds), 2) if speeds else 0.0,
        "min_clearance": round(min(min_clearances), 2) if min_clearances else None,
        "avg_min_clearance": round(sum(min_clearances) / len(min_clearances), 2) if min_clearances else None,
        "stuck_frames": stuck_frames,
        "stuck_pct": round(100.0 * stuck_frames / n, 1) if n else 0.0,
    }


def append_run_history(metrics: dict, path=None) -> Path:
    """Дописати один запис метрик у персистентну історію (з таймстампом) —
    з КОЖНОГО автономного прогону (тестова/випадкова мапа, «Запустити
    автономний»), незалежно від сесії Blender, доки не видалено out/."""
    out_path = Path(path) if path else HISTORY_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = load_run_history(out_path)
    record = dict(metrics)
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    history.append(record)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return out_path


def load_run_history(path=None) -> list:
    in_path = Path(path) if path else HISTORY_PATH
    if not in_path.exists():
        return []
    return json.loads(in_path.read_text(encoding="utf-8"))


def clear_run_history(path=None) -> None:
    """Стерти персистентну історію прогонів (кнопка «Очистити історію»)."""
    out_path = Path(path) if path else HISTORY_PATH
    if out_path.exists():
        out_path.unlink()


def _parse_args():
    import sys
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    import argparse
    ap = argparse.ArgumentParser(description="Headless-симуляція автономного польоту")
    ap.add_argument("--seed", type=int, default=None, help="сід мапи (типово випадковий)")
    ap.add_argument("--out", default=None, help="шлях JSON-виводу (типово out/trajectory.json)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    import random as _random

    args = _parse_args()
    seed = args.seed if args.seed is not None else _random.randint(0, 1_000_000)
    result = simulate(seed)
    out_path = save_trajectory(result, args.out)
    print("сід=%d статус=%s кадрів=%d -> %s" % (
        seed, result["meta"]["final_status"], result["meta"]["n_frames"], out_path))
