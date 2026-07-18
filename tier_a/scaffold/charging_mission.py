"""Демонстрація та верифікація двофазного польоту зі зарядною станцією.

Запуск:
    python tier_a/scaffold/charging_mission.py
    python tier_a/scaffold/charging_mission.py --seed 1 --charger 4 4
    python tier_a/scaffold/charging_mission.py --smoke   # швидка перевірка (150 ep)

Вихідні дані — таблиця з метриками по фазах для кожного seed.
"""

from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import torch

from tier_a.env_attitude.charging_env import ChargingStationEnv, Phase

# Намагаємось завантажити навчений агент
try:
    from tier_a.scaffold.team_solution_attitude import train_attitude, greedy_rollout_attitude
    _HAS_AGENT = True
except ImportError:
    _HAS_AGENT = False

PUBLIC_SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Rollout для ChargingStationEnv
# ---------------------------------------------------------------------------

def greedy_rollout_charging(
    env: ChargingStationEnv,
    qnet,
    max_steps: int = 6000,
) -> dict:
    """Запускає навчений агент (qnet) крізь повну двофазну місію.

    Повертає dict з:
        success         — bool: дрон долетів до GOAL в фазі 2
        collision       — bool: зіткнення в будь-якій фазі
        departed        — bool: вихід за межі в будь-якій фазі
        phase_reached   — остання Phase досягнута місією
        t_total         — float: загальний час симуляції (с)
        tracking_rmse_phase1 — RMSE фази START→CHARGER
        tracking_rmse_phase2 — RMSE фази CHARGER→GOAL
        tracking_rmse_total  — RMSE по всій місії
        charge_steps_done    — скільки кроків зарядки виконано
        steps           — загальна кількість кроків
    """
    obs = env.reset()
    info: dict = {
        "goal": False, "collision": False, "departed": False,
        "loss_of_control": False, "truncated": False,
        "phase": Phase.FLY_TO_CHARGER,
    }

    for _ in range(max_steps):
        phase = env.phase

        if phase == Phase.CHARGING:
            # Під час зарядки агент не діє — env сам виконує hover
            obs, _, done, info = env.step(0)  # action ігнорується в _step_charging
        else:
            with torch.no_grad():
                a = int(
                    qnet(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                    .argmax(dim=1)
                    .item()
                )
            obs, _, done, info = env.step(a)

        if done:
            break

    metrics = env.phase_metrics()
    return {
        "success":    bool(info.get("goal", False)),
        "collision":  bool(info.get("collision", False)),
        "departed":   bool(info.get("departed", False) or info.get("loss_of_control", False)),
        "phase_reached": env.phase.value,
        "t_total":    float(env._t_wall),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Навчання та оцінка
# ---------------------------------------------------------------------------

def run_charging_mission(
    seed: int,
    charger_xy: tuple[float, float] | None,
    episodes: int,
    verbose: bool = True,
) -> dict:
    """Тренує агент (на сегментному AttitudeEnv), потім запускає двофазну місію."""
    from tier_a.env_attitude.env import AttitudeEnv
    from tier_a.env_attitude.constants import N_TREES_ATTITUDE

    if verbose:
        print(f"\n[seed={seed}] Тренування агента ({episodes} ep)...", flush=True)

    # Тренуємо на звичайному AttitudeEnv (START→GOAL — базовий сегмент)
    train_env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=seed)
    qnet, _ = train_attitude(train_env, episodes=episodes, seed=seed, eval_every=0)

    # Будуємо двофазний env
    charge_env = ChargingStationEnv(
        charger_xy=charger_xy,
        seed=seed,
        n_trees=N_TREES_ATTITUDE,
    )

    if verbose:
        print(f"[seed={seed}] Зарядна станція: {charge_env.charger_xy}")
        print(f"[seed={seed}] Запуск двофазного rollout...", flush=True)

    result = greedy_rollout_charging(charge_env, qnet)
    result["seed"] = seed
    result["charger_xy"] = charge_env.charger_xy

    if verbose:
        _print_result(result)

    return result


def _print_result(r: dict) -> None:
    """Виводить метрики одного rollout."""
    ok = "✅ OK" if r["success"] else ("💥 Collision" if r["collision"] else
                                       ("🚫 Departed" if r["departed"] else "⏱ Timeout"))
    print(f"  Фінал: {ok}")
    print(f"  Фаза досягнута: {r['phase_reached']}")
    print(f"  Зарядка виконана: {r['charge_steps_done']} кроків")
    print(f"  RMSE Фаза 1 (→зарядка):  {r['tracking_rmse_phase1']:.4f}")
    print(f"  RMSE Фаза 2 (→ціль):     {r['tracking_rmse_phase2']:.4f}")
    print(f"  RMSE загальний:           {r['tracking_rmse_total']:.4f}")
    print(f"  Час польоту:              {r['t_total']:.2f} с")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Двофазна місія: дрон → зарядка → ціль")
    p.add_argument("--smoke", action="store_true",
                   help="Швидка перевірка (150 ep, 1 seed)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds для тестування (за замовчуванням PUBLIC_SEEDS)")
    p.add_argument("--charger", type=float, nargs=2, default=None,
                   metavar=("X", "Y"),
                   help="Координати зарядної станції. За замовчуванням — середина маршруту.")
    p.add_argument("--episodes", type=int, default=None,
                   help="Кількість епізодів навчання")
    args = p.parse_args()

    if not _HAS_AGENT:
        print("❌  Не знайдено train_attitude в team_solution_attitude.py. "
              "Заповніть функцію та спробуйте знову.")
        sys.exit(1)

    seeds = args.seeds or ([0] if args.smoke else PUBLIC_SEEDS)
    episodes = args.episodes or (150 if args.smoke else 600)
    charger_xy = tuple(args.charger) if args.charger else None

    print("\n" + "=" * 70)
    print("AI SkyRun — Двофазна місія з зарядною станцією")
    print(f"Seeds: {seeds}  |  Епізоди: {episodes}  |  "
          f"Зарядка: {charger_xy or 'auto (середина маршруту)'}")
    print("=" * 70)

    results = []
    for seed in seeds:
        try:
            r = run_charging_mission(seed, charger_xy, episodes)
            results.append(r)
        except ValueError as e:
            print(f"\n[seed={seed}] ⚠️  Помилка: {e}")

    if not results:
        print("Немає результатів.")
        return

    # Підсумкова таблиця
    print("\n" + "=" * 70)
    print(f"{'Seed':>5} {'Зарядка':>10} {'RMSE Ф1':>9} {'RMSE Ф2':>9} "
          f"{'RMSE заг':>9} {'Час':>6} {'Стан':>12}")
    print("-" * 70)
    for r in results:
        status = "OK" if r["success"] else (
            "Collision" if r["collision"] else (
            "Departed"  if r["departed"]  else r["phase_reached"]))
        ch = f"{r['charger_xy'][0]:.1f},{r['charger_xy'][1]:.1f}"
        print(f"{r['seed']:>5} {ch:>10} "
              f"{r['tracking_rmse_phase1']:>9.4f} "
              f"{r['tracking_rmse_phase2']:>9.4f} "
              f"{r['tracking_rmse_total']:>9.4f} "
              f"{r['t_total']:>6.1f} {status:>12}")
    print("=" * 70)

    n_success = sum(r["success"] for r in results)
    print(f"\nУспішних місій: {n_success}/{len(results)}")
    if args.smoke:
        print("(Smoke-режим — невеликий бюджет, низька збіжність очікувана)")
    print()


if __name__ == "__main__":
    main()
