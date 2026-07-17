#!/usr/bin/env python3
"""Скриптований зонд «флік» (діагностичний оракул у стилі Tier-A, НЕ агент —
читає ground truth свідомо): чи здатен дрон ФІЗИЧНО пройти вузьку щілину на
крутому крені, і за якої стратегії тяги.

Гіпотеза з фізики: тримати hover-тягу в крені не можна — бічна компонента
c·sinφ зносить на стовбур (за 60° і c=2g це 1.7g бічного прискорення!), а
вертикальна опора c·cosφ за 75° недосяжна навіть із 2g. Правильний маневр —
«флік»: різкий крен + СКИДАННЯ тяги (балістичний прохід, майже без бічного
зносу, ціна — просадка ½gt²) + відновлення на 2g після воріт.

    conda run -n drones python tier_b/probe_flick.py            # ґратка
    conda run -n drones python tier_b/probe_flick.py --debug    # один епізод
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from tier_b.envs.pirouette_aviary import GAPS, PirouetteAviary  # noqa: E402

G = 9.8
RATE_LIM = np.array([6.0, 6.0, 3.0])


def _act(c: float, wx: float, wy: float, wz: float) -> np.ndarray:
    """(c [м/с²], ω_sp [рад/с]) → дія [-1,1]⁴ (інверсія decode_action)."""
    a0 = np.clip(c / G - 1.0, -1.0, 1.0)
    w = np.clip(np.array([wx, wy, wz]) / RATE_LIM, -1.0, 1.0)
    return np.array([a0, *w], dtype=np.float32)


def run_episode(level: int, seed: int, phi_target_deg: float,
                c_flick_g: float = 0.05, v_cruise: float = 1.2,
                v_dash: float = 2.2, debug: bool = False) -> dict:
    """Флік-стратегія (фізика, не навчання):
      • центруємось повільно, за 1.2 м до воріт — розгін до v_dash
        (коротший час у зоні дерев ⇒ менші і знос, і просадка);
      • у фліку c = c_flick_g·g — компроміс: просадка (g − c·cosφ) проти
        зносу (c·sinφ); обидва ∝ t², тож головне — швидкість проходу;
      • приціл зміщено на +4 см проти відомого напрямку зносу (φ>0 ⇒ −y);
      • перед ворітьми +12 см висоти — бюджет на просадку."""
    import pybullet as p

    env = PirouetteAviary(level=level, master_seed=seed)
    env.reset()
    phi_t = np.radians(phi_target_deg)

    info = {}
    phase = "align"
    max_roll_in_gate = 0.0
    for i in range(600):
        pos, vel = env.pos[0], env.vel[0]
        roll, pitch, yaw = p.getEulerFromQuaternion(env.quat[0])

        gi = env.tracker.gates_passed
        if gi < len(env.forest.gate_centers):
            gx, gy = env.forest.gate_centers[gi]
        else:
            gx, gy = env.tracker.waypoints[-1][:2]

        near_gate = gi < 3 and 0 < gx - pos[0] < 3.5
        # ВИМІРЯНО (run #12): бічний імпульс крену залежить лише від кінцевих
        # кутів: Δv_y = c(1−cosφ)/ω̄ ≈ +0.98 м/с при c=g, φ=60° — знесло на
        # +0.77 м. Протидія — «краб»: підхід зі зміщенням і зустрічною v_y,
        # щоб імпульс маху доставив дрона В центр із v_y≈0 у момент проходу.
        crab_dy = 0.30 * (1.0 - np.cos(phi_t))       # калібровано з run #12
        crab_vy = -1.9 * (1.0 - np.cos(phi_t))
        aim_y = gy - (crab_dy if near_gate else 0.0)
        z_ref = 1.5 if near_gate else 1.0           # бюджет на просадку

        d_gate = gx - pos[0]
        # S-флік: preroll до −φ, мах −φ→+φ (бічні імпульси скорочуються:
        # ∫sinφ dt = 0 на симетричному маху), у зоні дерев — coast без тяги.
        # ВИМІРЯНО: крен вимагає тяги (момент ∝ f моторів) — крутити треба
        # на c≈1g, а «нульова тяга» можлива лише на короткому coast.
        t_preroll = phi_t / 6.0 + 0.10
        t_swing = 2.0 * phi_t / 6.0 + 0.10
        trigger_d = max(vel[0], 0.5) * (t_preroll + t_swing) + 0.32

        centered = abs(aim_y - pos[1]) < 0.025 and abs(vel[1]) < 0.12 \
            and abs(z_ref - pos[2]) < 0.25
        if phase == "align" and near_gate and 1.8 < d_gate and centered:
            phase = "dash"
        if phase == "dash" and gi < 3 and 0 < d_gate < trigger_d \
                and abs(aim_y - pos[1]) < 0.05 and vel[0] > 1.5:
            phase = "preroll"
        if phase == "preroll" and roll < -(phi_t - np.radians(6)):
            phase = "swing"
        if phase == "swing" and roll > phi_t - np.radians(6):
            phase = "coast"
        if phase == "coast" and pos[0] > gx + 0.12:
            phase = "recover"
        if phase == "recover" and abs(roll) < 0.15 and abs(pos[2] - 1.0) < 0.25:
            phase = "align"

        if phase == "preroll":
            wx = np.clip(8.0 * (-phi_t - roll), -6, 6)
            action = _act(1.1 * G, wx, 0.0, 0.0)
        elif phase == "swing":
            wx = 6.0
            action = _act(1.0 * G, wx, 0.0, 0.0)
        elif phase == "coast":
            # тримаємо крен, тяга скинута — прохід повз стовбури без зносу
            wx = np.clip(8.0 * (phi_t - roll), -6, 6)
            action = _act(c_flick_g * G, wx, 0.0, 0.0)
        else:
            c = G * (1.0 + 2.5 * (z_ref - pos[2]) - 1.0 * vel[2])
            if phase == "recover":
                c = min(2.0 * G, c + 0.6 * G)       # добираємо просадку

            y_err = aim_y - pos[1]
            if phase == "dash":
                v_ref = v_dash                       # менше часу в зоні дерев
            elif near_gate and d_gate < 2.2:
                v_ref = 0.0                          # стоп-кран: не готові — стоїмо
            elif near_gate:
                v_ref = 0.5 if abs(y_err) > 0.025 else v_cruise
            else:
                v_ref = v_cruise
            theta_ref = np.clip(0.4 * (v_ref - vel[0]), -0.4, 0.4)
            wy = np.clip(6.0 * (theta_ref - pitch), -6, 6)

            k_y = 2.2 if near_gate else 0.8
            if phase == "dash" and d_gate < trigger_d + 0.35:
                # крабимо: набираємо зустрічну v_y перед самим прероллом
                phi_ref = np.clip(-1.2 * (crab_vy - vel[1]), -0.5, 0.5)
            else:
                phi_ref = np.clip(-k_y * y_err + 0.7 * vel[1], -0.45, 0.45)
            wx = np.clip(6.0 * (phi_ref - roll), -6, 6)

            wz = np.clip(-2.0 * yaw, -3, 3)
            action = _act(c, wx, wy, wz)

        obs, r, term, trunc, info = env.step(action.reshape(1, 4))
        if any(abs(env.pos[0][0] - g[0]) < 0.25 for g in env.forest.gate_centers):
            max_roll_in_gate = max(max_roll_in_gate, abs(np.degrees(roll)))
        if debug and (i % 12 == 0 or phase != "cruise"):
            d_tree = float(np.min(np.hypot(env.forest.trees[:, 0] - env.pos[0][0],
                                           env.forest.trees[:, 1] - env.pos[0][1])))
            print(f"t={i/48:5.2f} {phase:8s} x={env.pos[0][0]:5.2f} "
                  f"y={env.pos[0][1]:5.2f} z={env.pos[0][2]:4.2f} "
                  f"roll={np.degrees(roll):6.1f} d_tree={d_tree:4.2f} gates={gi}")
        if term or trunc:
            break

    out = {k: info.get(k) for k in ("success", "collision", "jam",
                                    "out_of_bounds", "gates_passed")}
    out["max_roll_in_gate"] = round(max_roll_in_gate, 1)
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--phi", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--grid", action="store_true", default=None)
    args = ap.parse_args()

    if args.debug:
        out = run_episode(args.level, 900001, args.phi, debug=True)
        print(out)
        return

    print(f"{'рівень':>7} {'щілина':>7} {'крен цілі':>10} | "
          f"{'успіхи':>7} {'колізії':>8} {'сер.макс.крен у щілині':>24}")
    for level in (6, 7, 8):
        for phi in (45.0, 60.0, 75.0, 85.0):
            outs = [run_episode(level, 900001 + s, phi)
                    for s in range(args.seeds)]
            n_s = sum(bool(o["success"]) for o in outs)
            n_c = sum(bool(o["collision"]) for o in outs)
            rolls = [o["max_roll_in_gate"] for o in outs if o["success"]]
            mean_roll = f"{np.mean(rolls):.1f}°" if rolls else "-"
            gap = GAPS[level]
            print(f"L{level:>6} {gap:>6.2f}м {phi:>9.0f}° | "
                  f"{n_s:>4}/{args.seeds} {n_c:>8} {mean_roll:>24}")


if __name__ == "__main__":
    main()
