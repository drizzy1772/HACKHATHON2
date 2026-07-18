#!/usr/bin/env python3
"""Порівняння PPO vs SAC із Monitor-CSV: криві success/collision/jam за кроками,
трейс рівня курикулуму, довжина епізодів → PNG + summary.csv + md-таблиця.

    conda run -n drones python tier_b/compare.py \
        --ppo tier_b/runs/ppo/seed0 --sac tier_b/runs/sac/seed0
"""

from __future__ import annotations

import argparse
import csv
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

WINDOW = 200


def load_episodes(run_dir: str) -> list[dict]:
    """Зливає monitor_*.csv усіх воркерів у глобальний порядок за часом 't'."""
    eps = []
    for path in glob.glob(str(pathlib.Path(run_dir) / "monitor" / "monitor_*.csv*")):
        with open(path) as f:
            header = f.readline()  # #{"t_start":...}
            assert header.startswith("#")
            for row in csv.DictReader(f):
                eps.append({
                    "r": float(row["r"]), "l": int(row["l"]), "t": float(row["t"]),
                    "success": row.get("success", "False") == "True",
                    "collision": row.get("collision", "False") == "True",
                    "jam": row.get("jam", "False") == "True",
                    "level": int(row.get("level", -1)),
                })
    eps.sort(key=lambda e: e["t"])
    steps = 0
    for e in eps:
        steps += e["l"]
        e["cum_steps"] = steps
    return eps


def rolling(eps: list[dict], key: str, window: int = WINDOW):
    xs = np.array([e["cum_steps"] for e in eps])
    vals = np.array([float(e[key]) for e in eps])
    if len(vals) < 2:
        return xs, vals
    kernel = np.ones(min(window, len(vals)))
    num = np.convolve(vals, kernel, mode="valid")
    den = len(kernel)
    return xs[len(kernel) - 1:], num / den


def summarize(eps: list[dict], algo: str) -> dict:
    last = eps[-WINDOW:] if len(eps) >= WINDOW else eps
    per_level = {}
    for lv in sorted({e["level"] for e in eps}):
        lv_eps = [e for e in eps if e["level"] == lv]
        per_level[lv] = (sum(e["success"] for e in lv_eps), len(lv_eps))
    return {
        "algo": algo,
        "episodes": len(eps),
        "total_steps": eps[-1]["cum_steps"] if eps else 0,
        "final_success_rate": float(np.mean([e["success"] for e in last])) if last else 0.0,
        "final_collision_rate": float(np.mean([e["collision"] for e in last])) if last else 0.0,
        "final_jam_rate": float(np.mean([e["jam"] for e in last])) if last else 0.0,
        "max_level_reached": max((e["level"] for e in eps), default=-1),
        "per_level": per_level,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ppo", type=str, default="tier_b/runs/ppo/seed0")
    ap.add_argument("--sac", type=str, default="tier_b/runs/sac/seed0")
    ap.add_argument("--out-dir", type=str, default="tier_b/admin/reports")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {"PPO": load_episodes(args.ppo), "SAC": load_episodes(args.sac)}
    runs = {k: v for k, v in runs.items() if v}
    colors = {"PPO": "#1f77b4", "SAC": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for name, eps in runs.items():
        c = colors[name]
        x, y = rolling(eps, "success")
        axes[0, 0].plot(x, y, color=c, label=name)
        x, y = rolling(eps, "collision")
        axes[0, 1].plot(x, y, color=c, label=name)
        x, y = rolling(eps, "jam")
        axes[0, 1].plot(x, y, color=c, ls="--", alpha=0.6, label=f"{name} jam")
        axes[1, 0].plot([e["cum_steps"] for e in eps], [e["level"] for e in eps],
                        color=c, label=name)
        x, y = rolling(eps, "l")
        axes[1, 1].plot(x, y, color=c, label=name)

    axes[0, 0].set_title(f"Success rate (вікно {WINDOW} еп.)")
    axes[0, 1].set_title("Collision (суц.) і jam (пункт.) rate")
    axes[1, 0].set_title("Рівень курикулуму")
    axes[1, 1].set_title("Середня довжина епізоду, кроки")
    for ax in axes.flat:
        ax.set_xlabel("кроки середовища")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("CTBR pirouette: PPO vs SAC (спільні сіди лісів)")
    fig.tight_layout()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    png = out_dir / "tier_b_ppo_vs_sac.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    print(f"записано {png}")

    summaries = [summarize(eps, name) for name, eps in runs.items()]
    csv_path = out_dir / "tier_b_ppo_vs_sac_summary.csv"
    keys = ["algo", "episodes", "total_steps", "final_success_rate",
            "final_collision_rate", "final_jam_rate", "max_level_reached"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    print(f"записано {csv_path}")

    print("\n| algo | episodes | steps | success | collision | jam | max level |")
    print("|---|---|---|---|---|---|---|")
    for s in summaries:
        print(f"| {s['algo']} | {s['episodes']} | {s['total_steps']} "
              f"| {s['final_success_rate']:.2f} | {s['final_collision_rate']:.2f} "
              f"| {s['final_jam_rate']:.2f} | L{s['max_level_reached']} |")
    for s in summaries:
        print(f"\n{s['algo']} per-level (success/episodes): " +
              ", ".join(f"L{lv}: {v[0]}/{v[1]}" for lv, v in s["per_level"].items()))


if __name__ == "__main__":
    main()
