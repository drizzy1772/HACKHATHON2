"""Shared drawing helpers. Headless by default so the report builds over ssh."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from tier_d.env.constants import DOMAIN, GOAL_XY, START_XY, TREE_R

# One palette, used everywhere, legible on white and in a dark-themed page.
INK = "#12161c"
MUTED = "#7a8594"
GRIDC = "#dfe3e8"
TREE = "#2f6d4f"
TREE_EDGE = "#1d4531"
START = "#1f6feb"
GOAL = "#d1242f"
ORACLE = "#8250df"
RL = "#0969da"
APF = "#bc4c00"
RESIDUAL = "#1a7f37"
TRAP = "#cf222e"
LIDARC = "#e3b341"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor": GRIDC,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "font.family": "sans-serif",
})


def draw_forest(ax, trees: np.ndarray, *, alpha: float = 1.0, r: float = TREE_R) -> None:
    for t in trees:
        ax.add_patch(Circle(tuple(t), r, facecolor=TREE, edgecolor=TREE_EDGE,
                            lw=0.6, alpha=alpha, zorder=2))


def draw_endpoints(ax, *, size: int = 90) -> None:
    ax.scatter(*START_XY, s=size, c=START, marker="o", zorder=6,
               edgecolors="white", linewidths=1.4, label="A (start)")
    ax.scatter(*GOAL_XY, s=size + 40, c=GOAL, marker="*", zorder=6,
               edgecolors="white", linewidths=1.0, label="B (goal)")


def setup_domain(ax, title: str = "") -> None:
    ax.set_xlim(0, DOMAIN)
    ax.set_ylim(0, DOMAIN)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, int(DOMAIN) + 1, 2))
    ax.set_yticks(range(0, int(DOMAIN) + 1, 2))
    ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    if title:
        ax.set_title(title, pad=8)


def draw_occupancy(ax, blocked: np.ndarray, *, alpha: float = 0.18) -> None:
    ax.imshow(blocked, origin="lower", extent=(0, DOMAIN, 0, DOMAIN),
              cmap="Greys", alpha=alpha, zorder=1, interpolation="nearest")


def path_length(p: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def save(fig, path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
