"""The V(s) value surface -- the central demo asset (Phase 4).

Shows the story the concept promises: the drone bangs into trees, peaks grow
where it hurt, and the path straightens. The peaks rise from *reward*, never
from tree coordinates (INVARIANT 2) -- so they appear where the agent was
punished, which is only approximately where the trees are.
"""

from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from tier_d.scaffold.qtools import greedy_arrows, value_surface
from tier_d.env.constants import GRID_N, R_GOAL
from tier_d.env.occupancy import GOAL_CELL, cell_centers
from tier_d.viz.style import GOAL, GRIDC, MUTED, ORACLE, RL, START, draw_endpoints, draw_forest, setup_domain


def _surface_z(V: np.ndarray) -> np.ndarray:
    """The potential surface is -V: low at the goal, peaked at pain."""
    return -V


def plot_value_surface_3d(ax, V: np.ndarray, *, zlim: tuple[float, float] | None = None,
                          title: str = "") -> None:
    gx, gy = cell_centers()
    Z = _surface_z(V)
    ax.plot_surface(gx, gy, Z, cmap=cm.viridis, linewidth=0, antialiased=True,
                    rstride=1, cstride=1, alpha=0.96)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("-V(s)")
    ax.set_box_aspect((1, 1, 0.55))
    if zlim:
        ax.set_zlim(*zlim)
    ax.view_init(elev=46, azim=-121)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)
    if title:
        ax.set_title(title, pad=2)


def plot_policy_arrows(ax, Q: np.ndarray, blocked: np.ndarray, trees: np.ndarray,
                       path: np.ndarray | None = None, oracle_path: np.ndarray | None = None,
                       title: str = "") -> None:
    setup_domain(ax, title)
    gx, gy = cell_centers()
    u, v = greedy_arrows(Q)
    free = ~blocked
    ax.quiver(gx[free], gy[free], u[free], v[free], color=MUTED, alpha=0.55,
              scale=44, width=0.0032, zorder=3)
    draw_forest(ax, trees)
    if oracle_path is not None:
        ax.plot(oracle_path[:, 0], oracle_path[:, 1], color=ORACLE, lw=2.4, ls="--",
                zorder=4, label="A* optimum (L*)")
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color=RL, lw=2.6, zorder=5, label="learned greedy policy")
    draw_endpoints(ax)
    ax.legend(loc="upper left", fontsize=7.5)


def _unit(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def plot_learned_vs_ideal(fig, V_learned: np.ndarray, cost_to_go: np.ndarray) -> None:
    """Learned -V beside the oracle's navigation function. INVARIANT 6 made visible.

    Both are rescaled to [0,1]: they measure different things (discounted return
    vs shortest-path cost) and only their *shape* is comparable -- sink at the
    goal, ridges at the trees, no other basin.
    """
    gx, gy = cell_centers()

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(gx, gy, _unit(-V_learned), cmap=cm.viridis, linewidth=0,
                     antialiased=True, rstride=1, cstride=1, alpha=0.96)
    _style3d(ax1, "learned  −V(s)  (grown from reward)")

    # Unreachable/blocked cells become the highest ground, so trees read as peaks.
    ideal = cost_to_go.copy()
    finite_max = float(np.nanmax(ideal[np.isfinite(ideal)]))
    ideal[~np.isfinite(ideal)] = finite_max * 1.35

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(gx, gy, _unit(ideal), cmap=cm.magma, linewidth=0,
                     antialiased=True, rstride=1, cstride=1, alpha=0.96)
    _style3d(ax2, "oracle  U*(s)  navigation function\n(one minimum, at the goal)")


def _style3d(ax, title: str) -> None:
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("normalised")
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=46, azim=-121)
    ax.grid(False)
    ax.set_title(title, pad=2)


def animate_value_growth(snapshots: list[tuple[int, np.ndarray]], trees: np.ndarray,
                         blocked: np.ndarray, out_path, fps: int = 6) -> None:
    """GIF of -V(s) rising episode by episode. The recruiting visual."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    from tier_d.env.constants import R_COLLISION

    gx, gy = cell_centers()
    # Fixed z-limits across frames, or the surface "grows" only because the axis
    # rescales under it -- which would be a lie about what is happening.
    zmin, zmax = R_GOAL * -1.05, -R_COLLISION * 1.05

    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    def frame(i):
        ax.clear()
        ep, v, Qsnap = snapshots[i]
        V = v.reshape(GRID_N, GRID_N).copy()
        # Peak heights come from collisions actually suffered by episode `ep`,
        # so untouched trees are still flat -- the peaks visibly grow.
        from tier_d.scaffold.qtools import experienced_obstacle_value

        V[blocked] = experienced_obstacle_value(Qsnap, blocked)[blocked]
        V[GOAL_CELL] = R_GOAL
        ax.plot_surface(gx, gy, -V, cmap=cm.viridis, linewidth=0, antialiased=True,
                        rstride=1, cstride=1, vmin=zmin, vmax=zmax)
        ax.set_zlim(zmin, zmax)
        ax.set_box_aspect((1, 1, 0.55))
        ax.view_init(elev=46, azim=-121)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("-V(s)")
        ax.grid(False)
        ax.set_title(f"episode {ep}", pad=2)

    anim = FuncAnimation(fig, frame, frames=len(snapshots), interval=1000 // fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out_path}")
