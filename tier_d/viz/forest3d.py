"""3D forest render + drone flight. Stands in for the Webots demo (Phase 7).

Webots is not installed on this machine, so the learned greedy policy is flown
through a matplotlib 3D scene instead: real cylinders, real trajectory, real
LIDAR returns. ``tier_d/webots/bridge.py`` keeps the policy->waypoint conversion that a
Webots controller would consume, and is unit-tested without importing Webots.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from tier_d.env.constants import DOMAIN, GOAL_XY, START_XY, TREE_R
from tier_d.env.lidar import endpoints, scan
from tier_d.viz.style import GOAL, LIDARC, RL, START, TREE, TREE_EDGE, TRAP

TREE_H = 4.0
FLY_Z = 1.6


def _cylinder(ax, cx, cy, r=TREE_R, h=TREE_H, n=18, alpha=0.85):
    th = np.linspace(0, 2 * np.pi, n)
    z = np.array([0.0, h])
    TH, Z = np.meshgrid(th, z)
    X = cx + r * np.cos(TH)
    Y = cy + r * np.sin(TH)
    ax.plot_surface(X, Y, Z, color=TREE, alpha=alpha, linewidth=0, shade=True)
    # canopy cap
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), np.full(n, h),
            color=TREE_EDGE, lw=0.8, alpha=0.9)


def setup_scene(ax, trees: np.ndarray, *, elev: float = 32, azim: float = -128) -> None:
    # Semi-transparent trunks: the drone flies at z=1.6, inside the canopy, so
    # opaque cylinders hide both the trajectory and the goal behind them.
    for t in trees:
        _cylinder(ax, t[0], t[1], alpha=0.45)

    # Beacons rather than ground dots, for the same reason.
    ax.plot([START_XY[0]] * 2, [START_XY[1]] * 2, [0, TREE_H], color=START,
            lw=1.0, ls=":", alpha=0.7)
    ax.plot([GOAL_XY[0]] * 2, [GOAL_XY[1]] * 2, [0, TREE_H], color=GOAL,
            lw=1.2, ls=":", alpha=0.8)
    ax.scatter([START_XY[0]], [START_XY[1]], [FLY_Z], s=55, c=START,
               edgecolors="white", linewidths=1.0, depthshade=False, zorder=8)
    ax.scatter([GOAL_XY[0]], [GOAL_XY[1]], [FLY_Z], s=190, c=GOAL, marker="*",
               edgecolors="white", linewidths=0.8, depthshade=False, zorder=8)
    ax.set_xlim(0, DOMAIN); ax.set_ylim(0, DOMAIN); ax.set_zlim(0, TREE_H + 1)
    ax.set_box_aspect((1, 1, 0.42))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.02)


def draw_attitude_gizmo(ax, xyz: np.ndarray, phi: float, theta: float, psi: float,
                        length: float = 0.5) -> list:
    """Draw a roll/pitch/yaw body-frame gizmo (3 orthogonal axes) at ``xyz``.

    Tier-A only: the base 2D kit has no orientation. Body -> world rotation
    matches ``env_attitude.dynamics.rotation_matrix`` (ZYX Euler) so the gizmo
    reflects the same attitude the physics actually integrated. Returns the
    list of plotted Line3D artists (useful for ``FuncAnimation`` to remove
    and redraw each frame).
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    rx = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
    ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    rz = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
    R = rz @ ry @ rx

    axes = np.eye(3) * length
    colors = ("#cf222e", "#1a7f37", "#0969da")  # body x (nose), y (right), z (up)
    artists = []
    for i in range(3):
        tip = xyz + R @ axes[i]
        (line,) = ax.plot([xyz[0], tip[0]], [xyz[1], tip[1]], [xyz[2], tip[2]],
                          color=colors[i], lw=2.2, zorder=10)
        artists.append(line)
    return artists


def plot_flight_3d(ax, trees: np.ndarray, path: np.ndarray, title: str = "") -> None:
    setup_scene(ax, trees)
    z = np.full(len(path), FLY_Z)
    ax.plot(path[:, 0], path[:, 1], z, color=RL, lw=2.6, zorder=6)
    ax.plot(path[:, 0], path[:, 1], np.zeros(len(path)), color=RL, lw=0.9, alpha=0.28)
    ax.scatter(path[-1, 0], path[-1, 1], FLY_Z, s=40, c=RL, edgecolors="white", zorder=7)
    if title:
        ax.set_title(title, pad=2)


def _setup_attitude_scene(ax, trees: np.ndarray, ref_xy: np.ndarray, *,
                          elev: float = 26, azim: float = -100) -> None:
    """Tier-A analog of ``setup_scene``: bounds come from the local reference
    trajectory, not the global 10x10 ``DOMAIN`` -- gauntlet scenarios live in
    their own small patch of the forest."""
    for t in trees:
        _cylinder(ax, t[0], t[1], alpha=0.45)
    xs, ys = ref_xy[:, 0], ref_xy[:, 1]
    pad = 0.4
    lo = min(xs.min(), ys.min()) - pad
    hi = max(xs.max(), ys.max()) + pad
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(0, TREE_H * 0.55)
    ax.set_box_aspect((1, 1, 0.35))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.02)


def animate_attitude_flight(trees: np.ndarray, ref_xy: np.ndarray, xyz: np.ndarray,
                            attitudes, out_path, fps: int = 15,
                            show_lidar: bool = True, title: str = "") -> None:
    """GIF: a recorded Tier-A rollout (``xyz``/``attitudes`` from
    ``greedy_rollout_*(..., record=True)``) flown through its gauntlet, with
    the body-frame gizmo and rotating 3D LIDAR fan redrawn every frame."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    from tier_a.env_attitude.lidar3d import endpoints3d, scan3d

    fig = plt.figure(figsize=(6.6, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    n = len(xyz)

    def frame(i):
        ax.clear()
        _setup_attitude_scene(ax, trees, ref_xy, azim=-100 + 0.3 * i)
        ax.plot(ref_xy[:, 0], ref_xy[:, 1], np.full(len(ref_xy), FLY_Z),
                color=GOAL, lw=1.0, ls="--", alpha=0.5, zorder=4)
        p = xyz[i]
        phi, theta, psi = attitudes[i]
        ax.plot(xyz[: i + 1, 0], xyz[: i + 1, 1], xyz[: i + 1, 2],
                color=RL, lw=2.2, zorder=6)
        if show_lidar:
            d = scan3d(p, phi, theta, psi, trees)
            e = endpoints3d(p, phi, theta, psi, d)
            for ex, ey, ez in e:
                ax.plot([p[0], ex], [p[1], ey], [p[2], ez], color=LIDARC,
                        lw=0.5, alpha=0.55)
        draw_attitude_gizmo(ax, p, phi, theta, psi, length=0.4)
        ax.scatter(p[0], p[1], p[2], s=55, c=TRAP, edgecolors="white",
                   linewidths=1.0, depthshade=False, zorder=9)
        ax.set_title(f"{title} step {i}/{n - 1}", pad=2)

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 // fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out_path}")


def animate_flight(trees: np.ndarray, path: np.ndarray, out_path, fps: int = 10,
                   show_lidar: bool = True) -> None:
    """GIF: the drone flies the learned policy, LIDAR sweeping as it goes."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(6.6, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    n = len(path)

    def frame(i):
        ax.clear()
        setup_scene(ax, trees, azim=-128 + 0.35 * i)
        p = path[i]
        ax.plot(path[: i + 1, 0], path[: i + 1, 1], np.full(i + 1, FLY_Z),
                color=RL, lw=2.4, zorder=6)
        ax.plot(path[: i + 1, 0], path[: i + 1, 1], np.zeros(i + 1),
                color=RL, lw=0.8, alpha=0.25)
        if show_lidar:
            d = scan(p, trees)
            e = endpoints(p, d)
            for k, (ex, ey) in enumerate(e):
                ax.plot([p[0], ex], [p[1], ey], [FLY_Z, FLY_Z], color=LIDARC,
                        lw=0.5, alpha=0.55)
        ax.scatter(p[0], p[1], FLY_Z, s=55, c=TRAP, edgecolors="white",
                   linewidths=1.0, depthshade=False, zorder=9)
        ax.set_title(f"learned greedy policy — step {i}/{n - 1}", pad=2)

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 // fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out_path}")
