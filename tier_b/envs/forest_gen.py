"""Seeded-генерація лісу: 3 воріт уздовж +x + дерева-дистрактори.

PURE NUMPY — жодного pybullet/torch. Дефолтні константи дзеркалять
tier_b/config.yaml (тест test_config_matches_defaults у conda-env
стверджує збіг; у головному venv немає yaml, тому дефолти живуть і тут).

Відтворюваність (інваріант 1 кіта): той самий (gap, seed) → байт-у-байт
той самий ліс. Сід епізоду = master_seed·10_000 + episode_idx.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- дефолти = config.yaml["forest"] -----------------------------------------
GATE_X = (3.0, 6.0, 9.0)
GATE_JITTER_Y = 0.8
TREE_RADIUS = 0.20
TREE_HEIGHT = 4.0
N_DISTRACTORS = (6, 10)
DISTRACTOR_X = (1.5, 10.5)
DISTRACTOR_Y = (-3.0, 3.0)
CORRIDOR_CLEARANCE = 1.2
TREE_MIN_SEP = 0.5
CRUISE_Z = 1.0
START_XY = (0.0, 0.0)
FINISH_XY = (12.0, 0.0)

_MAX_ATTEMPTS = 200  # rejection sampling дистракторів


@dataclass(frozen=True)
class Forest:
    """trees: (N,3) — x, y, r кожного стовбура (перші 2·len(gate_x) — ворота).
    gate_centers: (3,2) — центри проходів. gap: вільна ширина проходу G, м."""

    trees: np.ndarray
    gate_centers: np.ndarray
    gap: float
    tree_height: float = TREE_HEIGHT


def episode_seed(master_seed: int, episode_idx: int) -> int:
    return master_seed * 10_000 + episode_idx


def _point_segment_dist(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(pt - (a + t * ab)))


def point_polyline_dist(pt, waypoints_xy) -> float:
    """Мінімальна відстань точки до полілінії маршруту (XY)."""
    pt = np.asarray(pt, dtype=np.float64)
    w = np.asarray(waypoints_xy, dtype=np.float64)
    return min(_point_segment_dist(pt, w[i], w[i + 1]) for i in range(len(w) - 1))


def generate_forest(gap: float, seed: int,
                    gate_x=GATE_X, gate_jitter_y=GATE_JITTER_Y,
                    tree_radius=TREE_RADIUS, tree_height=TREE_HEIGHT,
                    n_distractors=N_DISTRACTORS,
                    distractor_x=DISTRACTOR_X, distractor_y=DISTRACTOR_Y,
                    corridor_clearance=CORRIDOR_CLEARANCE,
                    tree_min_sep=TREE_MIN_SEP,
                    start_xy=START_XY, finish_xy=FINISH_XY) -> Forest:
    """Ворота: два дерева симетрично навколо зсунутого по y центру;
    відстань між ПОВЕРХНЯМИ стовбурів = gap (між центрами: gap + 2r).
    Дистрактори: rejection sampling поза коридором маршруту."""
    rng = np.random.default_rng(seed)

    centers = np.array([[gx, rng.uniform(-gate_jitter_y, gate_jitter_y)]
                        for gx in gate_x])

    trees: list[list[float]] = []
    half = gap / 2.0 + tree_radius  # центр стовбура від центру воріт
    for gx, cy in centers:
        trees.append([gx, cy - half, tree_radius])
        trees.append([gx, cy + half, tree_radius])

    polyline = np.vstack([np.asarray(start_xy)[None, :], centers,
                          np.asarray(finish_xy)[None, :]])

    n = int(rng.integers(n_distractors[0], n_distractors[1] + 1))
    placed = 0
    for _ in range(_MAX_ATTEMPTS):
        if placed >= n:
            break
        x = rng.uniform(*distractor_x)
        y = rng.uniform(*distractor_y)
        pt = np.array([x, y])
        if point_polyline_dist(pt, polyline) < corridor_clearance:
            continue
        if any(np.hypot(x - t[0], y - t[1]) < tree_min_sep + tree_radius + t[2]
               for t in trees):
            continue
        trees.append([x, y, tree_radius])
        placed += 1

    return Forest(trees=np.array(trees, dtype=np.float64),
                  gate_centers=centers, gap=float(gap),
                  tree_height=float(tree_height))
