"""34-променевий 3D LIDAR: точна конструкція напрямків (за ТЗ користувача).

Напрямки — PURE NUMPY (тестується без pybullet); сам скан — обгортка над
p.rayTestBatch, pybullet імпортується ліниво.

Конструкція (34 = 8 + 6 + 4 + 16):
  1) площина XOY: 8 променів кожні 45°
  2) площина YOZ: 8 позицій, але ±y уже пораховані в XOY → 6 нових
  3) площина XOZ: 8 позицій, але ±x (XOY) і ±z (YOZ) пораховані → 4 нових
  4) 4 «бісектрисні» великі кола × 4 нових промені:
       площини x=±y (містять вісь z) → усі 8 напрямків (±½, ±½, ±√2/2)
       площини y=±z (містять вісь x) → усі 8 напрямків (±√2/2, ±½, ±½)

Свідома анізотропія: бісектрисні кола обрано крізь вісь x (напрям польоту)
та вісь z — покриття зміщене туди, де дерева з'являються під час заходу.
Чесне обмеження (у README): крок 45° гарантує «бачити» стовбур r=0.2 м лише
з ~0.5–1 м; глобальну навігацію несе вейпоінт-спостереження, LIDAR — останній метр.
"""

from __future__ import annotations

import numpy as np

N_RAYS = 34
MAX_RANGE = 4.0  # м; = config.yaml["lidar"]["max_range"]

# бітова група дерев для фільтра колізій: промені бачать ЛИШЕ дерева
TREE_GROUP = 2


def ray_dirs_34() -> np.ndarray:
    """(34,3) одиничних напрямків у ТІЛЕСНІЙ системі координат, детермінований порядок."""
    dirs: list[list[float]] = []

    # 1) XOY
    for k in range(8):
        a = k * np.pi / 4.0
        dirs.append([np.cos(a), np.sin(a), 0.0])
    # 2) YOZ без ±y (k=0 → +y, k=4 → −y)
    for k in range(8):
        if k in (0, 4):
            continue
        a = k * np.pi / 4.0
        dirs.append([0.0, np.cos(a), np.sin(a)])
    # 3) XOZ без ±x (k=0,4) і ±z (k=2,6)
    for k in (1, 3, 5, 7):
        a = k * np.pi / 4.0
        dirs.append([np.cos(a), 0.0, np.sin(a)])
    # 4a) площини x=±y: u=(±s,s,0), v=(0,0,1), нові θ = 45°,135°,225°,315°
    s = np.sqrt(2.0) / 2.0
    for ux, uy in ((s, s), (s, -s)):
        for t in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
            ct, st = np.cos(t), np.sin(t)
            dirs.append([ct * ux, ct * uy, st])
    # 4b) площини y=±z: u=(1,0,0), v=(0,s,±s)
    for vy, vz in ((s, s), (s, -s)):
        for t in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
            ct, st = np.cos(t), np.sin(t)
            dirs.append([ct, st * vy, st * vz])

    out = np.array(dirs, dtype=np.float64)
    # чистимо −0.0 та мікрошум, щоб golden-тест був байт-стабільний
    out[np.abs(out) < 1e-12] = 0.0
    assert out.shape == (N_RAYS, 3)
    return out


RAY_DIRS = ray_dirs_34()


def scan(client: int, pos, quat_xyzw, max_range: float = MAX_RANGE) -> np.ndarray:
    """(34,) hitFraction ∈ [0,1]; 1.0 = промінь нічого не зустрів.

    Промені летять із тілесної системи у світову (R @ dir) і через
    collisionFilterMask=TREE_GROUP бачать лише дерева (не корпус, не землю).
    """
    import pybullet as p

    rot = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    world = RAY_DIRS @ rot.T
    origin = np.asarray(pos, dtype=np.float64)
    froms = np.tile(origin, (N_RAYS, 1))
    tos = origin[None, :] + world * max_range
    res = p.rayTestBatch(froms.tolist(), tos.tolist(),
                         collisionFilterMask=TREE_GROUP,
                         physicsClientId=client)
    return np.array([r[2] for r in res], dtype=np.float64)
