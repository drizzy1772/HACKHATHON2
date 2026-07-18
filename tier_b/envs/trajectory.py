"""Траєкторія: вейпоінти крізь центри воріт + потенціали для shaping.

PURE NUMPY. Два потенціали (інваріант 4 кіта — potential-based, Φ(terminal)=0
забезпечує САМ shaping-виклик у rewards.py, не цей модуль):

  Φ_prog(s) = −(‖p − w_i‖ + Σ довжин сегментів після w_i)   — прогрес маршрутом
  Φ_gate(s) = −‖p − наступний центр воріт (3D)‖              — притягання до щілини

Індекс вейпоінта — частина марковського стану середовища (пам'ять трекера),
тож Φ лишається функцією стану і policy-інваріантність зберігається.
"""

from __future__ import annotations

import numpy as np

from tier_b.envs.forest_gen import CRUISE_Z, FINISH_XY, START_XY, Forest

PASS_RADIUS = 0.25   # вейпоінт зараховано в цьому радіусі
FINISH_RADIUS = 0.5  # фініш = у цьому радіусі від останнього вейпоінта


def build_waypoints(forest: Forest, start_xy=START_XY, finish_xy=FINISH_XY,
                    z: float = CRUISE_Z) -> np.ndarray:
    """(5,3): старт → центри 3 воріт → фініш, усе на крейсерській висоті."""
    pts = [list(start_xy) + [z]]
    pts += [[gx, gy, z] for gx, gy in forest.gate_centers]
    pts += [list(finish_xy) + [z]]
    return np.array(pts, dtype=np.float64)


class GateTracker:
    """Стан проходження маршруту: наступний вейпоінт, зараховані ворота, фініш.

    Ворота зараховуються при перетині площини x = gate_x, якщо інтерпольований
    y у момент перетину лежить МІЖ центрами стовбурів (|y−cy| ≤ gap/2 + r).
    Проліт крізь дерево обірве епізод аналітичною колізією раніше, ніж бонус
    встигне "збрехати"; обхід воріт збоку бонусу не дає.
    """

    def __init__(self, forest: Forest, waypoints: np.ndarray,
                 pass_radius: float = PASS_RADIUS,
                 finish_radius: float = FINISH_RADIUS):
        self.forest = forest
        self.waypoints = waypoints
        self.pass_radius = pass_radius
        self.finish_radius = finish_radius
        n_gates = len(forest.gate_centers)
        self._gate_wp = list(range(1, 1 + n_gates))  # індекси вейпоінтів-воріт
        self.reset(waypoints[0])

    def reset(self, pos) -> None:
        self.next_wp = 1  # 0 — старт, уже "пройдений"
        self.gates_passed = 0
        self.finished = False
        self._prev = np.asarray(pos, dtype=np.float64).copy()

    # -- потенціали ------------------------------------------------------------
    def remaining_length(self, pos) -> float:
        pos = np.asarray(pos, dtype=np.float64)
        i = min(self.next_wp, len(self.waypoints) - 1)
        d = float(np.linalg.norm(pos - self.waypoints[i]))
        for j in range(i, len(self.waypoints) - 1):
            d += float(np.linalg.norm(self.waypoints[j + 1] - self.waypoints[j]))
        return d

    def phi_prog(self, pos) -> float:
        return -self.remaining_length(pos)

    def phi_gate(self, pos) -> float:
        pos = np.asarray(pos, dtype=np.float64)
        gi = self.gates_passed
        if gi >= len(self.forest.gate_centers):
            target = self.waypoints[-1]
        else:
            target = self.waypoints[self._gate_wp[gi]]
        return -float(np.linalg.norm(pos - target))

    # -- крок ------------------------------------------------------------------
    def update(self, pos) -> dict:
        """Повертає {"gate_passed": bool, "finished": bool}. Викликати раз на крок."""
        pos = np.asarray(pos, dtype=np.float64)
        events = {"gate_passed": False, "finished": False}

        # перетин площини наступних воріт
        if self.gates_passed < len(self.forest.gate_centers):
            gx, cy = self.forest.gate_centers[self.gates_passed]
            r = float(self.forest.trees[2 * self.gates_passed, 2])
            half_between_centers = self.forest.gap / 2.0 + r
            if self._prev[0] < gx <= pos[0]:
                t = (gx - self._prev[0]) / max(pos[0] - self._prev[0], 1e-12)
                y_cross = self._prev[1] + t * (pos[1] - self._prev[1])
                if abs(y_cross - cy) <= half_between_centers:
                    self.gates_passed += 1
                    events["gate_passed"] = True
                    self.next_wp = max(self.next_wp, 1 + self.gates_passed)

        # вейпоінт за радіусом (підстраховка, якщо площину «не перетнули» строго)
        if self.next_wp < len(self.waypoints) - 1:
            if np.linalg.norm(pos - self.waypoints[self.next_wp]) < self.pass_radius:
                self.next_wp += 1

        # фініш
        if not self.finished and \
                np.linalg.norm(pos - self.waypoints[-1]) < self.finish_radius:
            self.finished = True
            self.next_wp = len(self.waypoints) - 1
            events["finished"] = True

        self._prev = pos.copy()
        return events
