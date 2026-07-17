# -*- coding: utf-8 -*-
"""
ГОТОВА ІНФРАСТРУКТУРА для реактивної навігації (ХАКАТОН: сам алгоритм APF —
compute_desired_direction() — реалізують учасники в solution.py, не тут).

Тут лишається лише PathTracker — геометрична утиліта carrot-chasing (веде
lookahead-точку вздовж ламаної шляху), не є частиною завдання: корисна, якщо
оберете класичний підхід «ціль реактивного контролера = точка попереду на
глобальному шляху», а не сам фінальний чекпоінт.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Vec2 = Tuple[float, float]


# ── Прив'язка до глобального шляху (carrot-chasing / pure pursuit) ────────────────

class PathTracker:
    """Веде «морквину» (lookahead-точку) вздовж ламаної шляху. Прогрес МОНОТОННИЙ
    (сегмент-курсор ніколи не рухається назад) — інакше на шумному наближенні
    «морквина» могла б смикатись назад і застрягати сама."""

    def __init__(self, path: List[Vec2]):
        self.path = path
        self.seg = 0                     # індекс поточного сегмента (path[seg]→path[seg+1])

    @staticmethod
    def _closest_on_segment(p: Vec2, a: Vec2, b: Vec2):
        ax, ay = a; bx, by = b; px, py = p
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return a, 0.0
        t = ((px - ax) * dx + (py - ay) * dy) / L2
        t = max(0.0, min(1.0, t))
        return (ax + t * dx, ay + t * dy), t

    def _advance_segment(self, pos: Vec2, search_ahead: int = 4):
        """Знайти найближчу проєкцію на шлях, шукаючи ЛИШЕ вперед від self.seg
        (у вікні search_ahead сегментів) — забезпечує монотонність."""
        best_d = math.inf
        best_seg, best_t = self.seg, 0.0
        hi = min(len(self.path) - 1, self.seg + search_ahead)
        for i in range(self.seg, hi):
            proj, t = self._closest_on_segment(pos, self.path[i], self.path[i + 1])
            d = math.hypot(pos[0] - proj[0], pos[1] - proj[1])
            if d < best_d:
                best_d, best_seg, best_t = d, i, t
        self.seg = best_seg
        return best_seg, best_t

    def lookahead_point(self, pos: Vec2, lookahead: float) -> Vec2:
        """«Морквина»: точка на шляху на відстані lookahead ПОПЕРЕДУ найближчої
        проєкції дрона. Біля кінця шляху — фінальна точка (ціль)."""
        if len(self.path) < 2:
            return self.path[-1] if self.path else pos
        seg, t = self._advance_segment(pos)
        ax, ay = self.path[seg]; bx, by = self.path[seg + 1]
        remaining = lookahead - math.hypot(bx - (ax + t * (bx - ax)), by - (ay + t * (by - ay)))
        i = seg + 1
        cx, cy = bx, by
        while remaining > 0.0 and i < len(self.path) - 1:
            nx, ny = self.path[i + 1]
            seg_len = math.hypot(nx - cx, ny - cy)
            if seg_len >= remaining:
                frac = remaining / seg_len if seg_len > 1e-9 else 0.0
                return (cx + frac * (nx - cx), cy + frac * (ny - cy))
            remaining -= seg_len
            cx, cy = nx, ny
            i += 1
        return (cx, cy)                   # дійшли до кінця шляху раніше, ніж lookahead
