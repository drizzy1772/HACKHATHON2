# -*- coding: utf-8 -*-
"""
ЗАВДАННЯ ХАКАТОНУ: реалізувати автономну навігацію дрона (учасники редагують
ЛИШЕ цей файл — увесь інший код проєкту готовий і не потребує змін).

Дрон стартує в md.start, ціль — чекпоінт-фура на іншому кінці арени
(md.checkpoints[0]), між ними — ліс (дерева) і кілька тематичних перешкод.
sim_headless.simulate() щотіку (30 Гц) викликає ці функції РІВНО в такому
порядку:

  1. find_path(...)             — ОДИН РАЗ на старті прогону: глобальний
                                   маршрут старт→ціль в обхід перешкод.
  2. compute_desired_direction(...) — ЩОТІКУ: за поточною позицією й
                                   лідаром вирішує, у який бік летіти зараз
                                   (плюс StuckDetector.update() усередині —
                                   чи дрон застряг).
  3. step_autopilot(...)         — ЩОТІКУ: перетворює «бажаний напрям» на
                                   фактичний рух (швидкість, висота-по-рельєфу).

Зараз усі чотири — ПОРОЖНІ ЗАГЛУШКИ з безпечним дефолтом: дрон НІКУДИ НЕ
ЛЕТИТЬ (просто висить на місці) — це очікувано, а не помилка. Автономний
політ (кнопка «Запустити автономний» / headless-самоперевірка) чесно
запуститься й відпрацює час до тайм-ауту, просто без руху, доки ви не
допишете логіку.

ГОТОВА ІНФРАСТРУКТУРА (можна й варто використовувати, не є частиною завдання):
  • astar2d.build_occupancy_grid(md, cfg, cell_size) — булева сітка
    зайнятості арени (True = перешкода), якщо захочете власний пошук шляху
    поверх сітки, а не переписувати геометрію заново.
  • apf_controller.PathTracker(path) — веде «морквину» (lookahead-точку)
    вздовж ламаної шляху (метод .lookahead_point(pos, lookahead)); готовий
    інструмент, якщо оберете класичний підхід «ціль APF = точка попереду
    на глобальному шляху», а не сам фінальний чекпоінт.
  • kinematic_autopilot.AutopilotState — КОНТРАКТ стану дрона (поля
    x,y,z,vx,vy,vz,yaw,pitch,roll) — рендер/колізії читають САМЕ ці поля,
    форму міняти не можна, значення — вільно.
  • game_env.lidar2d — формат лідара (масив дистанцій по азимутних бінах),
    той самий сенсор, що бачить дрон і в ручному режимі.

Дозволено імпортувати й довільні сторонні пакети (numpy тощо), і додавати
власні допоміжні функції/класи в цьому ж файлі — головне, щоб сигнатури
чотирьох функцій нижче лишались сумісні (їх викликає sim_headless.py)."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from astar2d import build_occupancy_grid, cell_to_world, world_to_cell

Vec2 = Tuple[float, float]


# ═══════════════════════════ ШАР 1 — A*: глобальне планування шляху ═══════════════

def find_path(md, cfg, cell_size: float = 1.0) -> Optional[List[Vec2]]:

    grid, cs, nx, ny = build_occupancy_grid(md, cfg, cell_size)
    b = cfg.bounds

    start = world_to_cell(md.start[0], md.start[1], b, cs)              # (i, j)
    goal = world_to_cell(md.checkpoints[0][0], md.checkpoints[0][1], b, cs)

    def free(i: int, j: int) -> bool:
        return 0 <= i < nx and 0 <= j < ny and not grid[j][i]

    # оцінка «скільки лишилось до цілі» = пряма відстань у клітинках
    def h(i: int, j: int) -> float:
        return math.hypot(i - goal[0], j - goal[1])

    # 8 сусідів: прямий крок коштує 1, діагональний — √2
    neighbours = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                  (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                  (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]

    g_cost = {start: 0.0}                          # скільки пройдено від старту
    came_from = {}
    open_heap = [(h(*start), start)]               # черга за (пройдено + до цілі)

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            # відновити шлях назад і перевести клітинки у світові точки
            cells = [cur]
            while cur in came_from:
                cur = came_from[cur]
                cells.append(cur)
            cells.reverse()
            return [cell_to_world(i, j, b, cs) for (i, j) in cells]

        ci, cj = cur
        for di, dj, step in neighbours:
            ni, nj = ci + di, cj + dj
            if not free(ni, nj):
                continue
            new_g = g_cost[cur] + step
            if new_g < g_cost.get((ni, nj), math.inf):
                g_cost[(ni, nj)] = new_g
                came_from[(ni, nj)] = cur
                heapq.heappush(open_heap, (new_g + h(ni, nj), (ni, nj)))

    return None                                    # шляху немає → пряма лінія (fallback)


# ═══════════════════ ШАР 2 — APF: реактивне уникнення перешкод ════════════════════

@dataclass
class APFParams:
    """Стартовий шаблон параметрів — поля можна міняти/додавати/видаляти
    вільно, це ЛИШЕ ваш власний тюнинг для compute_desired_direction нижче."""
    k_attract: float = 1.0          # база притягання до цілі/морквини
    k_repulse: float = 6.0          # сила відштовхування від лідара
    influence_radius: float = 4.0   # d0 — далі перешкода не відштовхує (≤ lidar_range)
    lookahead: float = 3.0          # відстань «морквини» вперед по шляху, м
    stuck_boost_factor: float = 3.0 # у скільки разів підсилити притягання при застряганні
    stuck_boost_duration: float = 3.0  # тривалість бусту, с


def compute_desired_direction(pos: Vec2, lidar, bin_angles, tracker, stuck,
                              t: float, p: APFParams, boost_state: dict):
    """TODO(учасник): один тік реактивної навігації. Вхід:
      • pos — поточна (x,y) дрона;
      • lidar, bin_angles — масиви однакової довжини: lidar[i] — відстань до
        найближчої перешкоди в напрямку bin_angles[i] (рад, світова система),
        обрізана на дальність сенсора;
      • tracker — apf_controller.PathTracker(path) з find_path() (path може
        бути й дефолтною прямою лінією, якщо find_path ще не реалізовано);
      • stuck — ваш StuckDetector (нижче) — викликайте stuck.update(t,x,y),
        щоб він накопичував історію й підказував, чи дрон застряг;
      • boost_state — порожній dict, що ЖИВЕ між тіками (мутуйте на місці,
        якщо потрібна власна пам'ять між викликами, напр. таймер бусту).

    Повернути (напрям_xy, boosted, carrot):
      • напрям_xy — ОДИНИЧНИЙ (або нульовий) 2D-вектор бажаного руху;
      • boosted — bool, лише для HUD/телеметрії (чи зараз активний «вихід
        із застрягання»), на фізику не впливає;
      • carrot — (x,y) поточної проміжної цілі, теж лише для діагностики/HUD.

    Класичний орієнтир (не обов'язково саме так): сума «притягання» до
    tracker.lookahead_point(pos, p.lookahead) і «відштовхування» від
    активних (< p.influence_radius) променів лідара, нормалізована у
    одиничний вектор.

    Безпечний дефолт зараз: нульовий напрям — дрон горизонтально не рухається."""
    px, py = pos

    # 1. ПРИТЯГАННЯ — до «морквини» (точки попереду на маршруті)
    carrot = tracker.lookahead_point(pos, p.lookahead)
    ax, ay = carrot[0] - px, carrot[1] - py
    da = math.hypot(ax, ay)
    if da > 1e-9:
        ax, ay = ax / da, ay / da                  # одиничний напрям до цілі
    fx = p.k_attract * ax
    fy = p.k_attract * ay

    # 2. ВІДШТОВХУВАННЯ — геть від кожного дерева, ближчого за influence_radius
    rx, ry = 0.0, 0.0
    for dist, ang in zip(lidar, bin_angles):
        if 1e-6 < dist < p.influence_radius:
            # чим ближче дерево — тим сильніший поштовх (лінійно, 0 на межі)
            strength = p.k_repulse * (p.influence_radius - dist) / p.influence_radius
            rx -= strength * math.cos(ang)          # мінус = ПРОТИ напрямку на дерево
            ry -= strength * math.sin(ang)
    fx += rx
    fy += ry

    # 3. ЗАСТРЯГАННЯ — штовхаємо вбік (перпендикулярно до напрямку на ціль)
    is_stuck = stuck.update(t, px, py)
    if is_stuck:
        boost_state["until"] = t + p.stuck_boost_duration   # запустити таймер бусту
    boosted = t < boost_state.get("until", float("-inf"))
    if boosted:
        perp_x, perp_y = -ay, ax                    # перпендикуляр до напрямку на ціль
        # обрати бік, що ДАЛІ від дерева (у бік вільного простору)
        if perp_x * rx + perp_y * ry < 0:
            perp_x, perp_y = -perp_x, -perp_y
        fx += p.k_attract * p.stuck_boost_factor * perp_x
        fy += p.k_attract * p.stuck_boost_factor * perp_y

    # нормалізувати суму в ОДИНИЧНИЙ вектор
    n = math.hypot(fx, fy)
    if n < 1e-9:
        return (0.0, 0.0), boosted, carrot
    return (fx / n, fy / n), boosted, carrot


# ═══════════════════════ ШАР 3 — детектор застрягання ═════════════════════════════

@dataclass
class StuckDetector:
    """Стартовий шаблон — поля/структуру можна міняти вільно. Мета: за
    історією позицій дрона визначати, що він «застряг» (кружляє/тремтить на
    місці, не просувається до цілі), щоб compute_desired_direction міг якось
    відреагувати (наприклад, тимчасово підсилити притягання)."""
    radius: float = 2.2
    window_s: float = 3.0
    _history: List[Tuple[float, float, float]] = field(default_factory=list)

    def update(self, t: float, x: float, y: float) -> bool:
        """TODO(учасник): накопичити (t,x,y) і повернути True, якщо дрон
        зараз вважається застряглим.

        Безпечний дефолт зараз: завжди False — застрягання ніколи не
        спрацьовує (не критично, поки й сам рух ще не реалізовано)."""
        return False


# ═══════════════════ ШАР 4 — кінематичний автопілот-виконавець ════════════════════

@dataclass
class AutopilotParams:
    """Стартовий шаблон параметрів виконання — поля можна міняти/додавати
    вільно, це ЛИШЕ ваш власний тюнинг для step_autopilot нижче."""
    max_speed: float = 3.0           # цільова крейсерська швидкість, м/с
    max_accel: float = 4.0           # макс. прискорення, м/с²
    alt_clearance: float = 2.5       # цільова висота НАД рельєфом, м
    max_climb_rate: float = 2.0      # макс. вертикальна швидкість, м/с
    yaw_rate_max: float = 4.0        # макс. кутова швидкість курсу, рад/с


def step_autopilot(state, direction_xy: Vec2, terrain, dt: float,
                   p: AutopilotParams, min_lidar: float = None):
    """TODO(учасник): один крок руху дрона. Вхід:
      • state — kinematic_autopilot.AutopilotState (поточний x,y,z,
        vx,vy,vz,yaw,pitch,roll);
      • direction_xy — бажаний напрям від compute_desired_direction();
      • terrain — рельєф мапи: terrain.height_at(x,y) → висота землі в цій
        точці (для утримання висоти НАД рельєфом, а не абсолютної);
      • dt — крок часу, с;
      • min_lidar — відстань до найближчої перешкоди зараз (None, якщо
        невідомо) — можна використати для сповільнення біля перешкод.

    Повернути НОВИЙ AutopilotState (той самий тип полів — рендер і колізії
    читають їх напряму).

    Безпечний дефолт зараз: стан не змінюється — дрон висить на місці."""
    return state
