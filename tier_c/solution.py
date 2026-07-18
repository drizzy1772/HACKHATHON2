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
from kinematic_autopilot import AutopilotState

Vec2 = Tuple[float, float]


# ═══════════════════════════ ШАР 1 — A*: глобальне планування шляху ═══════════════

def _block_wreck_box(grid, md, cfg, b: float, cs: float, nx: int, ny: int) -> int:
    """Позначити зайнятими клітинки всередині СПРАВЖНЬОГО боксa уламка техніки.

    Уламок (`md.wreck_index`) має орієнтований хітбокс зі зсунутим центром, а
    astar2d бачить лише коло радіуса r — тож A* планував крізь нього. Повертає
    к-ть домальованих клітинок (0, якщо на мапі уламка немає)."""
    from game_env.generator import OBJECT_FOOTPRINTS, point_in_oriented_box, wreck_yaw

    wi = getattr(md, "wreck_index", None)
    wk = getattr(md, "wreck_kind", None)
    if wi is None or wk not in OBJECT_FOOTPRINTS:
        return 0

    tx, ty = float(md.trees[wi][0]), float(md.trees[wi][1])
    yaw = wreck_yaw(tx, ty)
    fp = OBJECT_FOOTPRINTS[wk]
    margin = cfg.drone_radius + 0.3 + 0.5 * cs   # запас: корпус + як у astar2d + півклітинки

    added = 0
    for j in range(ny):
        for i in range(nx):
            if grid[j][i]:
                continue
            cx, cy = cell_to_world(i, j, b, cs)
            if point_in_oriented_box(cx, cy, tx, ty, yaw, fp, margin=margin):
                grid[j][i] = 1
                added += 1
    return added

_NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
               (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def _astar_grid(grid, nx, ny, b, cs, start_cell, goal_cell):
    """A* між двома клітинками (без зрізання кутів). Світові точки або None."""
    def free(i, j):
        return 0 <= i < nx and 0 <= j < ny and not grid[j][i]
    if not free(*start_cell) or not free(*goal_cell):
        return None

    def h(i, j):
        return math.hypot(i - goal_cell[0], j - goal_cell[1])
    g_cost = {start_cell: 0.0}
    came_from = {}
    open_heap = [(h(*start_cell), start_cell)]
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal_cell:
            cells = [cur]
            while cur in came_from:
                cur = came_from[cur]
                cells.append(cur)
            cells.reverse()
            return [cell_to_world(i, j, b, cs) for (i, j) in cells]
        ci, cj = cur
        for di, dj, step in _NEIGHBOURS:
            ni, nj = ci + di, cj + dj
            if not free(ni, nj):
                continue
            if di != 0 and dj != 0 and not free(ci + di, cj) and not free(ci, cj + dj):
                continue                            # без зрізання кутів між двома деревами
            new_g = g_cost[cur] + step
            if new_g < g_cost.get((ni, nj), math.inf):
                g_cost[(ni, nj)] = new_g
                came_from[(ni, nj)] = cur
                heapq.heappush(open_heap, (new_g + h(ni, nj), (ni, nj)))
    return None


def _free_cell_near(grid, nx, ny, wx, wy, b, cs):
    """Найближча ВІЛЬНА клітинка до світової точки (wx, wy)."""
    ti, tj = world_to_cell(wx, wy, b, cs)
    best, bd = None, float("inf")
    for j in range(ny):
        for i in range(nx):
            if not grid[j][i]:
                d = (i - ti) ** 2 + (j - tj) ** 2
                if d < bd:
                    bd, best = d, (i, j)
    return best


def find_path(md, cfg, cell_size: float = 1.0) -> Optional[List[Vec2]]:
    grid, cs, nx, ny = build_occupancy_grid(md, cfg, cell_size)
    b = cfg.bounds
    _block_wreck_box(grid, md, cfg, b, cs, nx, ny)   # див. коментар у _block_wreck_box

    start = world_to_cell(md.start[0], md.start[1], b, cs)
    goal = world_to_cell(md.checkpoints[0][0], md.checkpoints[0][1], b, cs)

    # Прямий БЕЗПЕЧНИЙ маршрут (він і дає 33/33). Дві точки місії — на самому шляху:
    # зарядка на 1/3, аптечка на 2/3. Дрон не з'їжджає, а зупиняється на них дорогою.
    direct = _astar_grid(grid, nx, ny, b, cs, start, goal)
    if direct is None:
        _mission_reset(None, None, None)
        return None
    n = len(direct)
    charge = tuple(direct[n // 3]) if n >= 6 else None
    pickup = tuple(direct[2 * n // 3]) if n >= 6 else None
    _mission_reset(direct, charge, pickup)
    return direct


# ═══════════════════════════ БАТАРЕЯ + ЗАРЯДНА СТАНЦІЯ ════════════════════════════
# Дрон витрачає заряд на політ. Якщо поточного заряду НЕ ВИСТАЧАЄ, щоб дійти до
# цілі, він летить на зарядну станцію (найближча до центру вільна клітинка),
# заряджається до 100% і продовжує до цілі. Стан живе в модулі — обидві функції
# (рішення в compute_desired_direction, витрата в step_autopilot) його бачать.
BATTERY_DRAIN = 2.2        # % заряду на метр шляху
BATTERY_CHARGE = 2.0       # % за тік на зарядній станції
BATTERY_LEVELS = (100, 85, 65, 45, 35, 25, 15)     # рівні індикатора
ARRIVE_R = 3.5             # радіус «дрон над точкою місії», м
GRAB_CLEARANCE = 0.9       # висота над землею під час забору аптечки, м (опускаємось)
GRAB_TICKS = 20            # скільки тіків тримати над аптечкою (~0.7 с)
CHARGE_CLEARANCE = 0.5     # ПРИЗЕМЛЯЄМОСЬ на станції (низько над землею), м
CHARGE_DWELL_TICKS = 90    # тримаємось на станції ~3 с (90 тіків @ 30 Гц)

# МІСІЯ (5 фаз): to_charge → charging → to_pickup → grabbing → to_goal.
# Старт → ЗАРЯДКА (окрема точка) → АПТЕЧКА (окрема точка, опускаємось+беремо,
# вона ПРИЛИПАЄ) → ЦІЛЬ (везе людині).
_BAT = {"level": 100.0, "mode": "to_goal", "charge": None, "pickup": None,
        "charge2": None, "home": None, "done": False,
        "mark": 100, "carrying": False, "dwell": 0, "cdwell": 0,
        "trick": 0, "trick_dur": 1, "trick_spins": 0}


def _mission_reset(path, charge, pickup) -> None:
    """Новий прогін: повний заряд, точки місії, ще не несемо. home = старт;
    charge2 = зарядка НА ЗВОРОТНОМУ шляху (середина маршруту)."""
    _BAT.update(level=100.0, mark=100, carrying=False, dwell=0, cdwell=0, done=False, trick=0)
    _BAT["charge"] = tuple(charge) if charge else None
    _BAT["pickup"] = tuple(pickup) if pickup else None
    _BAT["home"] = tuple(path[0]) if (pickup and path and len(path) >= 2) else None
    _BAT["charge2"] = tuple(path[len(path) // 2]) if (pickup and path and len(path) >= 6) else None
    _BAT["mode"] = "to_charge" if _BAT["charge"] else ("to_pickup" if _BAT["pickup"] else "to_goal")


def mission_charge(md, cfg, cell_size: float = 1.0):
    """Точка зарядки НА ШЛЯХУ ТУДИ (для зеленої платформи у scene.py)."""
    find_path(md, cfg, cell_size)
    return _BAT["charge"] or (float(md.start[0]), float(md.start[1]))


def mission_charge2(md, cfg, cell_size: float = 1.0):
    """Точка зарядки НА ЗВОРОТНОМУ ШЛЯХУ (друга зелена платформа)."""
    find_path(md, cfg, cell_size)
    return _BAT["charge2"]           # None, якщо місії нема


def mission_pickup(md, cfg, cell_size: float = 1.0):
    """Точка аптечки (для маркера у scene.py)."""
    find_path(md, cfg, cell_size)
    return _BAT["pickup"] or (float(md.checkpoints[0][0]), float(md.checkpoints[0][1]))


# ═══════════════════ ШАР 2 — APF: реактивне уникнення перешкод ════════════════════

@dataclass
class APFParams:
    """Стартовий шаблон параметрів — поля можна міняти/додавати/видаляти
    вільно, це ЛИШЕ ваш власний тюнинг для compute_desired_direction нижче."""
    k_attract: float = 3.0          # притягання ДОМІНУЄ — веде дрон по безпечному A*-шляху
    k_repulse: float = 3.5          # > k_attract: впритул дерево ПЕРЕМАГАЄ притягання (лікує seed 14)
    influence_radius: float = 2.5   # d0 — реагуємо ЛИШЕ на близькі дерева: гладше (−31% поворотів)
    lookahead: float = 4.0          # (референсний carrot-chasing; наш трекер його не вживає)
    look_straight: float = 7.0      # НАШ трекер: lookahead на прямій (кривина 0°)
    look_turn: float = 2.0          # НАШ трекер: lookahead на «крутому» повороті
    turn_angle_deg: float = 45.0    # що вважати крутим: grid-A* дає СХОДИНКИ по 45°,
                                    # кутів 90° на шляху не буває взагалі (виміряно)
    curve_window: float = 6.0       # на скільки метрів уперед міряємо кривину
    simplify_eps: float = 0.3       # Дуглас–Пекер: 44 точки -> 6, сходинки геть,
                                    # лишаються справжні повороти (>0.8 зрізало б усе)
    stuck_boost_factor: float = 3.0 # у скільки разів підсилити притягання при застряганні
    stuck_boost_duration: float = 3.0  # тривалість бусту, с


class CurvatureTracker:
    """НАША заміна carrot-chasing: pure-pursuit зі ЗМІННИМ lookahead за кривиною.

    Наданий PathTracker тримає lookahead СТАЛИМ — це компроміс: великий зрізає
    кути (небезпечно), малий гальмує на прямих. Тут lookahead обирається щотіка:
      кривина = максимальний кут між сегментами у вікні curve_window метрів;
      0° (прямо) -> look_straight ;  90° (крутий поворот) -> look_turn ;  між — лінійно.
    Прогрес монотонний (сегмент-курсор не йде назад), як і в референсі.
    """

    def __init__(self, path: List[Vec2], simplify_eps: float = 0.5):
        self.path = path                                   # сирий A* (для звірки)
        # Сходинки grid-A* (зигзаг ▄▀▄▀ по 45°) — АРТЕФАКТ сітки, а не повороти.
        # Спрямляємо Дугласом–Пекером: лишаються тільки СПРАВЖНІ злами (обхід дерев),
        # і аж тоді «кривина попереду» означає те, що треба.
        self.track = self._simplify(path, simplify_eps) if simplify_eps > 0 else list(path)
        self.seg = 0

    @staticmethod
    def _simplify(path: List[Vec2], eps: float) -> List[Vec2]:
        """Дуглас–Пекер: викинути точки, ближчі за eps до хорди."""
        if len(path) < 3:
            return list(path)
        keep = [False] * len(path)
        keep[0] = keep[-1] = True
        stack = [(0, len(path) - 1)]
        while stack:
            i0, i1 = stack.pop()
            ax, ay = path[i0]; bx, by = path[i1]
            dx, dy = bx - ax, by - ay
            n = math.hypot(dx, dy)
            worst_d, worst_i = -1.0, -1
            for k in range(i0 + 1, i1):
                px, py = path[k]
                if n < 1e-9:
                    d = math.hypot(px - ax, py - ay)
                else:
                    d = abs(dy * px - dx * py + bx * ay - by * ax) / n   # відстань до прямої
                if d > worst_d:
                    worst_d, worst_i = d, k
            if worst_i > 0 and worst_d > eps:
                keep[worst_i] = True
                stack.append((i0, worst_i)); stack.append((worst_i, i1))
        return [path[i] for i, k in enumerate(keep) if k]

    @staticmethod
    def _closest_on_segment(p: Vec2, a: Vec2, b: Vec2):
        ax, ay = a; bx, by = b; px, py = p
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return a, 0.0
        tt = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return (ax + tt * dx, ay + tt * dy), tt

    def _advance(self, pos: Vec2, search_ahead: int = 4):
        best_d, best_seg, best_t = math.inf, self.seg, 0.0
        hi = min(len(self.track) - 1, self.seg + search_ahead)
        for i in range(self.seg, hi):
            proj, tt = self._closest_on_segment(pos, self.track[i], self.track[i + 1])
            d = math.hypot(pos[0] - proj[0], pos[1] - proj[1])
            if d < best_d:
                best_d, best_seg, best_t = d, i, tt
        self.seg = best_seg
        return best_seg, best_t

    def _curvature_ahead(self, seg: int, window: float) -> float:
        """Максимальний кут між сусідніми сегментами у вікні window метрів, рад."""
        max_ang, dist, i = 0.0, 0.0, seg
        while i < len(self.track) - 2 and dist < window:
            a, b, c = self.track[i], self.track[i + 1], self.track[i + 2]
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
                max_ang = max(max_ang, math.acos(max(-1.0, min(1.0, cos_a))))
            dist += n1
            i += 1
        return max_ang

    def _point_ahead(self, seg: int, tt: float, look: float) -> Vec2:
        ax, ay = self.track[seg]; bx, by = self.track[seg + 1]
        px, py = ax + tt * (bx - ax), ay + tt * (by - ay)
        remaining = look - math.hypot(bx - px, by - py)
        if remaining <= 0.0:
            n = math.hypot(bx - px, by - py)
            f = look / n if n > 1e-9 else 0.0
            return (px + f * (bx - px), py + f * (by - py))
        cx, cy, i = bx, by, seg + 1
        while remaining > 0.0 and i < len(self.track) - 1:
            nx, ny = self.track[i + 1]
            seg_len = math.hypot(nx - cx, ny - cy)
            if seg_len >= remaining:
                f = remaining / seg_len if seg_len > 1e-9 else 0.0
                return (cx + f * (nx - cx), cy + f * (ny - cy))
            remaining -= seg_len
            cx, cy, i = nx, ny, i + 1
        return (cx, cy)

    def lookahead_point(self, pos: Vec2, p: APFParams) -> Vec2:
        if len(self.track) < 2:
            return self.track[-1] if self.track else pos
        seg, tt = self._advance(pos)
        ang = self._curvature_ahead(seg, p.curve_window)
        turn_ref = math.radians(max(1.0, p.turn_angle_deg))   # 45° на grid-A*, не 90°
        f = min(1.0, ang / turn_ref)                          # 0 = прямо, 1 = крутий поворот
        look = p.look_straight + (p.look_turn - p.look_straight) * f
        return self._point_ahead(seg, tt, look)


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

    # 1. ПРИТЯГАННЯ — до «морквини». Замість наданого carrot-chasing вживаємо
    # СВІЙ CurvatureTracker (змінний lookahead за кривиною). Беремо лише маршрут
    # із наданого tracker; сам трекер живе в boost_state (він persist між тіками).
    own = boost_state.get("tracker")
    if own is None or own.path is not tracker.path:
        own = CurvatureTracker(tracker.path, simplify_eps=p.simplify_eps)
        boost_state["tracker"] = own
    carrot_goal = own.lookahead_point(pos, p)
    goal_pt = own.track[-1] if own.track else carrot_goal

    # МІСІЯ (5 фаз): старт → ЗАРЯДКА → АПТЕЧКА → ЦІЛЬ. Дрон НЕ з'їжджає з безпечного
    # A*-шляху (усі точки — на маршруті).
    charge = _BAT["charge"]
    pickup = _BAT["pickup"]
    charge2 = _BAT["charge2"]
    home = _BAT["home"]
    d_charge = math.hypot(charge[0] - px, charge[1] - py) if charge else float("inf")
    d_pick = math.hypot(pickup[0] - px, pickup[1] - py) if pickup else float("inf")
    d_c2 = math.hypot(charge2[0] - px, charge2[1] - py) if charge2 else float("inf")
    mode = _BAT["mode"]
    if mode == "to_charge":
        carrot = carrot_goal                                # транзит ПО БЕЗПЕЧНОМУ шляху
        if d_charge < ARRIVE_R:
            _BAT["mode"] = "charging"                        # прибули на зарядку
    elif mode == "charging":
        carrot = charge                                     # сідаємо й тримаємось на станції
        # перехід (зарядились + витримали 3 с) робить step_autopilot
    elif mode == "to_pickup":
        carrot = carrot_goal                                # транзит ПО БЕЗПЕЧНОМУ шляху
        if d_pick < ARRIVE_R:
            _BAT["mode"] = "grabbing"                        # прибули → опускаємось забирати
    elif mode == "grabbing":
        carrot = pickup                                     # тримаємось над аптечкою
    elif mode == "to_goal":                                 # несемо аптечку до людини
        carrot = carrot_goal
        if home is not None and math.hypot(goal_pt[0] - px, goal_pt[1] - py) < ARRIVE_R:
            _BAT["mode"] = "to_home"                         # ДОСТАВИВ → повертаємось додому
            _BAT["carrying"] = False                         # аптечку віддали людині
            _BAT.update(trick=45, trick_dur=45, trick_spins=2)   # ТРЮК: подвійна бочка-перемога
    else:  # ПОВЕРНЕННЯ ДОДОМУ: to_home → charging_home (зарядка!) → to_home2 → done
        home_tr = boost_state.get("home_tracker")
        if home_tr is None or boost_state.get("home_src") is not tracker.path:
            home_tr = CurvatureTracker(list(reversed(tracker.path)), simplify_eps=p.simplify_eps)
            boost_state["home_tracker"] = home_tr
            boost_state["home_src"] = tracker.path      # зворотний шлях, будуємо ОДИН раз
        if mode == "charging_home":
            carrot = charge2                             # сідаємо й тримаємось (перехід у step)
        else:
            carrot = home_tr.lookahead_point(pos, p)     # летимо зворотним шляхом
            if mode == "to_home" and charge2 is not None and d_c2 < ARRIVE_R:
                _BAT["mode"] = "charging_home"; _BAT["cdwell"] = 0   # ПІДЗАРЯДКА на поверненні
            elif home is not None and math.hypot(home[0] - px, home[1] - py) < ARRIVE_R:
                _BAT["done"] = True                      # повернувся на вишку → місію завершено

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
            # класична форма APF: сила ~1/dist -> ∞ впритул, ~0 на межі influence_radius.
            # Так близьке дерево ЗАВЖДИ перемагає притягання, а далеке не стягує з курсу.
            strength = p.k_repulse * (1.0 / dist - 1.0 / p.influence_radius)
            rx -= strength * math.cos(ang)          # мінус = ПРОТИ напрямку на дерево
            ry -= strength * math.sin(ang)
    fx += rx
    fy += ry

    # 3. ЗАСТРЯГАННЯ — штовхаємо вбік (перпендикулярно до напрямку на ціль)
    is_stuck = stuck.update(t, px, py)
    boosting = t < boost_state.get("until", float("-inf"))
    if is_stuck and not boosting:
        # ПОЧАТОК ривка: обираємо бік перпендикуляра ОДИН раз і ФІКСУЄМО на весь буст
        perp_x, perp_y = -ay, ax                    # перпендикуляр до напрямку на ціль
        if perp_x * rx + perp_y * ry < 0:           # бік, що ДАЛІ від дерева
            perp_x, perp_y = -perp_x, -perp_y
        boost_state["until"] = t + p.stuck_boost_duration
        boost_state["perp"] = (perp_x, perp_y)      # запам'ятати напрямок ривка
        boosting = True
    boosted = boosting
    if boosted:
        perp_x, perp_y = boost_state.get("perp", (-ay, ax))   # той самий бік, без миготіння
        fx += p.k_attract * p.stuck_boost_factor * perp_x
        fy += p.k_attract * p.stuck_boost_factor * perp_y

    # 4. ЗУПИНКА НА ТОЧЦІ МІСІЇ: на зарядці або над аптечкою й дуже близько —
    # глушимо мотори (завис), щоб дрон спокійно стояв, а не кружляв.
    if ((_BAT["mode"] == "grabbing" and d_pick < 1.0)
            or (_BAT["mode"] == "charging" and d_charge < 1.0)
            or (_BAT["mode"] == "charging_home" and d_c2 < 1.0)):
        return (0.0, 0.0), boosted, carrot

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
        self._history.append((t, x, y))

        # лишаємо тільки останні window_s секунд
        cutoff = t - self.window_s
        self._history = [pt for pt in self._history if pt[0] >= cutoff]

        # чекаємо, поки вікно наповниться на повний window_s (інакше — не суди)
        if t - self._history[0][0] < self.window_s * 0.9:
            return False

        # центр міні-зони і найбільший радіус від нього
        cx = sum(pt[1] for pt in self._history) / len(self._history)
        cy = sum(pt[2] for pt in self._history) / len(self._history)
        max_r = max(math.hypot(pt[1] - cx, pt[2] - cy) for pt in self._history)

        # якщо весь рух умістився в коло < radius — дрон тупцює = застряг
        return max_r < self.radius


# ═══════════════════ ШАР 4 — кінематичний автопілот-виконавець ════════════════════

@dataclass
class AutopilotParams:
    """Стартовий шаблон параметрів виконання — поля можна міняти/додавати
    вільно, це ЛИШЕ ваш власний тюнинг для step_autopilot нижче."""
    max_speed: float = 6.0           # цільова крейсерська швидкість, м/с (−51% часу, безпека не просіла)
    max_accel: float = 4.0           # макс. прискорення, м/с²
    alt_clearance: float = 2.5       # цільова висота НАД рельєфом, м
    max_climb_rate: float = 2.0      # макс. вертикальна швидкість, м/с
    yaw_rate_max: float = 4.0        # макс. кутова швидкість курсу, рад/с
    slow_radius: float = 3.0         # з якої відстані до дерева починаємо гальмувати, м
    min_speed_frac: float = 0.25     # мін. частка швидкості впритул до дерева


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
    dx, dy = direction_xy
    mag = math.hypot(dx, dy)

    # 1. ГОРИЗОНТАЛЬНА ШВИДКІСТЬ — повна в бажаному напрямку, але гальмуємо біля дерев
    if mag < 1e-9:
        vx = vy = 0.0
        speed = 0.0
    else:
        dx, dy = dx / mag, dy / mag                     # одиничний напрям
        slow = 1.0
        if min_lidar is not None and min_lidar < p.slow_radius:
            # лінійно: повна швидкість на межі slow_radius → min_speed_frac упритул
            slow = p.min_speed_frac + (1.0 - p.min_speed_frac) * (min_lidar / p.slow_radius)
            slow = max(p.min_speed_frac, min(1.0, slow))
        speed = p.max_speed * slow

        # #2: не даємо горизонталі обганяти набір висоти. Пробуємо нахил рельєфу
        # попереду; якщо він крутіший, ніж max_climb_rate дозволяє на цій швидкості —
        # ріжемо швидкість, щоб вертикаль встигала (інакше врізались би в схил).
        probe = max(0.5, speed * dt)
        ground_here = terrain.height_at(state.x, state.y)
        ground_ahead = terrain.height_at(state.x + dx * probe, state.y + dy * probe)
        slope = (ground_ahead - ground_here) / probe          # підйом на метр шляху
        if slope > 1e-3:
            speed_cap = p.max_climb_rate / slope              # швидкість, за якої vz встигає
            speed = min(speed, speed_cap)

        vx, vy = dx * speed, dy * speed

    # інтегруємо горизонтальну позицію
    x = state.x + vx * dt
    y = state.y + vy * dt

    # 2. ВИСОТА — тримаємо alt_clearance; над аптечкою опускаємось забрати;
    #    на зарядці СІДАЄМО (низько над землею).
    if _BAT["mode"] == "grabbing":
        clear = GRAB_CLEARANCE
    elif _BAT["mode"] in ("charging", "charging_home"):
        clear = CHARGE_CLEARANCE
    else:
        clear = p.alt_clearance
    target_z = terrain.height_at(x, y) + clear
    dz_max = p.max_climb_rate * dt
    dz = max(-dz_max, min(dz_max, target_z - state.z))  # обмежений набір/спуск висоти
    z = state.z + dz
    vz = dz / dt if dt > 1e-9 else 0.0

    # 3. КУРС + ВІЗУАЛЬНИЙ НАХИЛ — плавно довертаємось у бік руху, банкуючи в поворот
    yaw, pitch, roll = state.yaw, state.pitch, state.roll
    if mag > 1e-9:
        target_yaw = math.atan2(dy, dx)
        err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))  # найкоротший
        max_dyaw = p.yaw_rate_max * dt
        applied = max(-max_dyaw, min(max_dyaw, err))
        yaw += applied
        pitch = -0.15 * (speed / p.max_speed)           # ніс униз у русі — «летить уперед»
        roll = -0.20 * (applied / (max_dyaw + 1e-9))    # крен у бік повороту

    # МІСІЯ/БАТАРЕЯ: на зарядці — заряджаємось; над аптечкою — опускаємось і ЗАБИРАЄМО;
    # інакше — витрачаємо заряд на пройдений шлях.
    if _BAT["mode"] in ("charging", "charging_home"):
        _BAT["level"] = min(100.0, _BAT["level"] + BATTERY_CHARGE)
        landed = (z - terrain.height_at(x, y)) <= CHARGE_CLEARANCE + 0.25
        if landed:                                   # сіли — рахуємо 3 с витримки
            _BAT["cdwell"] += 1
            if _BAT["cdwell"] >= CHARGE_DWELL_TICKS and _BAT["level"] >= 99.9:
                if _BAT["mode"] == "charging":
                    _BAT["mode"] = "to_pickup" if _BAT["pickup"] else "to_goal"
                else:                                # charging_home → летимо далі додому
                    _BAT["mode"] = "to_home2"
                _BAT.update(trick=25, trick_dur=25, trick_spins=1)   # ТРЮК: бочка на злеті зі станції
    elif _BAT["mode"] == "grabbing":
        low_enough = (z - terrain.height_at(x, y)) <= GRAB_CLEARANCE + 0.25
        if low_enough:
            _BAT["dwell"] += 1
            if _BAT["dwell"] >= GRAB_TICKS:
                _BAT["carrying"] = True                  # ЗАБРАЛИ аптечку (вона прилипає)
                _BAT["mode"] = "to_goal"                 # → веземо до людини
                _BAT.update(trick=30, trick_dur=30, trick_spins=1)   # ТРЮК: бочка на радощах
    else:
        moved = math.hypot(vx, vy) * dt
        _BAT["level"] = max(0.0, _BAT["level"] - BATTERY_DRAIN * moved)
    for m in BATTERY_LEVELS:                         # оновити рівень індикатора
        if _BAT["mark"] > m >= _BAT["level"]:
            _BAT["mark"] = m

    # ТРЮК: бочка — крен обертається на повний(і) оберт(и) за тривалість трюку.
    # Суто візуально (колізії рахуються по x,y,z), тож безпечно й видовищно.
    if _BAT["trick"] > 0:
        frac = (_BAT["trick_dur"] - _BAT["trick"]) / max(1, _BAT["trick_dur"])
        roll = 2.0 * math.pi * _BAT["trick_spins"] * frac
        pitch = -0.25                                # трохи ніс униз для динаміки
        _BAT["trick"] -= 1

    return AutopilotState(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, yaw=yaw, pitch=pitch, roll=roll)
