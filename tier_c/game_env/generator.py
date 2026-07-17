# -*- coding: utf-8 -*-
"""
Генератор мапи: розсів лісу, чекпойнти, серіалізація для незмінності.

Мапа = рельєф (heightmap) + дерева-циліндри + послідовність чекпойнтів + точка
старту. Уся генерація детермінована за насінням (random.Random(seed) + шум
рельєфу з тим самим seed), тож seed=2026 завжди дає ту саму арену.

НЕЗМІННІСТЬ ЗМАГАННЯ (ключова вимога ТЗ):
  save_map_to_json / load_map_from_json зберігають і відновлюють РІВНО ту саму
  мапу (позиції та радіуси дерев, heightmap рельєфу, чекпойнти). Під час фіналу
  всі команди грейдяться на одному competition_map.json — абсолютно однаковому.
  JSON є авторитетним джерелом; seed лише відтворює його за потреби.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import SimConfig, DEFAULT_CONFIG
from .terrain import Terrain

# Дерево: (x, y, z_base, radius, height). z_base — висота підошви стовбура (= рельєф).
Tree = Tuple[float, float, float, float, float]
Vec3 = Tuple[float, float, float]
# Тематична перешкода: (kind, x, y, z, radius, collidable).
#   kind ∈ {'ka52', 'patron'};
#   collidable=True → LiDAR бачить і дотик = COLLISION (Ка-52);
#   False → декоративний орієнтир, LiDAR ПРОХОДИТЬ крізь (Патрон).
Obstacle = Tuple[str, float, float, float, float, bool]

# Автозак (assets/wreck_avtozak_source.glb) — єдиний із трьох уламків, чия
# СИРА геометрія (масштаб=1.0) виявилась удвічі-втричі заниженою за реальний
# розмір: виміряно офлайн bmesh-аналізом raw bbox — висота лише ~0.76 м (42%
# людського зросту 1.8 м; рендер-порівняння поруч з еталонним «зростом
# людини» показало явно іграшковий розмір, а не «розчавлений уламок»).
# tank/dshk при тому самому порівнянні виявились уже реалістичними — тому
# масштаб-корекція лише для автозака.
#
# ВИПРАВЛЕНО (офлайн, bmesh): вихідний assets/wreck_avtozak_source.glb мав
# ДВІ копії моделі поруч — цілу непошкоджену й розбиту (з уламками) — з
# порожнім проміжком між ними (~1 м у сирому масштабі). Саме тому уламок на
# мапі виглядав «сплющеним»/здвоєним: усі виміри (у т.ч. попередні RAW_* нижче)
# рахувались по СПІЛЬНОМУ bbox обох копій, а не однієї машини. Цілу копію
# видалено з файлу, лишилась тільки розбита; заразом викрилось, що довга вісь
# машини йде по СИРОМУ Y, не X (на відміну від wreck_tank/wreck_dshk) — мех
# довернуто на 90° при тій самій офлайн-правці, тож RAW_HALF_LEN/WID нижче
# знову означають те саме (X=довжина), що й для решти уламків.
#
# РЕДАГУЙТЕ ЛИШЕ ЦІ ТРИ ЧИСЛА — бажані РЕАЛЬНІ розміри автозака, у метрах.
# Масштаб моделі й хітбокс (нижче) порахуються самі — жодних формул вручну
# рахувати не треба.
AVTOZAK_TARGET_LENGTH = 4.997   # = довжина фури-цистерни (TRUCK_FOOTPRINT)
AVTOZAK_TARGET_WIDTH = 3.735    # = ширина фури (2.490) × 1.5 — фургон ширший
AVTOZAK_TARGET_HEIGHT = 2.699   # = висота фури (truck_height)

# Сирі виміряні розміри ВИПРАВЛЕНОГО assets/wreck_avtozak_source.glb
# (масштаб=1.0, лише розбита копія, довернуто носом на +X; офлайн bmesh-аналіз:
# bbox x:[-1.23,1.12] y:[1.53,3.12]) — НЕ редагувати вручну, лише якщо
# перевимірюєте сам файл наново.
_AVTOZAK_RAW_HALF_LEN = 1.177
_AVTOZAK_RAW_HALF_WID = 0.795
_AVTOZAK_RAW_HEIGHT = 0.733
_AVTOZAK_RAW_CENTER = (-0.055, 2.330)

# Множники масштабу для mesh_models.spawn() — рахуються з AVTOZAK_TARGET_*
# вище, самі ніколи не редагуються напряму.
AVTOZAK_MESH_SCALE = (
    AVTOZAK_TARGET_LENGTH / (_AVTOZAK_RAW_HALF_LEN * 2.0),
    AVTOZAK_TARGET_WIDTH / (_AVTOZAK_RAW_HALF_WID * 2.0),
    AVTOZAK_TARGET_HEIGHT / _AVTOZAK_RAW_HEIGHT,
)


# Уламок техніки — РІВНО ОДНЕ дерево на мапі замінюється (лише візуально в
# scene.py) на один із цих; (radius, height) — height ТОЧНО виміряний (підошва
# на Z=0; для автозака = AVTOZAK_TARGET_HEIGHT), а radius — навмисно
# консервативний (за найдовшою віссю) запас, який і далі використовує
# astar2d.py для планування шляху (більший запас там — безпечно, лише трохи
# менш оптимальний маршрут) та scene.py як fallback. ТОЧНА колізія
# (collision_and_bounds_status у blender_manual.py/sim_headless.py) для цих
# трьох кладів тепер рахує ОРІЄНТОВАНИЙ прямокутник із OBJECT_FOOTPRINTS
# нижче, а не це коло. kind БЕЗ суфікса "_source.glb" (формати файлів — турбота
# лише scene.py/mesh_models).
WRECK_DIMS = {
    "wreck_tank": (3.45, 2.055),
    "wreck_dshk": (2.65, 2.105),
    "wreck_avtozak": (3.25, AVTOZAK_TARGET_HEIGHT),
}

# Усі три уламки — витягнуті/прямокутні силуети, погано апроксимовані
# симетричним колом WRECK_DIMS-радіуса (радіус підганяли «на око» під
# найдовшу вісь — тож на короткій осі коло зайво блокує прольот, який
# насправді вільний). Точні прямокутні "сліди" — виміряно офлайн bmesh-
# аналізом кожного assets/wreck_*_source.glb (локальні координати ДО
# масштабу/повороту на мапі; half_len/half_wid — половина довжини/ширини,
# center — зсув геометричного центру моделі від точки прив'язки (0,0), бо
# жоден із трьох мешів не центрований на власному початку координат).
# Для wreck_avtozak half_len/half_wid/center рахуються з AVTOZAK_MESH_SCALE
# вище — footprint завжди відповідає РЕАЛЬНО розміщеному мешу, навіть якщо
# AVTOZAK_TARGET_* вище зміняться.
OBJECT_FOOTPRINTS = {
    "wreck_avtozak": {
        "half_len": _AVTOZAK_RAW_HALF_LEN * AVTOZAK_MESH_SCALE[0],
        "half_wid": _AVTOZAK_RAW_HALF_WID * AVTOZAK_MESH_SCALE[1],
        "center": (_AVTOZAK_RAW_CENTER[0] * AVTOZAK_MESH_SCALE[0],
                  _AVTOZAK_RAW_CENTER[1] * AVTOZAK_MESH_SCALE[1]),
    },
    "wreck_tank": {"half_len": 2.881, "half_wid": 1.216, "center": (-0.211, -0.093)},
    "wreck_dshk": {"half_len": 2.650, "half_wid": 0.993, "center": (0.0, 0.0)},
}


def wreck_yaw(x: float, y: float) -> float:
    """Детермінований псевдовипадковий курс уламка з (x, y) — ОДНА формула і
    для візуального розміщення (scene.py), і для колізійного боксу уламка
    (collision_and_bounds_status у blender_manual.py/sim_headless.py), щоб
    хітбокс завжди збігався з тим, що видно на екрані."""
    return (x * 12.9898 + y * 78.233) % (2.0 * math.pi)


def point_in_oriented_box(px: float, py: float, x: float, y: float, yaw: float,
                          footprint: dict, margin: float = 0.0) -> bool:
    """Чи точка (px, py) — всередині орієнтованого прямокутника footprint
    (OBJECT_FOOTPRINTS[...] або TRUCK_FOOTPRINT; half_len/half_wid/center у
    ЛОКАЛЬНИХ координатах моделі), розміщеного в (x, y) з курсом yaw
    (0.0 для об'єктів без повороту, напр. фури), + margin (типово радіус
    корпуса дрона/літака)."""
    lx, ly = footprint["center"]
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cx = x + lx * cyaw - ly * syaw
    cy = y + lx * syaw + ly * cyaw
    dx, dy = px - cx, py - cy
    local_x = dx * cyaw + dy * syaw
    local_y = -dx * syaw + dy * cyaw
    return (abs(local_x) < footprint["half_len"] + margin
            and abs(local_y) < footprint["half_wid"] + margin)


# Фура-чекпоінт (assets/checkpoint_truck_source.glb) — та сама проблема:
# SimConfig.truck_len/truck_width раніше не відповідали реальному мешу взагалі
# (8×3.5 м проти виміряних ~5.0×2.49 м — хітбокс на 60%/40% більший за видиму
# машину). truck_len/truck_width у SimConfig тепер = реальні виміряні розміри;
# TRUCK_FOOTPRINT додає ще й зсув локального центру (фура завжди без курсу,
# yaw=0, тож обертання не потрібне — лише зсув).
TRUCK_FOOTPRINT = {"half_len": 2.499, "half_wid": 1.245, "center": (1.501, 0.500)}


@dataclass
class MapData:
    """Повний, серіалізовний опис арени. Приховується від коду учасника."""

    seed: int
    bounds: float
    grid_res: int
    heightmap: np.ndarray
    trees: List[Tree]
    checkpoints: List[Vec3]
    start: Vec3
    obstacles: List[Obstacle] = field(default_factory=list)
    ceiling: float = 0.0            # ефективна стеля = СЕРЕДНЯ висота крон, м
    wreck_index: int = -1           # індекс у trees, замінений на уламок технiки (-1 = немає)
    wreck_kind: str = ""            # ключ WRECK_DIMS ("wreck_tank"/"wreck_dshk"/"wreck_avtozak")

    def terrain(self, cfg: SimConfig) -> Terrain:
        """Відновити об'єкт рельєфу з heightmap (для симулятора/аналітики)."""
        return Terrain.from_heightmap(cfg, self.heightmap)

    # ── Серіалізація ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "meta": {
                "format": "drone-hackathon.map/v2",
                "seed": self.seed,
                "bounds": self.bounds,
                "grid_res": self.grid_res,
                "ceiling": round(float(self.ceiling), 6),
                "coordinate_system": "right-handed, X forward, Y left, Z up",
                "units": "m",
                "tree_fields": ["x", "y", "z_base", "radius", "height"],
                "obstacle_fields": ["kind", "x", "y", "z", "radius", "collidable"],
                "wreck_index": self.wreck_index,
                "wreck_kind": self.wreck_kind,
            },
            "heightmap": [[round(float(v), 6) for v in row] for row in self.heightmap],
            "trees": [[round(float(v), 6) for v in t] for t in self.trees],
            "checkpoints": [[round(float(v), 6) for v in c] for c in self.checkpoints],
            "obstacles": [[o[0]] + [round(float(v), 6) for v in o[1:5]] + [bool(o[5])]
                          for o in self.obstacles],
            "start": [round(float(v), 6) for v in self.start],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapData":
        m = d["meta"]
        trees = [tuple(t) for t in d["trees"]]
        # Стеля: беремо з файлу; для старих мап (v1) обчислюємо як середню крону
        ceiling = float(m.get("ceiling", 0.0))
        if ceiling <= 0.0 and trees:
            ceiling = sum(float(t[4]) for t in trees) / len(trees)
        return cls(
            seed=int(m["seed"]),
            bounds=float(m["bounds"]),
            grid_res=int(m["grid_res"]),
            heightmap=np.asarray(d["heightmap"], dtype=float),
            trees=trees,
            checkpoints=[tuple(c) for c in d["checkpoints"]],
            obstacles=[(str(o[0]), float(o[1]), float(o[2]), float(o[3]),
                        float(o[4]), bool(o[5])) for o in d.get("obstacles", [])],
            ceiling=ceiling,
            start=tuple(d["start"]),
            wreck_index=int(m.get("wreck_index", -1)),
            wreck_kind=str(m.get("wreck_kind", "")),
        )

    def save_map_to_json(self, filename) -> Path:
        """Записати мапу у JSON (авторитетний артефакт для грейдингу)."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_map_from_json(cls, filename) -> "MapData":
        """Прочитати мапу з JSON — відновлює рівно ту саму арену."""
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        return cls.from_dict(data)


class MapGenerator:
    """Детермінована процедурна генерація мапи за насінням."""

    def __init__(self, cfg: SimConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    # ── Публічні точки входу ─────────────────────────────────────────────────────

    def build(self, seed: int | None = None) -> MapData:
        """Згенерувати мапу за seed (типово — змагальне насіння з конфіга).
        УСІ сценарії/стадії (A-D, 5 фур) прибрано — лишається рівно ОДИН рух:
        старт → фіксована ціль на протилежному куті арени (та сама діагональ)."""
        seed = self.cfg.default_seed if seed is None else int(seed)
        rng = random.Random(seed)
        terrain = Terrain.generate(self.cfg, seed)

        start = self._make_start(terrain)
        goal_xy = self._fixed_goal_xy()
        trees = self._scatter_trees(terrain, rng, avoid=[start, (goal_xy[0], goal_xy[1], 0.0)])
        # Стеля = СЕРЕДНЯ висота крон (ТЗ) — рахуємо ДО заміни на уламок, щоб
        # один занижений («приземлений») об'єкт не псував середню висоту лісу.
        ceiling = self._effective_ceiling(trees)
        checkpoints = self._make_checkpoint(terrain, goal_xy, ceiling)
        obstacles = self._make_obstacles(terrain, trees, checkpoints, rng, start, ceiling)
        trees, wreck_index, wreck_kind = self._place_wreck(trees, rng)

        return MapData(
            seed=seed,
            bounds=self.cfg.bounds,
            grid_res=self.cfg.grid_res,
            heightmap=terrain.heightmap.copy(),
            trees=trees,
            checkpoints=checkpoints,
            obstacles=obstacles,
            ceiling=ceiling,
            start=start,
            wreck_index=wreck_index,
            wreck_kind=wreck_kind,
        )

    def _place_wreck(self, trees: List[Tree], rng: random.Random):
        """Замінити РІВНО ОДНЕ випадкове дерево на випадковий уламок техніки:
        (radius, height) у самому кортежі дерева стають габаритом УЛАМКА (не
        дерева) — тож існуюча колізійна перевірка по md.trees (незмінна, і в
        blender_manual.py, і в sim_headless.py) автоматично рахує правильний
        хітбокс, без жодних спеціальних випадків у самій колізії.

        Дерева розсіюються з відступом лише 2 м від краю арени (_scatter_trees) —
        достатньо для тонкого стовбура, але НЕДОСТАТНЬО для уламка техніки
        (радіус 2.65-3.45 м, WRECK_DIMS): підміна дерева впритул до краю
        вилазила б хітбоксом уламка за bounds. Тому кандидатів обираємо лише
        серед дерев, чий центр не ближче за НАЙБІЛЬШИЙ можливий радіус уламка
        до краю — безпечно для БУДЬ-ЯКОГО kind, який випаде далі."""
        if not trees:
            return trees, -1, ""
        b = self.cfg.bounds
        margin = max(r for r, _h in WRECK_DIMS.values())
        candidates = [i for i, (x, y, *_rest) in enumerate(trees)
                     if abs(x) <= b - margin and abs(y) <= b - margin]
        if not candidates:
            return trees, -1, ""   # уся мапа тісна (не мало б траплятись) — без уламка, безпечніше за вихід за межі
        idx = rng.choice(candidates)
        kind = rng.choice(list(WRECK_DIMS))
        r, h = WRECK_DIMS[kind]
        x, y, z_base, _old_r, _old_h = trees[idx]
        trees = list(trees)
        trees[idx] = (x, y, z_base, r, h)
        return trees, idx, kind

    def _effective_ceiling(self, trees: List[Tree]) -> float:
        """Ефективна стеля = середня висота крон дерев (абс. запобіжник, якщо лісу
        немає). Саме її переліт карається «Під дією РЕБ» + дискваліфікацією."""
        if not trees:
            return self.cfg.ceiling
        return sum(float(t[4]) for t in trees) / len(trees)

    def build_competition(self, path="competition_map.json",
                          seed: int | None = None) -> MapData:
        """Змагальна мапа: якщо файл існує — завантажити (незмінність);
        інакше згенерувати за насінням і зберегти його ж."""
        p = Path(path)
        if p.exists():
            return MapData.load_map_from_json(p)
        data = self.build(seed if seed is not None else self.cfg.default_seed)
        data.save_map_to_json(p)
        return data

    # ── Складові генерації ───────────────────────────────────────────────────────

    def _make_start(self, terrain: Terrain) -> Vec3:
        """Старт — у куті арени, на фіксованій висоті над рельєфом."""
        b = self.cfg.bounds
        x, y = -b + 4.0, -b + 4.0
        z = terrain.height_at(x, y) + self.cfg.start_clearance
        return (x, y, z)

    def _fixed_goal_xy(self) -> Tuple[float, float]:
        """Ціль B — ФІКСОВАНИЙ протилежний кут арени від старту (дзеркальний відступ
        4 м від краю), тож обидві точки лежать на одній діагоналі (пряма y=x)."""
        b = self.cfg.bounds
        return (b - 4.0, b - 4.0)

    def _scatter_trees(self, terrain: Terrain, rng: random.Random,
                       avoid: List[Vec3]) -> List[Tree]:
        """Розсів дерев методом відбраковування: жодних перетинів крон, відступ
        від точок avoid (старт). Радіус і висота — випадкові в межах конфіга."""
        cfg = self.cfg
        edge = cfg.bounds - 2.0            # не саджати впритул до краю
        placed: List[Tree] = []
        max_attempts = cfg.n_trees * 60
        attempts = 0

        while len(placed) < cfg.n_trees and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(-edge, edge)
            y = rng.uniform(-edge, edge)
            r = rng.uniform(cfg.tree_r_min, cfg.tree_r_max)

            # Відступ від старту й інших спеціальних точок
            if any(math.hypot(x - ax, y - ay) < r + cfg.start_clearance
                   for ax, ay, _ in avoid):
                continue
            # Без перетину з уже поставленими деревами (з проміжком margin)
            if any(math.hypot(x - tx, y - ty) < r + tr + cfg.tree_margin
                   for tx, ty, _, tr, _ in placed):
                continue

            z_base = terrain.height_at(x, y)
            h = rng.uniform(cfg.tree_h_min, cfg.tree_h_max)
            placed.append((x, y, z_base, r, h))

        return placed

    def _make_checkpoint(self, terrain: Terrain, goal_xy: Tuple[float, float],
                         ceiling: float) -> List[Vec3]:
        """ЄДИНА фіксована ціль-фура на protилежному куті арени (goal_xy) — не
        випадкова: висота лише прив'язується до рельєфу під нею й обрізається
        ЕФЕКТИВНОЮ СТЕЛЕЮ (середня крона), як і раніше. Список із 1 елемента —
        формат MapData.checkpoints лишається сумісним із рештою коду (сцена/HUD)."""
        cfg = self.cfg
        x, y = goal_xy
        cap = max(cfg.cp_clearance + 0.5, ceiling - 0.5)
        z = min(terrain.height_at(x, y) + cfg.cp_clearance, cap)
        return [(x, y, z)]

    def _make_obstacles(self, terrain: Terrain, trees: List[Tree],
                        checkpoints: List[Vec3], rng: random.Random,
                        start: Vec3, ceiling: float) -> List[Obstacle]:
        """Тематичні перешкоди українського контексту (детерміновано за насінням).
        Колізійний (LiDAR бачить, дотик → «Борт втрачено»): Ка-52, завис у повітрі.
        Декоративний орієнтир (LiDAR проходить крізь): пес Патрон."""
        cfg = self.cfg
        edge = cfg.bounds - 3.0
        clr = cfg.obstacle_clear

        def free_spot() -> Tuple[float, float]:
            for _ in range(400):
                x = rng.uniform(-edge, edge)
                y = rng.uniform(-edge, edge)
                if any(math.hypot(x - tx, y - ty) < tr + clr for tx, ty, _, tr, _ in trees):
                    continue
                if any(math.hypot(x - cx, y - cy) < clr for cx, cy, _ in checkpoints):
                    continue
                if math.hypot(x - start[0], y - start[1]) < clr:
                    continue
                return x, y
            return rng.uniform(-edge, edge), rng.uniform(-edge, edge)

        obstacles: List[Obstacle] = []
        # Ка-52 — ЛИШЕ ЗА МЕЖАМИ ігрової арени (не «підстерігає» серед дерев):
        # або ЗБОКУ (кільце одразу за горизонтальними межами bounds), або
        # ЗГОРИ (у межах XY, але далеко вище стелі — недосяжно в нормальному
        # польоті). Радіус хітбокса лишається під реальний розмах гвинта
        # (~15 м) — з таким розміщенням фактичне зіткнення практично
        # неможливе (межі/стеля дискваліфікують раніше, ніж пілот дотягнеться).
        if rng.random() < 0.5:
            # Збоку: арена КВАДРАТНА, тож "за межами" перевіряється ПООСЬОВО
            # (abs(x) > bounds or abs(y) > bounds у collision_and_bounds_status) —
            # РАДІАЛЬНА відстань від центру тут не годиться (по діагоналі
            # обидві координати можуть лишитись під bounds, хоч і далі за
            # bounds·√2 від центру). Тому одну вісь свідомо виносимо за bounds,
            # а вздовж іншої — довільний зсув (може й собі бути за межами, для
            # покриття кутів арени).
            side_axis = rng.choice(("x", "y"))
            side_sign = rng.choice((-1.0, 1.0))
            beyond = side_sign * (cfg.bounds + rng.uniform(5.0, 15.0))
            along = rng.uniform(-cfg.bounds - 10.0, cfg.bounds + 10.0)
            x, y = (beyond, along) if side_axis == "x" else (along, beyond)
            z = terrain.height_at(x, y) + rng.uniform(4.0, 8.0)
        else:
            x = rng.uniform(-edge, edge)
            y = rng.uniform(-edge, edge)
            z = ceiling + rng.uniform(6.0, 12.0)
        obstacles.append(("ka52", x, y, z, 7.5, True))
        # Пес Патрон — талісман ДСНС (ДЕКОРАТИВНИЙ, LiDAR проходить крізь)
        x, y = free_spot()
        obstacles.append(("patron", x, y, terrain.height_at(x, y) + 0.25, 0.4, False))
        return obstacles
