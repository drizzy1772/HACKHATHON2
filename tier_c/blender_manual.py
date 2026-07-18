# -*- coding: utf-8 -*-
"""
Live-in-Blender РУЧНИЙ політ у СЕРЕДОВИЩІ коміту 2e2bc5d.

Реалістичний рушій (sandbox/realistic_flight) крутиться всередині Blender і керується
з клавіатури, а СЦЕНА — та сама «ігрова» арена, що була на 2e2bc5d: рельєф-пагорби,
ліс (дерева по нормалі), фури-чекпоінти з напівпрозорими зонами-прольотами, тематичні
перешкоди (Ка-52 виключно в небі, пес Патрон) та випадковий уламок техніки в лісі.
Сцена будується оригінальним кодом (game_env, узятий з 2e2bc5d).

Запуск (інтерактивно):
    /Applications/Blender.app/Contents/MacOS/Blender -P blender_manual.py
Самоперевірка (headless):
    /Applications/Blender.app/Contents/MacOS/Blender -b -P blender_manual.py

Керування: W/S — тангаж · A/D — крен · Q/E — курс · Space/Shift — газ · R — скид ·
           C — камера (Chase↔FPV). Лише англійська (латинська) розкладка клавіатури —
           кирилична розкладка для керування НАВМИСНО заблокована: event.unicode на
           PRESS засікає кириличний символ на фізичному місці керуючої клавіші й просто
           ІГНОРУЄ цю подію (а не транслює її, як раніше) — див. LATIN_CONTROL_KEYS /
           _CYRILLIC_CONTROL_CHARS нижче.

Бокова панель (N-панель 3D-в'юпорта, вкладка «Дрон»): роль Адмін/Учасник (пароль),
підказка керування, кнопки Старт/Скид/Камера, параметри мапи (seed, к-сть дерев) з
перегенерацією та випадковим сідом — усе під замком «Мапа», лише для Адміна.
На старті панель/тулбар/шапка сховані (лише в'юпорт) — відкрити назад: Ctrl+Space,
або штатна клавіша N (Blender сам відкриває/закриває N-панель — гра її не перехоплює).

Камера ЛИШЕ від першої (FPV) або третьої (Chase) особи — вільної орбіти мишею немає:
щотіку в'юпорт примусово повертається у вигляд активної камери.

HUD (оверлей у в'юпорті) — ДВІ незалежні частини:
  1. Телеметрійна панель угорі ліворуч — дослівно draw_hud() пісочниці engine_test.py:
     газ %, висота, верт. швидкість, швидкість, тангаж/крен/курс + стовпчик газу з
     міткою висоти зависання, підказка керування й поточна камера внизу екрана.
  2. Статусний оверлей (дослівно за _draw_status_overlay коміту 2e2bc5d):
       • КОЛІЗІЯ (дотик рельєфу/дерева/тематичної перешкоди/кузова фури) → «БОРТ ВТРАЧЕНО»;
       • МЕЖІ (переліт стелі АБО виліт за горизонтальні межі арени) → «ПІД ДІЄЮ РЕБ»
         + напівпрозорий білий шум на в'юпорт.
При терміналньому статусі фізика ЗАМОРОЖУЄТЬСЯ (дрон завис на місці); R — скид у старт.

Орієнтація дрона = матриця рушія Rz(yaw)·Ry(pitch)·Rx(roll), через matrix_world.
УВАГА: керування РУЧНЕ (стик 1 не задіяний); зарахування фінішу/таймауту НЕ реалізовано
— лише межі й зіткнення, та ПЕРЕМОГА (чекпоінт), про яку нижче.

Перемога в ручному режимі: долетівши до чекпоінта (той самий циліндр-«маяк») —
автоматично стартує НОВА випадкова мапа («наступний раунд»), а в боковій панелі
НАЗАВЖДИ (доти сесії) з'являється кнопка «Пересісти на Потужноліт» (easter egg —
ukrainian-aviation-meme, assets/potuznolit_source.glb, масштабовано до розміру
справжнього надлегкого літака — RealisticPlane.PLANE_SCALE). Керування ТЕЖ
перемикається: замість RealisticQuad (нахил напряму векторизує тягу) — окрема
аеродинамічна модель фіксованого крила RealisticPlane (керма в'ялі на малій
швидкості, зрив потоку при надто різкому тангажі) — «реалістичне керування
літаком». Клавіша Tab — те саме, ярлик.
"""

import dataclasses
import math
import os
import random
import sys

import bpy
import mathutils

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from realistic_flight import (RealisticQuad, QuadParams,       # noqa: E402
                              RealisticPlane, PlaneParams)
from game_env import (DEFAULT_CONFIG as CFG, STATUS_RUNNING,  # noqa: E402
                      STATUS_COLLISION, STATUS_DISQUALIFIED,
                      STATUS_FINISHED, STATUS_TIMEOUT, STATUS_MESSAGES)
from game_env import animate_drone                            # noqa: E402
from game_env import mesh_models                               # noqa: E402
from game_env import propellers                                # noqa: E402
from game_env.generator import (MapGenerator, MapData, wreck_yaw,   # noqa: E402
                                point_in_oriented_box, OBJECT_FOOTPRINTS,
                                TRUCK_FOOTPRINT)
from game_env.scene import build_environment                  # noqa: E402
from game_env.lidar2d import (binned_lidar_2d, lidar_obstacles_xyr,  # noqa: E402
                              bin_angles)
import sim_headless                                            # noqa: E402
import metrics_report                                          # noqa: E402

# APF-карта потенціалу (адмінська теплокарта в HUD) — ЕТАЛОННА математика,
# живе окремо в admin/ (не роздається учасникам хакатону разом із
# solution.py-заглушками, інакше Адмін-режим був би готовою підказкою до
# розв'язку). Якщо admin/ відсутня (типовий стан учасницької копії
# репо) — просто немає кнопки APF-карти, решта гри працює як завжди.
try:
    from admin.solution import (APFParams, potential_field_grid,  # noqa: E402
                                blended_display_field)
    _APF_MAP_AVAILABLE = True
except ImportError:
    _APF_MAP_AVAILABLE = False

SEED = 2026             # запасний детермінований сід (лише headless-самоперевірка)
KEY_HZ = 60.0

# Керування — ЛИШЕ латинські клавіші (англійська розкладка). event.unicode/event.type
# зіставляються з цим словником; кирилична розкладка НЕ мапиться на керування взагалі.
LATIN_CONTROL_KEYS = {
    "w": "W", "W": "W",
    "s": "S", "S": "S",
    "a": "A", "A": "A",
    "d": "D", "D": "D",
    "q": "Q", "Q": "Q",
    "e": "E", "E": "E",
    "r": "R", "R": "R",
    "c": "C", "C": "C",
}

# Кириличні символи на фізичному місці керуючих клавіш (ЙЦУКЕН — укр./рос. розкладки).
# Побачивши будь-який із них у event.unicode на PRESS — подію ЗАБОРОНЕНО, ігноруємо
# цілком (не транслюємо в keycode), незалежно від того, що покаже event.type для тієї
# самої фізичної клавіші: керування кирилицею вимкнено навмисно.
_CYRILLIC_CONTROL_CHARS = {
    "ц", "Ц", "і", "І", "ы", "Ы", "ф", "Ф", "в", "В", "й", "Й", "у", "У", "к", "К", "с", "С",
}

CONTROL_KEYS = {"W", "S", "A", "D", "Q", "E", "SPACE", "LEFT_SHIFT", "RIGHT_SHIFT", "R", "C"}

# ── Режим польоту: ручний або автономний ──────────────────────────────────────────
# «manual»    → оператор DRONE_OT_manual (RealisticQuad, клавіатура)
# «autonomous» → оператор DRONE_OT_autonomous (replay trajectory з headless)
_FLIGHT_MODE = "manual"          # типово — ручний (безпечно для учасника)

# ── Роль доступу (Адмін/Учасник), захищена паролем — дослівно з 2e2bc5d ───────────
try:  # пароль живе в admin/ — в учасницькій копії адмін-режим недоступний
    from admin.solution import ADMIN_PASSWORD  # noqa: E402
except ImportError:
    ADMIN_PASSWORD = None    # None ніколи не збігається з введеним паролем
_IS_ADMIN = False                # типово — Учасник (безпечно за замовчуванням)

# Розблоковано «Потужноліт» (easter egg-літак) — НАЗАВЖДИ на цю сесію Blender,
# щойно пілот хоч раз долетів до чекпоінта в ручному режимі (див. _on_checkpoint_reached).
_PLANE_UNLOCKED = False

# ── Пропорції моделей: автоцистерна (assets/checkpoint_truck_source.glb, 8×3.5×2.7 м —
# СТАНДАРТ у СІМ CFG) — БАЗА для решти моделей. Дрон (~0.42 м) і уламки техніки вже
# нормалізовано під реальний метраж під час офлайн-підготовки asset'ів. «Потужноліт»
# (potuznolit_source.glb) експортовано в тому ж крихітному масштабі, що й дрон (щоб reskin
# не міняв хітбокс) — тут масштабуємо його ОКРЕМО, лише візуально, до розміру
# справжнього надлегкого літака з відкритою кабіною (~4.4 м фюзеляж — між дроном і
# фурою, реалістично для цього класу апаратів).
PLANE_SCALE = 10.5             # 0.42 м (сирий glb) × 10.5 ≈ 4.4 м фюзеляжу
# Колізійний радіус «Потужнольоту» — БІЛЬШИЙ за дрон-хітбокс (0.35 м), але
# СВІДОМО набагато менший за реальну візуальну півширину крила (~1.85 м при
# PLANE_SCALE — буквальна сфера такого радіуса майже ніколи не пролізла б між
# деревами: ліс розсіяний із мін. проміжком між кронами лише ~0.7 м, тож повний
# «розмах крила»-хітбокс = миттєва аварія одразу після пересідання). Тут —
# ігровий компроміс візуал/колізія заради керованості, а не буквальна фізика.
PLANE_COLLISION_RADIUS = 0.7    # м


def is_admin() -> bool:
    """Чи активний режим Адміністратора (доступ до тестової мапи: seed/дерева/перегенерація)."""
    return _IS_ADMIN


def _random_seed() -> int:
    """Новий випадковий сід мапи (кожен запуск учасника — інша арена)."""
    return random.randint(0, 1_000_000)

# Єдине джерело стану сцени/польоту — і модальний оператор, і кнопки бокової панелі,
# і HUD-оверлей читають/пишуть сюди (панель і HUD — окремі виклики Blender, не мають
# доступу до приватних полів об'єкта-оператора).
_RUNTIME = {
    "cfg": CFG, "md": None, "terrain": None, "start": (0.0, 0.0, 3.0),
    "drone": None, "fpv": None, "chase": None, "cam_fpv": False,
    "quad": None, "status": STATUS_RUNNING, "running": False, "telemetry": None,
    "crash_text": None, "vehicle": "drone", "switch_grace": 0,
    "autonomous_gen": 0, "n_runs_history": None, "traj_seed": None,
    "show_apf": False, "apf_field": None,   # APF-карта потенціалу — лише Адмін, див. _draw_apf_map
}

# Скільки тіків після перемикання дрон⟷Потужноліт колізія ІГНОРУЄТЬСЯ: сам
# перемикач раптово змінює body_radius (0.35 м дрона → PLANE_COLLISION_RADIUS),
# а позиція лишається та сама — без цієї «пільги» точка, яка щойно була вільною
# для дрона, миттєво опинялась би «всередині» набагато більшого хітбокса
# літака, і гравець бачив би «БОРТ ВТРАЧЕНО» одразу після Tab, без жодної аварії.
SWITCH_GRACE_TICKS = 60   # ≈1.0 с при KEY_HZ=60 (запас, щоб «Потужноліт» встиг відлетіти від того, що було впритул під дроном)
_HUD_HANDLE = None

# Разовий easter egg: 10% шанс, що напис аварії замінюється на цей (замість
# STATUS_MESSAGES[STATUS_COLLISION] = «БОРТ ВТРАЧЕНО»). Ролл — один раз за подію
# аварії (у момент переходу в STATUS_COLLISION), не щокадру.
CRASH_EASTER_EGG_CHANCE = 0.10
CRASH_EASTER_EGG_TEXT = "Догана нахуй"

# Кольори HUD — узяті напряму з палітри пісочниці engine_test.py (0..255 → 0..1)
_HUD_PANEL_BG = (22 / 255, 25 / 255, 31 / 255, 210 / 255)
_HUD_DRONE_C = (250 / 255, 205 / 255, 70 / 255, 1.0)
_HUD_GREEN = (95 / 255, 215 / 255, 125 / 255, 1.0)
_HUD_WHITE = (240 / 255, 240 / 255, 240 / 255, 1.0)
_HUD_GREY = (150 / 255, 155 / 255, 165 / 255, 1.0)
_HUD_BAR_BG = (40 / 255, 44 / 255, 52 / 255, 1.0)

# Кольори гіроскопа (штучний горизонт у маленькому віконечку панелі) —
# авіаційна палітра: небо/земля приглушені, лінія горизонту — яскраво-зелена
# (класичний colimator-HUD).
_FPV_SKY = (0.30, 0.45, 0.68, 0.55)
_FPV_GROUND = (0.36, 0.27, 0.16, 0.55)
_FPV_HORIZON = (0.15, 0.95, 0.25, 0.95)

# Радар-коло 2D-лідара: сірий диск, центр яскравіший/непрозоріший, край темніший/
# прозоріший (радіальний градієнт); сектор із виявленою перешкодою тоне в цей же
# градієнт до червоного — тим сильніше, чим ближче перешкода.
_RADAR_RADIUS = 90
_RADAR_MARGIN = 22
_RADAR_CENTER_GRAY = (0.62, 0.62, 0.65, 0.55)
_RADAR_RIM_GRAY = (0.30, 0.30, 0.33, 0.12)
_RADAR_HIT = (0.95, 0.15, 0.12, 0.85)


def _pose_matrix_raw(pos, yaw, pitch, roll) -> mathutils.Matrix:
    """Поза → трансформа об'єкта: R = Rz(yaw)·Ry(pitch)·Rx(roll) (як у рушії)."""
    rot = (mathutils.Matrix.Rotation(yaw,   4, 'Z')
           @ mathutils.Matrix.Rotation(pitch, 4, 'Y')
           @ mathutils.Matrix.Rotation(roll,  4, 'X'))
    return mathutils.Matrix.Translation(mathutils.Vector(pos)) @ rot


def pose_matrix(q) -> mathutils.Matrix:
    """Те саме, з живого RealisticQuad (ручний режим)."""
    return _pose_matrix_raw(q.pos, q.yaw, q.pitch, q.roll)


def collision_and_bounds_status(pos, md, terrain, cfg, body_radius=None):
    """Дослівно логіка _update_status коміту 2e2bc5d (дротик — сама геометрія):
      • МЕЖІ — переліт стелі (середня висота крон, md.ceiling) АБО виліт за
        горизонтальні межі арени [-bounds, bounds] по x/y (старий рушій сам не пускав
        так далеко через leash-інтегратор; реалістичний рушій літає вільно, тож межу
        перевіряємо явно) → STATUS_DISQUALIFIED;
      • дотик рельєфу / стовбура дерева / колізійної тематичної перешкоди / кузова
        фури-чекпойнта → STATUS_COLLISION.
    body_radius — радіус хітбокса корпуса; типово cfg.drone_radius (дрон), але
    для «Потужнольоту» (набагато більший реальний фюзеляж) виклик передає
    PLANE_COLLISION_RADIUS — інакше велетенський літак безкарно проходив би
    крізь дерева/фуру, у які реально врізається крилом."""
    dr = cfg.drone_radius if body_radius is None else body_radius
    x, y, z = pos
    if z > md.ceiling or abs(x) > cfg.bounds or abs(y) > cfg.bounds:
        return STATUS_DISQUALIFIED

    if z - terrain.height_at(x, y) < dr:
        return STATUS_COLLISION

    for i, (tx, ty, z_base, r, h) in enumerate(md.trees):
        if i == md.wreck_index and md.wreck_kind in OBJECT_FOOTPRINTS:
            # Уламок техніки — витягнутий/прямокутний силует, окремий
            # орієнтований хітбокс (не кругла апроксимація WRECK_DIMS-радіуса
            # для всіх трьох) — див. game_env/generator.py OBJECT_FOOTPRINTS.
            if point_in_oriented_box(x, y, tx, ty, wreck_yaw(tx, ty),
                                     OBJECT_FOOTPRINTS[md.wreck_kind], margin=dr):
                if z_base - dr <= z <= z_base + h + dr:
                    return STATUS_COLLISION
            continue
        if math.hypot(x - tx, y - ty) < r + dr:
            if z_base - dr <= z <= z_base + h + dr:
                return STATUS_COLLISION

    for _kind, ox, oy, oz, r, collidable in md.obstacles:
        if collidable and math.dist((x, y, z), (ox, oy, oz)) < r + dr:
            return STATUS_COLLISION

    for cx, cy, _cz in md.checkpoints:
        gz = terrain.height_at(cx, cy)
        # Фура — реальні виміряні розміри (TRUCK_FOOTPRINT), не старі 8×3.5 м
        # без жодного зв'язку з мешем; yaw=0.0 — фура завжди без курсу.
        if (point_in_oriented_box(x, y, cx, cy, 0.0, TRUCK_FOOTPRINT, margin=dr)
                and gz - dr <= z <= gz + cfg.truck_height):
            return STATUS_COLLISION

    return STATUS_RUNNING


def _update_chase_camera_pose(pos, yaw):
    """Оновити положення 3POV (Chase) камери, щоб вона слідувала за дроном ззаду
    під кутом, але не успадковувала його крен і тангаж (як у комп'ютерних іграх)."""
    chase = _RUNTIME.get("chase")
    if chase is None:
        return
    x, y, z = pos
    dist = 3.5
    height = 1.2
    
    # Позиція камери ззаду по курсу дрона (yaw)
    cx = x - dist * math.cos(yaw)
    cy = y - dist * math.sin(yaw)
    cz = z + height
    
    chase.location = (cx, cy, cz)
    # Нахил вниз та поворот за yaw дрона
    tilt = math.atan2(height, dist)
    chase.rotation_euler = (math.pi / 2 - tilt, 0.0, yaw - math.pi / 2)


def _in_checkpoint_zone(pos):
    """Чи перебуває позиція всередині того самого напівпрозорого циліндра
    зарахування, що видно в сцені навколо чекпоінта — умова ПЕРЕМОГИ в
    ручному режимі (див. _on_checkpoint_reached)."""
    md = _RUNTIME.get("md")
    cfg = _RUNTIME.get("cfg")
    if not md or not md.checkpoints:
        return False
    cx, cy, cz = md.checkpoints[0]
    x, y, z = pos
    return (math.hypot(x - cx, y - cy) < cfg.cp_cyl_radius
            and cz - cfg.cp_cyl_height / 2.0 <= z <= cz + cfg.cp_cyl_height / 2.0)


def _make_quad(pos, yaw=0.0):
    q = RealisticQuad(QuadParams())
    q.reset(pos, yaw)
    return q


def _make_plane(pos, yaw=0.0):
    p = RealisticPlane(PlaneParams())
    p.reset(pos, yaw)
    return p


def switch_to_plane():
    """Пересісти на «Потужноліт» (assets/potuznolit_source.glb, easter egg —
    ukrainian-aviation-meme): підміняємо mesh-дані ТОГО САМОГО об'єкта 'Drone'
    (FPV/Chase-камери прикріплені саме до нього — перепризначати не треба) і
    масштабуємо його до РЕАЛЬНОГО розміру надлегкого літака (PLANE_SCALE —
    сирий glb експортовано в масштабі дрона, щоб reskin не міняв старий
    хітбокс; тепер масштаб суто візуальний, колізія рахується окремо через
    PLANE_COLLISION_RADIUS). Керування ТЕЖ перемикається — RealisticQuad
    поступається місцем RealisticPlane (реалістична аеродинаміка фіксованого
    крила) у _RUNTIME["quad"]; позиція/курс переносяться для безшовного
    продовження польоту. Обидва меші нормалізовано носом на +X, тож підміна
    не збиває орієнтацію."""
    drone = _RUNTIME.get("drone")
    if drone is None or _RUNTIME.get("vehicle") == "plane":
        return
    temp = mesh_models.spawn("potuznolit_source.glb", "_PlaneMeshSource")
    drone.data = temp.data
    bpy.data.objects.remove(temp, do_unlink=True)
    drone.scale = (PLANE_SCALE, PLANE_SCALE, PLANE_SCALE)
    # Гвинт-«хрестовина» додається ОКРЕМИМ об'єктом поверх носа (drone.data не
    # чіпаємо) — щоб потім крутити лише його, а не весь літак.
    propellers.sync_propellers_for_vehicle(drone, "plane")
    _RUNTIME["vehicle"] = "plane"
    _RUNTIME["switch_grace"] = SWITCH_GRACE_TICKS
    old = _RUNTIME.get("quad")
    pos = tuple(old.pos) if old is not None else _RUNTIME["start"]
    yaw = old.yaw if old is not None else 0.0
    _RUNTIME["quad"] = _make_plane(pos, yaw)


def switch_to_drone():
    """Повернутися з «Потужноліт» на дрон (той самий об'єкт 'Drone', масштаб і
    керування — назад до RealisticQuad) — після розблокування можна вільно
    перемикатись в обидва боки."""
    drone = _RUNTIME.get("drone")
    if drone is None or _RUNTIME.get("vehicle") == "drone":
        return
    temp = mesh_models.spawn("drone_source.glb", "_DroneMeshSource")
    drone.data = temp.data
    bpy.data.objects.remove(temp, do_unlink=True)
    drone.scale = (1.0, 1.0, 1.0)
    propellers.sync_propellers_for_vehicle(drone, "drone")
    _RUNTIME["vehicle"] = "drone"
    _RUNTIME["switch_grace"] = SWITCH_GRACE_TICKS
    old = _RUNTIME.get("quad")
    pos = tuple(old.pos) if old is not None else _RUNTIME["start"]
    yaw = old.yaw if old is not None else 0.0
    _RUNTIME["quad"] = _make_quad(pos, yaw)


def toggle_vehicle():
    """Перемкнути дрон⟷Потужноліт (лише якщо вже розблоковано хоч раз
    долетівши до чекпоінта — _PLANE_UNLOCKED)."""
    if not _PLANE_UNLOCKED:
        return
    if _RUNTIME.get("vehicle") == "drone":
        switch_to_plane()
    else:
        switch_to_drone()


def _next_round():
    """Викликається через bpy.app.timers ПІСЛЯ того, як поточний модальний
    оператор встиг самозавершитись (див. _on_checkpoint_reached) — нова
    випадкова мапа + автостарт ручного польоту («наступний раунд»)."""
    build_scene(seed=_random_seed())
    reset_flight()
    bpy.ops.wm.drone_manual('INVOKE_DEFAULT')
    return None


def _on_checkpoint_reached():
    """Перемога в ручному режимі: чекпоінт досягнуто → «Потужноліт» стає
    доступним НАЗАВЖДИ (ця сесія Blender) у боковій панелі, і стартує нова
    випадкова мапа. Регенерація сцени відкладена в bpy.app.timers — інакше
    clear_scene() всередині build_scene() прибрав би об'єкти під ногами
    модального оператора, що ЗАРАЗ виконує свій _tick()."""
    global _PLANE_UNLOCKED
    _PLANE_UNLOCKED = True
    _RUNTIME["running"] = False
    bpy.app.timers.register(_next_round, first_interval=0.05)


def reset_flight():
    """Скинути дрон у старт мапи й зняти термінальний статус (клавіша R і панель).
    Повернення на дрон (якщо був у «Потужнольоті») — ПЕРЕД скидом позиції: інакше
    switch_to_drone() перенесла б нову RealisticQuad у ще-старе (польотне)
    положення «Потужнольоту», а не в старт мапи."""
    if _RUNTIME.get("vehicle") == "plane":
        switch_to_drone()

    q = _RUNTIME.get("quad")
    if q is not None:
        q.reset(_RUNTIME["start"])

    _update_chase_camera_pose(_RUNTIME["start"], 0.0)
    _RUNTIME["traj_frame"] = 0
    _RUNTIME["status"] = STATUS_RUNNING
    _RUNTIME["crash_text"] = None

    _ensure_camera_is_active()

    drone = _RUNTIME.get("drone")
    if drone is not None:
        traj = _RUNTIME.get("trajectory")
        if traj and len(traj) > 0:
            frame = traj[0]
            drone.matrix_world = _pose_matrix_raw(
                (frame["x"], frame["y"], frame["z"]), frame["yaw"], frame["pitch"], frame["roll"])
            _update_chase_camera_pose((frame["x"], frame["y"], frame["z"]), frame["yaw"])
        else:
            drone.matrix_world = mathutils.Matrix.Translation(mathutils.Vector(_RUNTIME["start"]))
            _update_chase_camera_pose(_RUNTIME["start"], 0.0)


def _ensure_camera_is_active():
    """Переконатися, що активна камера сцени встановлена на поточну камеру FPV/Chase,
    і 3D-в'юпорт примусово заблокований у режимі CAMERA."""
    cam = _RUNTIME.get("fpv") if _RUNTIME.get("cam_fpv") else _RUNTIME.get("chase")
    if cam is not None:
        bpy.context.scene.camera = cam
        area = _view3d_area()
        if area is not None:
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            # Прибираємо блендерівську розмітку, сітку, значки камер тощо
            area.spaces.active.overlay.show_overlays = False


def _view3d_area():
    """Перша область VIEW_3D активного вікна, або None (вікно/екран ще не готові)."""
    window = getattr(bpy.context, "window", None)
    if window is None or window.screen is None:
        return None
    return next((a for a in window.screen.areas if a.type == 'VIEW_3D'), None)


def toggle_camera():
    """Перемкнути активну камеру (Chase↔FPV) І заблокувати сам в'юпорт у вигляд
    камери (дослівно _set_camera коміту 2e2bc5d) — самого лише scene.camera
    НЕДОСТАТНЬО: якщо в'юпорт не в режимі 'CAMERA', користувач і далі бачить вільну
    орбіту, а не картинку з дрона."""
    _RUNTIME["cam_fpv"] = not _RUNTIME["cam_fpv"]
    cam = _RUNTIME["fpv"] if _RUNTIME["cam_fpv"] else _RUNTIME["chase"]
    bpy.context.scene.camera = cam
    area = _view3d_area()
    if area is not None:
        area.spaces.active.region_3d.view_perspective = 'CAMERA'


def _enforce_camera_lock():
    """Камера має бути ЛИШЕ від першої (FPV) або третьої (Chase) особи — жодної
    вільної орбіти мишею. Blender сам виходить із виду 'CAMERA' щойно користувач
    покрутить в'юпорт мишею, тож щотіку примусово повертаємо його назад — за 1/60с
    вільна орбіта непомітно «відскакує» до зафіксованої камери."""
    area = _view3d_area()
    if area is None:
        return
    r3d = area.spaces.active.region_3d
    if r3d.view_perspective != 'CAMERA':
        r3d.view_perspective = 'CAMERA'


def _enter_kiosk_view():
    """Лише в'юпорт: сховати шапку, тулбар (T) і бокову панель (N) — розгорнути
    3D-в'юпорт на все вікно (нативний ігровий/презентаційний режим Blender). Плюс
    одразу кольоровий рендеринг (Material Preview) — типовий вигляд Blender-в'юпорта
    (Solid, без кольору матеріалів) інакше показував би сірі об'єкти без текстур."""
    window = bpy.context.window
    area = _view3d_area()
    if window is None or area is None:
        return False
    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.screen.screen_full_area(use_hide_panels=True)
    # screen_full_area підміняє window.screen на тимчасовий fullscreen-екран — стара
    # посилання на area вже НЕвалідна, тож область треба знайти заново.
    area2 = _view3d_area()
    if area2 is not None:
        area2.spaces.active.shading.type = 'RENDERED'
        area2.spaces.active.overlay.show_overlays = False
    return True


def _lock_scene_selection():
    """Заборонити вибір ВСІХ об'єктів у в'юпорті (крім дрона і камер —
    їх теж блокуємо для чистоти кіоск-режиму). Так клік мишею в 3D-в'юпорті
    не виділяє дерева/рельєф/фури і не збиває вид камери."""
    for obj in bpy.data.objects:
        obj.hide_select = True


def _build_scene_visuals(md, cfg):
    """Спільна частина побудови сцени з ГОТОВОЇ MapData: порожня сцена → світло/небо →
    середовище 2e2bc5d → дрон + дві камери → скидання _RUNTIME. Використовується і
    ручним режимом (md щойно згенеровано), і автономним (md — З ТАБЛИЦІ, щоб сцена
    ТОЧНО збігалася з тим, що порахував headless-прогін, а не перегенеровувалась
    незалежно)."""
    terrain = md.terrain(cfg)

    animate_drone.clear_scene()
    animate_drone.setup_world_and_light()
    build_environment(md, cfg, terrain)       # рельєф + ліс + фури + перешкоди

    drone = animate_drone.make_drone()
    fpv, chase = animate_drone.add_cameras(drone)
    bpy.context.scene.camera = chase
    drone.matrix_world = mathutils.Matrix.Translation(mathutils.Vector(md.start))
    propellers.sync_propellers_for_vehicle(drone, "drone")   # build_scene завжди стартує дроном

    # Заборонити вибір усіх об'єктів — клік мишею не збиватиме вид камери
    _lock_scene_selection()

    # Вмикаємо швидкий рендер Eevee для гарного 3D-освітлення з тінями
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    lidar_obs = lidar_obstacles_xyr(md)
    lidar_angles = bin_angles(cfg.lidar_n_az)

    _RUNTIME.update({
        "cfg": cfg, "md": md, "terrain": terrain, "start": tuple(md.start),
        "drone": drone, "fpv": fpv, "chase": chase, "cam_fpv": False,
        "quad": None, "status": STATUS_RUNNING, "running": False, "telemetry": None,
        "crash_text": None, "trajectory": None, "traj_frame": 0, "traj_hz": None,
        "traj_seed": None,
        "lidar_obs": lidar_obs, "lidar_angles": lidar_angles,
        "vehicle": "drone", "switch_grace": 0,
        "apf_field": None,   # нова мапа — стара APF-сітка (якщо була) більше не годиться
    })
    _update_chase_camera_pose(md.start, 0.0)
    return drone


def build_scene(seed=SEED, n_trees=None):
    """РУЧНИЙ режим: згенерувати нову мапу за seed і побудувати сцену.
    n_trees=None → типова кількість з конфіга (60); інакше override (SimConfig
    незмінний, тож клонуємо через dataclasses.replace)."""
    cfg = CFG if n_trees is None else dataclasses.replace(CFG, n_trees=int(n_trees))
    md = MapGenerator(cfg).build(seed=int(seed))
    return _build_scene_visuals(md, cfg)


def build_scene_from_mapdata(md, cfg):
    """АВТОНОМНИЙ режим: побудувати сцену з УЖЕ ГОТОВОЇ MapData (з таблиці
    sim_headless.simulate() — саме та мапа, яку рахував headless-прогін, а не нова
    незалежна генерація)."""
    return _build_scene_visuals(md, cfg)


# ── Модальний оператор польоту ────────────────────────────────────────────────────

class DRONE_OT_manual(bpy.types.Operator):
    """Модальний оператор: таймер жене фізику, клавіші задають команди рушія."""
    bl_idname = "wm.drone_manual"
    bl_label = "Дрон: ручний політ (2e2bc5d)"

    @classmethod
    def poll(cls, context):
        return not _RUNTIME.get("running", False)

    def execute(self, context):
        # Фабрика фізики за поточним апаратом (панель/Tab могли перемкнути
        # vehicle ще ДО старту, наприклад одразу після минулого чекпоінта).
        if _RUNTIME.get("vehicle") == "plane":
            _RUNTIME["quad"] = _make_plane(_RUNTIME["start"])
        else:
            _RUNTIME["quad"] = _make_quad(_RUNTIME["start"])
        self.keys = set()
        self._held_by_type = {}   # сирий event.type з ПРЕСУ -> логічний символ (див. modal())
        _RUNTIME["status"] = STATUS_RUNNING
        _RUNTIME["running"] = True
        
        # Примусово активуємо та блокуємо камеру при старті
        _ensure_camera_is_active()
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / KEY_HZ, window=context.window)
        wm.modal_handler_add(self)
        print("Дрон: ручний політ у середовищі 2e2bc5d — W/S тангаж, A/D крен, "
              "Q/E курс, Space/Shift газ, C камера, R скид")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Якщо політ зупинено ззовні (наприклад, змінено режим керування)
        if not _RUNTIME.get("running", False):
            return self._finish(context)

        # Вікно втратило фокус ОС (Alt+Tab, системна панель, перемикання
        # розкладки клавіатури тощо) — БУДЬ-ЯКА затиснута клавіша може більше
        # ніколи не отримати свій RELEASE (система інколи «з'їдає» цю подію,
        # не довівши до Blender), і без цього скидання вона лишалась би
        # «застряглою» в self.keys навіки (дрон летить в один бік без зупинки).
        # Найчастіший практичний випадок — стандартне macOS-скорочення
        # перемикання розкладки Control+Space: SPACE водночас керує газом.
        if event.type == 'WINDOW_DEACTIVATE':
            self.keys.clear()
            self._held_by_type.clear()
            return {'RUNNING_MODAL'}

        # Перекладаємо клавішу на латинський аналог. Кирилична розкладка для
        # керування ЗАБОРОНЕНА навмисно: якщо PRESS показує кириличний символ
        # на місці керуючої клавіші (event.unicode у _CYRILLIC_CONTROL_CHARS) —
        # подія повністю ІГНОРУЄТЬСЯ (key_char лишається None), незалежно від
        # того, що покаже event.type для тієї самої фізичної клавіші.
        #
        # RELEASE-подія в Blender НЕ несе event.unicode (лише PRESS) — тож
        # реліз розпізнається через LATIN_CONTROL_KEYS[event.type]. Про запас —
        # якщо на деяких платформах event.type для літер «пливе» між
        # розкладками — на ПРЕСІ запам'ятовуємо, який логічний символ
        # відповідав САМЕ ЦЬОМУ сирому event.type (self._held_by_type), і на
        # релізі спершу дивимось туди: реліз завжди «пара» до свого пресу,
        # незалежно від event.type (захищає й від втраченого RELEASE після
        # WINDOW_DEACTIVATE, і від будь-якого іншого платформного дрейфу).
        key_char = None
        if event.value == 'PRESS':
            if event.unicode in _CYRILLIC_CONTROL_CHARS:
                pass   # кирилиця — ігноруємо подію повністю, керування заборонено
            elif event.unicode and event.unicode in LATIN_CONTROL_KEYS:
                key_char = LATIN_CONTROL_KEYS[event.unicode]
            elif event.type in LATIN_CONTROL_KEYS:
                key_char = LATIN_CONTROL_KEYS[event.type]
            elif event.type in {"SPACE", "LEFT_SHIFT", "RIGHT_SHIFT"}:
                key_char = event.type
            if key_char is not None:
                self._held_by_type[event.type] = key_char
        elif event.value == 'RELEASE':
            key_char = self._held_by_type.pop(event.type, None)
            if key_char is None:
                if event.type in LATIN_CONTROL_KEYS:
                    key_char = LATIN_CONTROL_KEYS[event.type]
                elif event.type in {"SPACE", "LEFT_SHIFT", "RIGHT_SHIFT"}:
                    key_char = event.type

        if key_char == 'C' and event.value == 'PRESS':
            toggle_camera()
            return {'RUNNING_MODAL'}

        if key_char == 'R' and event.value == 'PRESS':
            reset_flight()
            return {'RUNNING_MODAL'}

        if event.type == 'TAB' and event.value == 'PRESS':
            toggle_vehicle()
            return {'RUNNING_MODAL'}

        if key_char is not None:
            if event.value == 'PRESS':
                self.keys.add(key_char)
            elif event.value == 'RELEASE':
                self.keys.discard(key_char)
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            self._tick()
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def _tick(self):
        # НЕ кешуємо self.drone НІ self.quad — читаємо обидва з _RUNTIME щотіку:
        # Tab/кнопка «Дрон ⟷ Потужноліт» підміняють ОБИДВА (switch_to_plane/
        # switch_to_drone) просто зараз, під час активного польоту, тож
        # закешоване self.quad миттєво «протухло» б і фізика лишилась старою.
        drone = _RUNTIME.get("drone")
        q = _RUNTIME.get("quad")
        if drone is None or q is None:
            return
        vehicle = _RUNTIME.get("vehicle", "drone")
        if _RUNTIME["status"] == STATUS_RUNNING:
            k = self.keys
            pitch = (1.0 if "W" in k else 0.0) - (1.0 if "S" in k else 0.0)
            roll = (1.0 if "D" in k else 0.0) - (1.0 if "A" in k else 0.0)
            yaw = (1.0 if "Q" in k else 0.0) - (1.0 if "E" in k else 0.0)
            throttle = (1.0 if "SPACE" in k else 0.0) \
                - (1.0 if ("LEFT_SHIFT" in k or "RIGHT_SHIFT" in k) else 0.0)
            q.step(throttle, pitch, roll, yaw, 1.0 / KEY_HZ)
            drone.matrix_world = pose_matrix(q)
            propellers.spin(vehicle, q.throttle, 1.0 / KEY_HZ)
            _update_chase_camera_pose(tuple(q.pos), q.yaw)
            body_radius = PLANE_COLLISION_RADIUS if vehicle == "plane" else _RUNTIME["cfg"].drone_radius
            new_status = collision_and_bounds_status(
                tuple(q.pos), _RUNTIME["md"], _RUNTIME["terrain"], _RUNTIME["cfg"], body_radius)
            grace = _RUNTIME.get("switch_grace", 0)
            if grace > 0:
                # Пільга одразу після дрон⟷Потужноліт: body_radius щойно стрибнув
                # (0.35 м → PLANE_COLLISION_RADIUS чи навпаки) на ТІЙ САМІЙ позиції —
                # COLLISION у цьому вікні не рахуємо (не справжня аварія, а артефакт
                # миттєвої зміни хітбокса); МЕЖІ (DISQUALIFIED) не залежать від
                # body_radius, тож їх пільга не чіпає.
                _RUNTIME["switch_grace"] = grace - 1
                if new_status == STATUS_COLLISION:
                    new_status = STATUS_RUNNING
            if new_status == STATUS_COLLISION and _RUNTIME["status"] != STATUS_COLLISION:
                # Свіжа аварія (перехід у COLLISION) — один ролл easter egg на подію,
                # не щокадру (інакше напис миготів би між варіантами, поки заморожено).
                _RUNTIME["crash_text"] = (CRASH_EASTER_EGG_TEXT
                                          if random.random() < CRASH_EASTER_EGG_CHANCE
                                          else STATUS_MESSAGES[STATUS_COLLISION])
            _RUNTIME["status"] = new_status
            if _in_checkpoint_zone(tuple(q.pos)):
                _on_checkpoint_reached()
                return
        # Телеметрія оновлюється завжди (і в заморожені кадри — HUD показує останні
        # відомі значення, а не зникає в момент аварії/дискваліфікації). Лідар — лише
        # для радар-HUD (ручний політ САМ по собі ним не керується, керує пілот).
        ox, oy, orr = _RUNTIME["lidar_obs"]
        cfg = _RUNTIME["cfg"]
        lidar = binned_lidar_2d(ox, oy, orr, q.pos[0], q.pos[1], cfg.lidar_n_az, cfg.lidar_range)
        terrain = _RUNTIME["terrain"]
        _RUNTIME["telemetry"] = {
            "vehicle": vehicle,
            "throttle": q.throttle, "hover_throttle": getattr(q, "hover_throttle", None),
            "x": q.pos[0], "y": q.pos[1],
            "z": q.pos[2], "agl": q.pos[2] - terrain.height_at(q.pos[0], q.pos[1]),
            "vz": q.vel[2], "speed": q.speed,
            "pitch": q.pitch, "roll": q.roll, "yaw": q.yaw,
            "stall_speed": getattr(q, "stall_speed", None),
            "lidar": [float(v) for v in lidar],
        }
        _enforce_camera_lock()

    def _finish(self, context):
        wm = context.window_manager
        if getattr(self, "_timer", None):
            wm.event_timer_remove(self._timer)
        _RUNTIME["running"] = False
        print("Дрон: політ завершено")
        return {'CANCELLED'}


class DRONE_OT_autonomous(bpy.types.Operator):
    """Автономний РЕЖИМ ВІДТВОРЕННЯ. ЖОДНОЇ фізики/AI тут — усе порахував headless
    sim_headless.simulate() ЗАЗДАЛЕГІДЬ; тут лише читаємо таблицю кадрів і щотіку
    виставляємо позу дрона (+ телеметрію для HUD). Нічого не кешує в self — і drone,
    і trajectory читаються з _RUNTIME щотіку, тож ніщо не «псується», якщо між
    прогонами сцену перебудували."""
    bl_idname = "wm.drone_autonomous"
    bl_label = "Дрон: автономний політ (replay)"

    @classmethod
    def poll(cls, context):
        return _RUNTIME.get("trajectory") is not None and not _RUNTIME.get("running", False)

    def execute(self, context):
        _RUNTIME["traj_frame"] = 0
        _RUNTIME["status"] = STATUS_RUNNING
        _RUNTIME["crash_text"] = None
        _RUNTIME["running"] = True

        # Токен покоління: кожен запуск інкрементує лічильник і запам'ятовує
        # СВОЄ значення на self. Якщо цей самий метод execute() викликають ще
        # раз, поки СТАРИЙ модальний інстанс ще активний (усі три кнопки мапи
        # роблять саме так — synchronously ставлять running=False й одразу ж
        # INVOKE_DEFAULT новий інстанс, який знову ставить running=True ще ДО
        # того, як старий встиг побачити False у своєму modal()) — старий
        # інстанс більше НІКОЛИ не побачить running=False і продовжив би
        # тікати паралельно з новим, подвоюючи (потроюючи...) швидкість
        # відтворення траєкторії з кожною новою мапою. Порівняння покоління
        # нижче ловить це незалежно від прапорця running.
        _RUNTIME["autonomous_gen"] = _RUNTIME.get("autonomous_gen", 0) + 1
        self._gen = _RUNTIME["autonomous_gen"]

        # Примусово активуємо та блокуємо камеру при старті
        _ensure_camera_is_active()

        wm = context.window_manager
        hz = _RUNTIME.get("traj_hz") or KEY_HZ
        self._timer = wm.event_timer_add(1.0 / hz, window=context.window)
        wm.modal_handler_add(self)
        print("Дрон: автономне відтворення — C камера")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Застарілий інстанс (див. execute()) — новіший запуск уже перехопив
        # відтворення. Прибираємо ЛИШЕ свій власний таймер і виходимо — НЕ
        # через self._finish()/_RUNTIME["running"]=False, бо це чуже, актуальне
        # для НОВІШОГО інстансу, значення: якби застарілий інстанс скинув його
        # тут, він забрав би керування щойно запущеним відтворенням.
        if self._gen != _RUNTIME.get("autonomous_gen"):
            wm = context.window_manager
            if getattr(self, "_timer", None):
                wm.event_timer_remove(self._timer)
            return {'CANCELLED'}

        # Якщо політ зупинено ззовні (наприклад, змінено режим керування)
        if not _RUNTIME.get("running", False):
            return self._finish(context)

        # Перекладаємо клавішу на латинський аналог — кирилична розкладка
        # заборонена (див. DRONE_OT_manual.modal для пояснення).
        key_char = None
        if event.value == 'PRESS' and event.unicode in _CYRILLIC_CONTROL_CHARS:
            pass
        elif event.value == 'PRESS' and event.unicode and event.unicode in LATIN_CONTROL_KEYS:
            key_char = LATIN_CONTROL_KEYS[event.unicode]
        elif event.type in LATIN_CONTROL_KEYS:
            key_char = LATIN_CONTROL_KEYS[event.type]

        if key_char == 'C' and event.value == 'PRESS':
            toggle_camera()
            return {'RUNNING_MODAL'}
        if key_char == 'R' and event.value == 'PRESS':
            reset_flight()
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            self._tick()
            return {'PASS_THROUGH'}
        return {'PASS_THROUGH'}

    def _tick(self):
        traj = _RUNTIME.get("trajectory")
        drone = _RUNTIME.get("drone")
        if not traj or drone is None:
            return
        i = _RUNTIME["traj_frame"]
        if i >= len(traj):
            return                            # останній кадр — дрон лишається на місці
        frame = traj[i]
        drone.matrix_world = _pose_matrix_raw(
            (frame["x"], frame["y"], frame["z"]), frame["yaw"], frame["pitch"], frame["roll"])
        # Автономний replay завжди дрон (кінематика без важеля газу) — крутимо
        # гвинти на фіксованому «крейсерському» газу, суто для візуалу.
        propellers.spin("drone", 0.8, 1.0 / (_RUNTIME.get("traj_hz") or KEY_HZ))
        _update_chase_camera_pose((frame["x"], frame["y"], frame["z"]), frame["yaw"])

        # Аптечка: несемо (carrying) → висить під дроном; НЕ несемо → на своєму місці
        # (це й фіксить перезапуск: кадри стартують з carrying=False → аптечка вдома).
        _med = bpy.data.objects.get("Medkit")
        _cross = bpy.data.objects.get("MedkitCrossH")
        if frame.get("carrying"):
            if _med is not None:
                _med.location = (frame["x"], frame["y"], frame["z"] - 0.35)
            if _cross is not None:
                _cross.location = (frame["x"], frame["y"], frame["z"] - 0.23)
        else:
            _home = _RUNTIME.get("medkit_home")
            _chome = _RUNTIME.get("medkit_cross_home")
            if _med is not None and _home is not None:
                _med.location = _home
            if _cross is not None and _chome is not None:
                _cross.location = _chome

        new_status = frame["status"]
        if new_status == STATUS_COLLISION and _RUNTIME["status"] != STATUS_COLLISION:
            _RUNTIME["crash_text"] = (CRASH_EASTER_EGG_TEXT
                                      if random.random() < CRASH_EASTER_EGG_CHANCE
                                      else STATUS_MESSAGES[STATUS_COLLISION])
        _RUNTIME["status"] = new_status

        terrain = _RUNTIME.get("terrain")
        agl = frame["z"] - terrain.height_at(frame["x"], frame["y"]) if terrain is not None else frame["z"]
        _RUNTIME["telemetry"] = {
            "vehicle": "drone",             # автономний replay — завжди дрон, «Потужноліт» лише ручний
            "throttle": None, "hover_throttle": None,   # немає важеля газу в кінематиці
            "x": frame["x"], "y": frame["y"],
            "z": frame["z"], "agl": agl, "vz": 0.0, "speed": frame["speed"],
            "pitch": frame["pitch"], "roll": frame["roll"], "yaw": frame["yaw"],
            "stall_speed": None,
            "lidar": frame["lidar"],
        }
        _RUNTIME["traj_frame"] = i + 1

    def _finish(self, context):
        wm = context.window_manager
        if getattr(self, "_timer", None):
            wm.event_timer_remove(self._timer)
        _RUNTIME["running"] = False
        print("Дрон: автономний політ завершено")
        return {'CANCELLED'}


# ── Кнопки бокової панелі ────────────────────────────────────────────────────────

class DRONE_OT_reset(bpy.types.Operator):
    bl_idname = "wm.drone_reset"
    bl_label = "Скид"

    @classmethod
    def poll(cls, context):
        return _RUNTIME.get("drone") is not None

    def execute(self, context):
        reset_flight()
        return {'FINISHED'}


class DRONE_OT_toggle_camera(bpy.types.Operator):
    bl_idname = "wm.drone_toggle_camera"
    bl_label = "Камера"

    @classmethod
    def poll(cls, context):
        return _RUNTIME.get("drone") is not None

    def execute(self, context):
        toggle_camera()
        return {'FINISHED'}


class DRONE_OT_switch_vehicle(bpy.types.Operator):
    """Перемкнути дрон⟷«Потужноліт» (easter egg-літак). Кнопка з'являється в
    панелі НАЗАВЖДИ (ця сесія), щойно пілот хоч раз долетів до чекпоінта."""
    bl_idname = "wm.drone_switch_vehicle"
    bl_label = "Дрон ⟷ Потужноліт"
    bl_description = "Перемкнути керування між дроном і «Потужноліт»"

    @classmethod
    def poll(cls, context):
        return _PLANE_UNLOCKED and _RUNTIME.get("drone") is not None

    def execute(self, context):
        toggle_vehicle()
        label = "дрон" if _RUNTIME.get("vehicle") == "drone" else "Потужноліт"
        self.report({'INFO'}, f"Керування: {label}")
        return {'FINISHED'}


def _get_history_count() -> int:
    """К-ть збережених прогонів (out/autonomous_runs.json), кешована в _RUNTIME —
    підвантажується з диска ЛИШЕ один раз (перше звернення, з панелі або з
    _load_autonomous_map), а не щоразу під час перемальовування панелі."""
    if _RUNTIME.get("n_runs_history") is None:
        _RUNTIME["n_runs_history"] = len(sim_headless.load_run_history())
    return _RUNTIME["n_runs_history"]


def _load_autonomous_map(seed):
    """Спільна логіка обох адмін-кнопок мапи: порахувати ПОВНИЙ автономний прогін
    headless (sim_headless.simulate — лідар+APF+A*+кінематичний автопілот, УСЕ
    заздалегідь), відновити MapData САМЕ З РЕЗУЛЬТАТУ (не перегенеровувати незалежно
    — сцена має точно відповідати тому, що порахувала таблиця), і одразу запустити
    автономне відтворення. Дрон у Blender після цього НІЧОГО не рахує — лише читає
    _RUNTIME["trajectory"]."""
    result = sim_headless.simulate(seed=seed)
    md = MapData.from_dict(result["map"])
    cfg = dataclasses.replace(CFG, n_trees=result["meta"]["n_trees"])
    build_scene_from_mapdata(md, cfg)
    # запам'ятати «домівку» аптечки (де scene.py її заспавнив), щоб при перезапуску
    # вона поверталась на місце, а не лишалась висіти під дроном
    _med = bpy.data.objects.get("Medkit")
    _cross = bpy.data.objects.get("MedkitCrossH")
    _RUNTIME["medkit_home"] = tuple(_med.location) if _med else None
    _RUNTIME["medkit_cross_home"] = tuple(_cross.location) if _cross else None
    _RUNTIME["trajectory"] = result["frames"]
    _RUNTIME["traj_hz"] = result["meta"]["sim_hz"]
    _RUNTIME["traj_seed"] = seed   # яку мапу вже порахували — щоб «Запустити
                                   # автономний» не рахував і не логував її ЩЕ РАЗ
    # Персистентні метрики якості (out/autonomous_runs.json) — з КОЖНОГО
    # автономного прогону, незалежно від сесії Blender; кнопка «Метрики
    # алгоритму» показує їх усі разом із мапою/графіками ЦЬОГО прогону.
    # _get_history_count() ОБОВ'ЯЗКОВО ДО append_run_history() — інакше, якщо
    # кеш ще не підвантажений цієї сесії (None), він сам зчитав би файл
    # ПІСЛЯ дописування нового запису й порахував би цей прогін двічі (+1 тут
    # поверх уже присутнього в щойно прочитаному файлі).
    _get_history_count()
    run_metrics = sim_headless.compute_run_metrics(result)
    sim_headless.append_run_history(run_metrics)
    _RUNTIME["n_runs_history"] += 1
    print("Дрон: автономна мапа seed=%d готова — статус=%s, кадрів=%d" % (
        seed, result["meta"]["final_status"], result["meta"]["n_frames"]))


class DRONE_OT_load_test_map(bpy.types.Operator):
    """Завантажити ТЕСТОВУ (фіксовану, детерміновану) мапу — той самий seed щоразу,
    зручно для повторюваного тестування. Лише Адмін.
    ЛИШЕ рахує траєкторію й будує сцену — реплей НЕ стартує автоматично: користувач
    сам натискає «▶ Запустити автономний», коли буде готовий."""
    bl_idname = "wm.drone_load_test_map"
    bl_label = "Завантажити тестову мапу"

    @classmethod
    def poll(cls, context):
        return is_admin()

    def execute(self, context):
        _RUNTIME["running"] = False
        _load_autonomous_map(SEED)
        return {'FINISHED'}


class DRONE_OT_load_random_map(bpy.types.Operator):
    """Завантажити мапу з НОВИМ ВИПАДКОВИМ сідом — єдиний спосіб отримати іншу мапу
    (ручного вводу конкретного сіда більше немає, лише випадковий вибір).
    ЛИШЕ рахує траєкторію й будує сцену — реплей НЕ стартує автоматично: користувач
    сам натискає «▶ Запустити автономний», коли буде готовий."""
    bl_idname = "wm.drone_load_random_map"
    bl_label = "Випадкова мапа"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        _RUNTIME["running"] = False
        _load_autonomous_map(_random_seed())
        return {'FINISHED'}


class DRONE_OT_toggle_apf_map(bpy.types.Operator):
    """Показати/сховати теплокарту APF-потенціалу поточної мапи (лише Адмін) —
    діагностичний оверлей для налаштування APFParams, не частина гри учасника.
    Сама сітка рахується лінькво в _draw_apf_map (кешується в _RUNTIME["apf_field"],
    інвалідується на кожній новій мапі) — тут лише перемикається видимість."""
    bl_idname = "wm.drone_toggle_apf_map"
    bl_label = "APF карта"
    bl_description = "Показати/сховати карту потенціальних полів (APF) поточної мапи"

    @classmethod
    def poll(cls, context):
        return _APF_MAP_AVAILABLE and is_admin() and _RUNTIME.get("md") is not None

    def execute(self, context):
        _RUNTIME["show_apf"] = not _RUNTIME.get("show_apf", False)
        return {'FINISHED'}


class DRONE_OT_regen_manual_map(bpy.types.Operator):
    """Перегенерувати нову випадкову мапу в РУЧНОМУ режимі (без headless-симуляції).
    Дозволено всім."""
    bl_idname = "wm.drone_regen_manual_map"
    bl_label = "Нова мапа (ручний режим)"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        _RUNTIME["running"] = False
        new_seed = _random_seed()
        build_scene(seed=new_seed)
        reset_flight()
        print("Дрон: нова ручна мапа seed=%d" % new_seed)
        return {'FINISHED'}


class DRONE_OT_launch_autonomous(bpy.types.Operator):
    """Запустити автономний політ — все в одній кнопці. Автоматично запускає
    симуляцію для поточної відкритої мапи (бере її seed) — АЛЕ лише якщо для
    цього seed'а ще НЕМАЄ порахованої траєкторії (_RUNTIME["traj_seed"]).
    Якщо мапу вже завантажено кнопкою «Завантажити тестову/випадкову мапу» —
    просто стартує вже готове відтворення, без повторного прорахунку.
    Раніше рахувало ПОВТОРНО щоразу — той самий прогін дублювався в історії
    метрик (out/autonomous_runs.json), якщо користувач спершу завантажив
    мапу, а тоді натиснув цю кнопку.
    Дозволено всім."""
    bl_idname = "wm.drone_launch_autonomous"
    bl_label = "Запустити автономний політ"
    bl_description = ("Розрахувати симуляцію для поточної карти та запустити відтворення")

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        _RUNTIME["running"] = False
        md = _RUNTIME.get("md")
        seed = md.seed if md is not None else SEED
        if _RUNTIME.get("trajectory") is not None and _RUNTIME.get("traj_seed") == seed:
            self.report({'INFO'}, f"Траєкторія для seed={seed} вже порахована — запускаю без перерахунку")
        else:
            self.report({'INFO'}, f"Розрахунок автономної траєкторії для відкритої карти seed={seed}...")
            _load_autonomous_map(seed)
        bpy.ops.wm.drone_autonomous('INVOKE_DEFAULT')
        return {'FINISHED'}


class DRONE_OT_show_metrics(bpy.types.Operator):
    """Згенерувати й відкрити в браузері HTML-звіт якості автономного
    алгоритму: інтерактивна мапа + графіки ОСТАННЬОГО прогону цієї сесії
    (з _RUNTIME, без перерахунку) плюс таблиця й трендові графіки ПО ВСІХ
    прогонах, збережених у out/autonomous_runs.json (персистентно, включно
    з попередніми сесіями Blender). Немає bpy-залежностей у самій генерації
    звіту (metrics_report.py) — лише читання готових даних."""
    bl_idname = "wm.drone_show_metrics"
    bl_label = "Метрики алгоритму"
    bl_description = "Згенерувати й відкрити HTML-звіт якості автономного алгоритму"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        md = _RUNTIME.get("md")
        cfg = _RUNTIME.get("cfg")
        traj = _RUNTIME.get("trajectory")
        history = sim_headless.load_run_history()
        apf_field = None
        if md is not None and _APF_MAP_AVAILABLE:
            try:
                f = _compute_apf_field(md)
                apf_field = {
                    "xs": [float(v) for v in f["xs"]],
                    "ys": [float(v) for v in f["ys"]],
                    "mix": [[float(v) for v in row] for row in f["mix"]],
                    "target": [float(f["target"][0]), float(f["target"][1])],
                }
            except Exception as exc:   # noqa: BLE001
                print("APF-поле для звіту: не вдалося порахувати —", exc)
        html = metrics_report.build_report_html(
            map_dict=(md.to_dict() if md is not None else None),
            frames=traj,
            history=history,
            cp_radius=(cfg.cp_cyl_radius if cfg is not None else None),
            drone_radius=(cfg.drone_radius if cfg is not None else None),
            apf_field=apf_field,
        )
        out_path = sim_headless.OUT_DIR / "metrics_report.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        import webbrowser
        webbrowser.open(out_path.as_uri())
        self.report({'INFO'}, f"Звіт метрик відкрито: {out_path}")
        return {'FINISHED'}


class DRONE_OT_clear_metrics_history(bpy.types.Operator):
    """Стерти персистентну історію прогонів (out/autonomous_runs.json) — усі
    записи метрик з усіх сесій Blender, показані в кнопці «Метрики алгоритму»,
    видаляються назавжди. Поточний прогін ЦІЄЇ сесії (_RUNTIME["trajectory"])
    не чіпається — лише збережена історія на диску."""
    bl_idname = "wm.drone_clear_metrics_history"
    bl_label = "Очистити історію"
    bl_description = "Видалити всю збережену історію прогонів (out/autonomous_runs.json)"

    @classmethod
    def poll(cls, context):
        return _get_history_count() > 0

    def execute(self, context):
        sim_headless.clear_run_history()
        _RUNTIME["n_runs_history"] = 0
        self.report({'INFO'}, "Історію прогонів очищено")
        return {'FINISHED'}


class DRONE_OT_set_mode(bpy.types.Operator):
    """Перемкнути між ручним і автономним режимом керування.
    Автономний вимагає попередньо завантаженої траєкторії."""
    bl_idname = "wm.drone_set_mode"
    bl_label = "Режим польоту"
    bl_description = "Перемкнути між Ручним і Автономним режимом"

    mode: bpy.props.EnumProperty(
        name="Режим",
        items=[
            ("manual",     "Ручний",     "Керування з клавіатури"),
            ("autonomous", "Автономний", "Відтворення headless-траєкторії"),
        ],
        default="manual",
    )

    @classmethod
    def poll(cls, context):
        return True   # режим — лише налаштування; перемикати можна завжди

    def execute(self, context):
        global _FLIGHT_MODE
        _FLIGHT_MODE = self.mode
        label = "Ручний" if self.mode == "manual" else "Автономний"
        
        # Негайно зупиняємо старий оператор і респавнимо дрон
        _RUNTIME["running"] = False
        reset_flight()
        
        # Переконаємося, що камера активна для нового режиму
        _ensure_camera_is_active()
        
        self.report({'INFO'}, f"Режим змінено на: {label}. Дрон респавнено.")
        return {'FINISHED'}


class DRONE_OT_switch_role(bpy.types.Operator):
    """Перемкнути роль Адмін ⟷ Учасник. Можливо ЛИШЕ після вірного пароля
    (дослівно HACK_OT_switch_role коміту 2e2bc5d)."""
    bl_idname = "wm.drone_switch_role"
    bl_label = "Змінити режим"
    bl_description = "Перемкнути Адмін/Учасник (потрібен пароль)"

    password: bpy.props.StringProperty(name="Пароль", subtype='PASSWORD', default="")

    def invoke(self, context, event):
        self.password = ""
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context):
        col = self.layout.column()
        target = "Учасник" if is_admin() else "Адмін"
        col.label(text=f"Перемкнути на режим: {target}", icon='LOCKED')
        col.prop(self, "password")

    def execute(self, context):
        global _IS_ADMIN
        if self.password != ADMIN_PASSWORD:
            self.report({'ERROR'}, "Невірний пароль — режим не змінено")
            return {'CANCELLED'}
        _IS_ADMIN = not _IS_ADMIN
        self.report({'INFO'}, f"Режим: {'Адмін' if _IS_ADMIN else 'Учасник'}")
        return {'FINISHED'}


def _setup_compositor_for_vision(mode):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    
    # Очищуємо старі ноди
    for node in tree.nodes:
        tree.nodes.remove(node)
        
    rl = tree.nodes.new(type="CompositorNodeRLayers")
    rl.location = (0, 0)
    
    comp = tree.nodes.new(type="CompositorNodeComposite")
    comp.location = (1200, 0)
    
    if mode == "DAY":
        tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
        
    elif mode == "NIGHT":
        # Яскравіше нічне бачення: піднімаємо яскравість (Multiply) і фарбуємо в зелений
        math_node = tree.nodes.new(type="CompositorNodeMath")
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = 2.0 # Підсилюємо світло
        math_node.location = (200, 0)

        mix = tree.nodes.new(type="CompositorNodeMixRGB")
        mix.blend_type = 'MULTIPLY'
        mix.inputs[1].default_value = (0.2, 1.0, 0.2, 1.0) # Зелений
        mix.inputs[0].default_value = 1.0 # Fac
        mix.location = (400, 0)
        
        # Шум
        noise = tree.nodes.new(type="CompositorNodeTexture")
        tex = bpy.data.textures.new("NV_Noise", type='NOISE')
        tex.noise_scale = 0.05
        noise.texture = tex
        noise.location = (400, -200)

        mix_noise = tree.nodes.new(type="CompositorNodeMixRGB")
        mix_noise.blend_type = 'ADD'
        mix_noise.inputs[0].default_value = 0.1 # Сила шуму
        mix_noise.location = (600, 0)

        # Сяйво
        glare = tree.nodes.new(type="CompositorNodeGlare")
        glare.glare_type = 'GHOSTS'
        glare.threshold = 0.5
        glare.location = (800, 0)
        
        tree.links.new(rl.outputs["Image"], math_node.inputs[0])
        tree.links.new(math_node.outputs["Value"], mix.inputs[2])
        tree.links.new(mix.outputs["Image"], mix_noise.inputs[1])
        tree.links.new(noise.outputs["Value"], mix_noise.inputs[2])
        tree.links.new(mix_noise.outputs["Image"], glare.inputs["Image"])
        tree.links.new(glare.outputs["Image"], comp.inputs["Image"])
        
    elif mode == "THERMAL":
        # Z-depth to thermal color
        map_value = tree.nodes.new(type="CompositorNodeMapValue")
        map_value.offset = [-2.0]
        map_value.size = [0.035]
        map_value.use_min = True
        map_value.min = [0.0]
        map_value.use_max = True
        map_value.max = [1.0]
        map_value.location = (300, -200)
        
        color_ramp = tree.nodes.new(type="CompositorNodeValToRGB")
        color_ramp.color_ramp.elements[0].position = 0.0
        color_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0) # Близько (Гаряче)
        color_ramp.color_ramp.elements.new(0.2)
        color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 0.0, 1.0)
        color_ramp.color_ramp.elements.new(0.4)
        color_ramp.color_ramp.elements[2].color = (1.0, 0.0, 0.0, 1.0)
        color_ramp.color_ramp.elements.new(0.7)
        color_ramp.color_ramp.elements[3].color = (0.0, 0.0, 1.0, 1.0)
        color_ramp.color_ramp.elements[1].position = 1.0
        color_ramp.color_ramp.elements[-1].color = (0.0, 0.0, 0.2, 1.0) # Далеко (Холодно)
        color_ramp.location = (500, -200)
        
        tree.links.new(rl.outputs["Depth"], map_value.inputs["Value"])
        tree.links.new(map_value.outputs["Value"], color_ramp.inputs["Fac"])
        tree.links.new(color_ramp.outputs["Image"], comp.inputs["Image"])


def update_vision_mode(self, context):
    mode = self.vision_mode
    world = bpy.data.worlds.get("World")
    if world and world.use_nodes:
        tree = world.node_tree
        # Очистити ноди світу, окрім Output
        for node in tree.nodes:
            if node.type != 'OUTPUT_WORLD':
                tree.nodes.remove(node)
                
        out = tree.nodes.get("World Output") or tree.nodes.new("ShaderNodeOutputWorld")
        bg = tree.nodes.new("ShaderNodeBackground")
        tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
        
        if mode == "DAY":
            bg.inputs["Color"].default_value = (0.45, 0.62, 0.85, 1.0)
        elif mode == "THERMAL":
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        elif mode == "NIGHT":
            # Нічне небо з градієнтом (як на картинці: синій горизонт -> темний зеніт)
            tex_co = tree.nodes.new("ShaderNodeTexCoord")
            sep = tree.nodes.new("ShaderNodeSeparateXYZ")
            
            # Нормаль Z йде від 0 (горизонт) до 1 (зеніт)
            map_z = tree.nodes.new("ShaderNodeMapRange")
            map_z.inputs[1].default_value = -0.2 # From Min (трохи нижче горизонту)
            map_z.inputs[2].default_value = 0.8  # From Max (майже зеніт)
            
            sky_ramp = tree.nodes.new("ShaderNodeValToRGB")
            sky_ramp.color_ramp.elements[0].position = 0.0
            sky_ramp.color_ramp.elements[0].color = (0.0, 0.4, 0.9, 1.0) # Яскравий синій горизонт
            sky_ramp.color_ramp.elements[1].position = 1.0
            sky_ramp.color_ramp.elements[1].color = (0.0, 0.01, 0.05, 1.0) # Темно-синій зеніт
            
            # Додаємо зірки через Noise Texture
            noise = tree.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 400.0
            noise.inputs["Detail"].default_value = 10.0
            
            star_ramp = tree.nodes.new("ShaderNodeValToRGB")
            star_ramp.color_ramp.elements[0].position = 0.55
            star_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
            star_ramp.color_ramp.elements[1].position = 0.7
            star_ramp.color_ramp.elements[1].color = (2.0, 2.5, 3.0, 1.0) # Яскраві блакитнуваті зірки
            
            mix = tree.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = 'ADD'
            
            tree.links.new(tex_co.outputs["Normal"], sep.inputs["Vector"])
            tree.links.new(sep.outputs["Z"], map_z.inputs["Value"])
            tree.links.new(map_z.outputs["Result"], sky_ramp.inputs["Fac"])
            
            tree.links.new(noise.outputs["Fac"], star_ramp.inputs["Fac"])
            
            tree.links.new(sky_ramp.outputs["Color"], mix.inputs[1])
            tree.links.new(star_ramp.outputs["Color"], mix.inputs[2])
            tree.links.new(mix.outputs["Color"], bg.inputs["Color"])
                
    sun = bpy.data.lights.get("Sun")
    if sun:
        sun.energy = 3.0 if mode == "DAY" else 0.0

    spot = bpy.data.objects.get("NightSpot")
    if spot and spot.type == 'LIGHT':
        spot.data.energy = 2000.0 if mode == "NIGHT" else 0.0
        spot.data.spot_blend = 0.8 # М'якші краї
        spot.data.spot_size = math.radians(90) # Ширший кут
        
    # Місяць (об'єкт Sphere, що світиться)
    moon = bpy.data.objects.get("Moon")
    if mode == "NIGHT":
        if not moon:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=10.0, location=(-100, 100, 50))
            moon = bpy.context.active_object
            moon.name = "Moon"
            
            mat = bpy.data.materials.new(name="MoonMat")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs["Emission"].default_value = (1.0, 1.0, 0.8, 1.0)
            mat.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 5.0
            moon.data.materials.append(mat)
            
            # Додаємо легке заливне світло від місяця
            moon_light_data = bpy.data.lights.new("MoonLight", type="SUN")
            moon_light_data.energy = 0.1
            moon_light_data.color = (0.7, 0.8, 1.0)
            moon_light = bpy.data.objects.new("MoonLight", moon_light_data)
            bpy.context.collection.objects.link(moon_light)
            moon_light.rotation_euler = (math.radians(60), 0.0, math.radians(-45))
            moon_light.parent = moon
        else:
            moon.hide_render = False
            moon.hide_viewport = False
            ml = bpy.data.objects.get("MoonLight")
            if ml and ml.type == 'LIGHT':
                ml.data.energy = 0.1
    else:
        if moon:
            moon.hide_render = True
            moon.hide_viewport = True
            ml = bpy.data.objects.get("MoonLight")
            if ml and ml.type == 'LIGHT':
                ml.data.energy = 0.0
            
    _setup_compositor_for_vision(mode)


class DRONE_PT_panel(bpy.types.Panel):
    """N-панель 3D-в'юпорта, вкладка «Дрон»:
      • Перемикач режиму (Ручний / Автономний)
      • Підказка керування
      • Старт/Скид/Камера
      • Адмін: перегенерація мапи (ручна або автономна)
    Ручного вводу сіда немає — лише фіксована тестова або випадкова карта."""
    bl_label = "Дрон"
    bl_idname = "VIEW3D_PT_drone_manual"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Дрон"

    def draw(self, context):
        layout = self.layout
        admin = is_admin()
        mode = _FLIGHT_MODE

        # ── Роль (Адмін / Учасник) ──────────────────────────────────────────
        row = layout.row(align=True)
        row.label(text=("Режим: АДМІН" if admin else "Режим: Учасник"),
                  icon=('FUND' if admin else 'USER'))
        row.operator("wm.drone_switch_role", text="", icon='LOCKED')

        if admin and _APF_MAP_AVAILABLE:
            layout.operator("wm.drone_toggle_apf_map",
                            text=("Сховати APF карту" if _RUNTIME.get("show_apf")
                                  else "Показати APF карту"),
                            icon='FORCE_TURBULENCE',
                            depress=_RUNTIME.get("show_apf", False))

        layout.separator()

        # ── Візуальні режими ────────────────────────────────────────────────
        box_vision = layout.box()
        box_vision.label(text="Візуалізація", icon='RESTRICT_VIEW_OFF')
        box_vision.prop(context.scene, "vision_mode", text="")

        layout.separator()

        # ── Перемикач Ручний / Автономний ───────────────────────────────────
        box_mode = layout.box()
        box_mode.label(text="Режим керування", icon='SETTINGS')
        row_mode = box_mode.row(align=True)
        op_m = row_mode.operator("wm.drone_set_mode", text="Ручний",
                                 icon='RESTRICT_SELECT_OFF',
                                 depress=(mode == "manual"))
        op_m.mode = "manual"
        op_a = row_mode.operator("wm.drone_set_mode", text="Автономний",
                                 icon='WORLD',
                                 depress=(mode == "autonomous"))
        op_a.mode = "autonomous"

        layout.separator()

         # ── Ручний режим ─────────────────────────────────────────────────────
        if mode == "manual":
            box = layout.box()
            box.label(text="Ручне керування", icon='PLAY')
            box.label(text="W/S — тангаж")
            box.label(text="A/D — крен")
            box.label(text="Q/E — курс")
            box.label(text="Space/Shift — газ")
            box.label(text="C — camera · R — скид")

            layout.operator("wm.drone_manual", text="Старт ручного польоту", icon='PLAY')
            row = layout.row(align=True)
            row.operator("wm.drone_reset", text="Скид", icon='LOOP_BACK')
            row.operator("wm.drone_toggle_camera", text="Камера", icon='CAMERA_DATA')

            if _PLANE_UNLOCKED:
                vehicle_label = "дрон" if _RUNTIME.get("vehicle") == "drone" else "Потужноліт"
                layout.operator("wm.drone_switch_vehicle",
                                text=f"Пересісти на {'Потужноліт' if vehicle_label == 'дрон' else 'дрон'}",
                                icon='WORLD', depress=(vehicle_label != "дрон"))

            layout.separator()
            box_map = layout.box()
            md = _RUNTIME.get("md")
            current_seed = md.seed if md is not None else SEED
            box_map.label(text=f"Мапа (поточний seed: {current_seed})", icon='MESH_GRID')
            box_map.operator("wm.drone_regen_manual_map",
                             text="Нова випадкова мапа", icon='MOD_NOISE')

        # ── Автономний режим ─────────────────────────────────────────────────
        else:
            box2 = layout.box()
            box2.label(text="Автономний політ", icon='WORLD')

            # Головна кнопка — завантажує мапу (якщо потрібно) і запускає replay
            box2.operator("wm.drone_launch_autonomous",
                          text="▶  Запустити автономний", icon='PLAY')

            md = _RUNTIME.get("md")
            current_seed = md.seed if md is not None else SEED
            box2.label(text=f"Карта (seed: {current_seed})", icon='INFO')

            traj_ready = _RUNTIME.get("trajectory") is not None
            if traj_ready:
                n = len(_RUNTIME.get("trajectory") or [])
                box2.label(text=f"Траєкторія: {n} кадрів", icon='CHECKMARK')
            else:
                box2.label(text="(буде прораховано при старті)", icon='INFO')

            row2 = box2.row(align=True)
            row2.operator("wm.drone_toggle_camera", text="Камера", icon='CAMERA_DATA')

            layout.separator()
            box_amap = layout.box()
            box_amap.label(text="Зміна карти", icon='MESH_GRID')
            if admin:
                box_amap.operator("wm.drone_load_test_map",
                                  text="Тестова мапа (фікс. seed)", icon='MESH_GRID')
            box_amap.operator("wm.drone_load_random_map",
                              text="Нова випадкова мапа", icon='MOD_NOISE')

            layout.separator()
            box_metrics = layout.box()
            box_metrics.label(text="Якість алгоритму", icon='GRAPH')
            box_metrics.label(text=f"Збережено прогонів: {_get_history_count()}", icon='INFO')
            box_metrics.operator("wm.drone_show_metrics",
                                 text="Метрики алгоритму", icon='GRAPH')
            box_metrics.operator("wm.drone_clear_metrics_history",
                                 text="Очистити історію", icon='TRASH')


# ── Телеметрійна панель (дослівно за draw_hud() пісочниці engine_test.py) ─────────

# Геометрія телеметрійної панелі — модульні константи (не локальні змінні
# _draw_flight_hud), щоб _draw_apf_map міг прилаштуватись впритул до неї
# («зміщена до гіроскопу та інших даних про політ»), не дублюючи магічні числа.
_FLIGHT_HUD_X, _FLIGHT_HUD_Y = 14, 14
_FLIGHT_HUD_W, _FLIGHT_HUD_H = 250, 360   # H += 110 — вільна смуга внизу під гіроскоп


def _draw_flight_hud():
    """Персистентна панель угорі ліворуч: газ/висота/верт. швидкість/швидкість/
    тангаж/крен/курс + стовпчик газу з міткою висоти зависання; підказка керування й
    поточна камера — знизу екрана. Малюється, поки є телеметрія (політ стартував хоч
    раз); при заморожуванні фізики (аварія/межі) лишаються останні відомі значення."""
    tel = _RUNTIME.get("telemetry")
    if tel is None:
        return
    region = getattr(bpy.context, "region", None)
    if region is None:
        return
    w, h = region.width, region.height
    px, py = _FLIGHT_HUD_X, _FLIGHT_HUD_Y
    pw, ph = _FLIGHT_HUD_W, _FLIGHT_HUD_H
    top = h - py
    bottom = top - ph

    autonomous = tel.get("throttle") is None   # немає важеля газу → кінематичний replay
    is_plane = tel.get("vehicle") == "plane"
    stall = tel.get("stall_speed")

    try:
        import gpu
        from gpu_extras.batch import batch_for_shader

        def rect(x0, y0, x1, y1, color):
            sh.uniform_float("color", color)
            batch_for_shader(sh, "TRIS",
                             {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]},
                             indices=[(0, 1, 2), (0, 2, 3)]).draw(sh)

        gpu.state.blend_set("ALPHA")
        sh = gpu.shader.from_builtin("UNIFORM_COLOR")
        sh.bind()
        rect(px, bottom, px + pw, top, _HUD_PANEL_BG)         # панель-підкладка

        if not autonomous and not is_plane:
            # Стовпчик газу (праворуч у панелі) + мітка висоти зависання — лише
            # ручний режим квадрокоптера (кінематичний автопілот не має важеля
            # газу, а в літака немає «зависання» — див. гілку is_plane нижче).
            bx, bw, bh = px + 205, 22, 150
            bar_bottom = top - 44 - bh
            rect(bx, bar_bottom, bx + bw, bar_bottom + bh, _HUD_BAR_BG)
            fh = bh * max(0.0, min(1.0, tel["throttle"]))
            rect(bx, bar_bottom, bx + bw, bar_bottom + fh, _HUD_GREEN)
            hov_y = bar_bottom + bh * tel["hover_throttle"]
            sh.uniform_float("color", _HUD_DRONE_C)
            batch_for_shader(sh, "LINES",
                             {"pos": [(bx - 3, hov_y), (bx + bw + 3, hov_y)]}).draw(sh)
        elif not autonomous and is_plane and stall:
            # Літак: та сама смужка тепер — швидкість відносно звалювання
            # (0..2×stall_speed), з міткою порогу зриву потоку (замість «висоти
            # зависання», якої в літака немає — підйомна сила залежить від
            # швидкості, а не від нахилу вектора тяги).
            bx, bw, bh = px + 205, 22, 150
            bar_bottom = top - 44 - bh
            rect(bx, bar_bottom, bx + bw, bar_bottom + bh, _HUD_BAR_BG)
            speed_frac = max(0.0, min(1.0, tel["speed"] / (2.0 * stall)))
            bar_color = _HUD_GREEN if tel["speed"] >= stall else _RADAR_HIT
            rect(bx, bar_bottom, bx + bw, bar_bottom + bh * speed_frac, bar_color)
            stall_y = bar_bottom + bh * max(0.0, min(1.0, stall / (2.0 * stall)))
            sh.uniform_float("color", _RADAR_HIT)
            batch_for_shader(sh, "LINES",
                             {"pos": [(bx - 3, stall_y), (bx + bw + 3, stall_y)]}).draw(sh)
        gpu.state.blend_set("NONE")
    except Exception:   # noqa: BLE001
        pass

    try:
        import blf
        font = 0
        blf.color(font, *_HUD_DRONE_C)
        blf.size(font, 18)
        title = "АВТОНОМНИЙ ПОЛІТ" if autonomous else ("ПОТУЖНОЛІТ" if is_plane else "РУЧНИЙ ПОЛІТ")
        blf.position(font, px + 12, top - 26, 0)
        blf.draw(font, title)

        lines = [("висота %6.1f м" % tel["z"], _HUD_WHITE)]
        if not autonomous and not is_plane:
            lines.insert(0, ("газ    %3d %%" % round(tel["throttle"] * 100), _HUD_GREEN))
            lines.append(("верт   %+5.1f м/с" % tel["vz"], _HUD_WHITE))
        elif not autonomous and is_plane:
            lines.insert(0, ("газ    %3d %%" % round(tel["throttle"] * 100), _HUD_GREEN))
            stall_color = _RADAR_HIT if stall and tel["speed"] < stall else _HUD_WHITE
            lines.append(("звал.  %6.1f м/с" % (stall or 0.0), stall_color))
            lines.append(("верт   %+5.1f м/с" % tel["vz"], _HUD_WHITE))
        lines += [
            ("швидк  %6.1f м/с" % tel["speed"], _HUD_WHITE),
            ("тангаж %+5.0f°" % math.degrees(tel["pitch"]), _HUD_WHITE),
            ("крен   %+5.0f°" % math.degrees(tel["roll"]), _HUD_WHITE),
            ("курс   %+5.0f°" % math.degrees(tel["yaw"]), _HUD_WHITE),
        ]
        blf.size(font, 15)
        for i, (s, c) in enumerate(lines):
            blf.color(font, *c)
            blf.position(font, px + 12, top - 58 - i * 20, 0)
            blf.draw(font, s)

        blf.color(font, *_HUD_GREY)
        blf.size(font, 13)
        blf.position(font, 14, 40, 0)
        cam_label = "FPV" if _RUNTIME.get("cam_fpv") else "Chase"
        blf.draw(font, "камера: %s" % cam_label)
        blf.position(font, 14, 18, 0)
        blf.draw(font, "C камера · R скид" if autonomous else
                       "W/S тангаж · A/D крен · Space/Shift газ · Q/E курс · "
                       "C камера · R скид")

        if not autonomous and _PLANE_UNLOCKED:
            vehicle_label = "дрон" if _RUNTIME.get("vehicle") == "drone" else "Потужноліт"
            prompt = "Tab — перемкнути дрон⟷Потужноліт (зараз: %s)" % vehicle_label
            blf.size(font, 15)
            blf.color(font, *_HUD_DRONE_C)
            tw, _th = blf.dimensions(font, prompt)
            blf.position(font, (w - tw) / 2.0, 60, 0)
            blf.draw(font, prompt)
    except Exception:   # noqa: BLE001
        pass

    # Гіроскоп (штучний горизонт) — маленьке віконечко у вільній смузі внизу
    # ЦІЄЇ Ж панелі (не окремий повноекранний HUD) — саме це малося на увазі
    # під «реалістичним керуванням/HUD».
    _draw_gyro_widget(px + pw / 2.0, bottom + _GYRO_RADIUS + 14, tel["pitch"], tel["roll"])


_GYRO_RADIUS = 40


def _draw_gyro_widget(cx, cy, pitch, roll):
    """Малий гіроскоп (штучний горизонт) — квадратне віконечко ~2×_GYRO_RADIUS,
    обрізане scissor-тестом (щоб небо/земля не «протікали» за межі кола-рамки):
    лінія горизонту ОБЕРТАЄТЬСЯ з креном і ЗСУВАЄТЬСЯ з тангажем (та сама
    математика, що й у втраченому повноекранному кокпіт-HUD, лише в мініатюрі
    й БЕЗ шкали тангажу/стрічок — саме «візуалізація гіроскопа в маленькому
    віконечку», а не окремий HUD-режим). Нерухомий центральний маркер корпуса
    і кільце-рамка малюються ПОЗА обрізкою, завжди повністю видимі."""
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader

        R = _GYRO_RADIUS
        ppd = R / 35.0   # ±35° тангажу вміщується в радіус гіроскопа
        pitch_deg = math.degrees(pitch)

        def to_screen(xl, yl):
            cr, sr = math.cos(roll), math.sin(roll)
            return (cx + xl * cr - yl * sr, cy + xl * sr + yl * cr)

        def quad(pts, color):
            sh.uniform_float("color", color)
            batch_for_shader(sh, "TRIS", {"pos": pts}, indices=[(0, 1, 2), (0, 2, 3)]).draw(sh)

        gpu.state.blend_set("ALPHA")
        sh = gpu.shader.from_builtin("UNIFORM_COLOR")
        sh.bind()

        gpu.state.scissor_test_set(True)
        gpu.state.scissor_set(int(cx - R), int(cy - R), int(2 * R), int(2 * R))

        BIG = R * 6.0
        horizon_yl = pitch_deg * ppd
        sky = [to_screen(-BIG, horizon_yl), to_screen(BIG, horizon_yl),
               to_screen(BIG, horizon_yl + BIG), to_screen(-BIG, horizon_yl + BIG)]
        ground = [to_screen(-BIG, horizon_yl), to_screen(BIG, horizon_yl),
                  to_screen(BIG, horizon_yl - BIG), to_screen(-BIG, horizon_yl - BIG)]
        quad(sky, _FPV_SKY)
        quad(ground, _FPV_GROUND)

        sh.uniform_float("color", _FPV_HORIZON)
        batch_for_shader(sh, "LINES",
                         {"pos": [to_screen(-BIG, horizon_yl), to_screen(BIG, horizon_yl)]}).draw(sh)

        gpu.state.scissor_test_set(False)

        # Нерухомий маркер корпуса («ватерлінія») — НЕ обертається з креном.
        sh.uniform_float("color", _HUD_DRONE_C)
        batch_for_shader(sh, "LINES", {"pos": [(cx - R - 6, cy), (cx - 10, cy)]}).draw(sh)
        batch_for_shader(sh, "LINES", {"pos": [(cx + 10, cy), (cx + R + 6, cy)]}).draw(sh)

        # Кільце-рамка навколо гіроскопа.
        ring = [(cx + R * math.cos(2.0 * math.pi * k / 32), cy + R * math.sin(2.0 * math.pi * k / 32))
               for k in range(33)]
        sh.uniform_float("color", _HUD_GREY)
        batch_for_shader(sh, "LINE_STRIP", {"pos": ring}).draw(sh)

        gpu.state.blend_set("NONE")
    except Exception:   # noqa: BLE001
        pass


# ── APF-карта (карта потенціальних полів) — лише Адмін, діагностичний оверлей ─────
# Панель угорі праворуч (вільний кут: телеметрія — зверху ліворуч, радар лідара —
# знизу ліворуч). Синє = «спокійно» (близько до чекпоінта, далеко від перешкод),
# червоне = «небезпечно/далеко від цілі» — те саме blended_display_field, що й у
# apf_controller.py (незалежна нормалізація притягання/відштовхування, інакше
# притягання «засвітило» б усю шкалу й приховало перешкоди).
_APF_PANEL_SIZE = 200
_APF_GRID_N = 48
_APF_COOL = (0.20, 0.45, 0.95, 0.75)
_APF_HOT = (0.95, 0.20, 0.20, 0.85)


def _compute_apf_field(md, n=_APF_GRID_N):
    """Порахувати APF-поле поточної мапи (без запису в _RUNTIME — викликач сам
    кладе результат у кеш). Ціль притягання — ФІКСОВАНИЙ чекпоінт мапи напряму
    (md.checkpoints[0]), а не «морквина» A*-шляху (та рухається щотіку й існує
    лише під час активного автономного прогону) — це ОГЛЯДОВЕ поле «пташиного
    польоту» для всієї мапи, не тік-за-тіком траєкторія одного прогону."""
    ox, oy, orr = _RUNTIME.get("lidar_obs") or ([], [], [])
    cfg = _RUNTIME.get("cfg") or CFG
    p = APFParams()
    target = tuple(md.checkpoints[0][:2])
    xs, ys, U_a, U_r = potential_field_grid(
        ox, oy, orr, target, md.bounds, n,
        p.k_attract, p.k_repulse, p.influence_radius, d_floor=cfg.drone_radius)
    mix = blended_display_field(U_a, U_r)
    return {"seed": md.seed, "xs": xs, "ys": ys, "mix": mix, "target": target}


def _draw_apf_map():
    """Малює кешовану APF-теплокарту (якщо Адмін увімкнув її кнопкою в панелі).
    Сітка рахується ЛІНЬКВО й ЛИШЕ ОДИН РАЗ на мапу: якщо кеш відсутній чи
    належить іншому seed (мапу щойно змінили) — рахує заново тут-таки й кладе
    назад у _RUNTIME["apf_field"] — самозцілюється без окремого «мапу змінено»
    стану. Решта кадрів після цього лише перебудовують GPU-батч із готових
    чисел (дешево), саму сітку не перераховують — АНІМОВАНА лише позиція
    дрона (з телеметрії, щотіку), не саме поле."""
    if not (_APF_MAP_AVAILABLE and is_admin() and _RUNTIME.get("show_apf")):
        return
    md = _RUNTIME.get("md")
    if md is None:
        return
    field = _RUNTIME.get("apf_field")
    if field is None or field["seed"] != md.seed:
        try:
            field = _compute_apf_field(md)
        except Exception:   # noqa: BLE001
            return
        _RUNTIME["apf_field"] = field

    region = getattr(bpy.context, "region", None)
    if region is None:
        return
    w, h = region.width, region.height
    pw = ph = _APF_PANEL_SIZE
    # Впритул праворуч від телеметрійної панелі (де й гіроскоп), той самий
    # верхній відступ — інструменти польоту згруповані в одному кутку, а не
    # розкидані по в'юпорту.
    px = _FLIGHT_HUD_X + _FLIGHT_HUD_W + 14
    py = h - _FLIGHT_HUD_Y - ph
    bounds = md.bounds

    def _to_panel(x, y):
        return (px + (x + bounds) / (2.0 * bounds) * pw,
                py + (y + bounds) / (2.0 * bounds) * ph)

    xs, ys, mix = field["xs"], field["ys"], field["mix"]
    n = len(xs)
    cell_w = pw / n
    cell_h = ph / n
    tel = _RUNTIME.get("telemetry")

    try:
        import gpu
        from gpu_extras.batch import batch_for_shader

        gpu.state.blend_set("ALPHA")
        sh = gpu.shader.from_builtin("UNIFORM_COLOR")
        sh.bind()
        sh.uniform_float("color", _HUD_PANEL_BG)
        batch_for_shader(sh, "TRIS",
                         {"pos": [(px, py), (px + pw, py), (px + pw, py + ph), (px, py + ph)]},
                         indices=[(0, 1, 2), (0, 2, 3)]).draw(sh)

        sh2 = gpu.shader.from_builtin("SMOOTH_COLOR")
        verts, cols, idx = [], [], []
        for j in range(n):
            y0 = py + j * cell_h
            for i in range(n):
                f = float(mix[j, i])
                c = tuple(_APF_COOL[k] + (_APF_HOT[k] - _APF_COOL[k]) * f for k in range(4))
                x0 = px + i * cell_w
                b = len(verts)
                verts += [(x0, y0), (x0 + cell_w, y0), (x0 + cell_w, y0 + cell_h), (x0, y0 + cell_h)]
                cols += [c, c, c, c]
                idx += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
        batch_for_shader(sh2, "TRIS", {"pos": verts, "color": cols}, indices=idx).draw(sh2)

        sh.bind()
        sh.uniform_float("color", (1.0, 1.0, 1.0, 0.35))
        batch_for_shader(sh, "LINE_STRIP",
                         {"pos": [(px, py), (px + pw, py), (px + pw, py + ph),
                                 (px, py + ph), (px, py)]}).draw(sh)

        def _dot(x, y, color, r=4):
            cx, cy = _to_panel(x, y)
            sh.uniform_float("color", color)
            batch_for_shader(sh, "TRIS",
                             {"pos": [(cx - r, cy - r), (cx + r, cy - r),
                                     (cx + r, cy + r), (cx - r, cy + r)]},
                             indices=[(0, 1, 2), (0, 2, 3)]).draw(sh)

        _dot(field["target"][0], field["target"][1], _HUD_GREEN)
        _dot(md.start[0], md.start[1], _HUD_GREY, r=3)   # старт — тьмяний, лише орієнтир

        # Жива позиція дрона (АНІМОВАНА — з телеметрії, оновлюється щотіку,
        # на відміну від статичного поля/старту/чекпоінта вище). Курс —
        # коротка риска в напрямку yaw (панель світова, БЕЗ повороту за
        # курсом дрона, тож напрям малюється напряму через cos/sin(yaw)).
        if tel is not None and "x" in tel:
            _dot(tel["x"], tel["y"], _HUD_DRONE_C, r=5)
            hx, hy = _to_panel(tel["x"], tel["y"])
            yaw = tel.get("yaw", 0.0)
            tip = (hx + 10.0 * math.cos(yaw), hy + 10.0 * math.sin(yaw))
            sh.uniform_float("color", _HUD_DRONE_C)
            batch_for_shader(sh, "LINES", {"pos": [(hx, hy), tip]}).draw(sh)

        gpu.state.blend_set("NONE")
    except Exception:   # noqa: BLE001
        pass

    try:
        import blf
        # Панель упирається у верхній край в'юпорта (py+ph = h-14) — підпис
        # малюємо ВСЕРЕДИНІ панелі (не над нею, як «ЛІДАР» над радаром нижче:
        # там під написом є вільне місце до краю екрана, тут — ні), поверх
        # теплокарти, з тінню для читабельності на будь-якому кольорі клітинки.
        blf.enable(0, blf.SHADOW)
        blf.shadow(0, 3, 0.0, 0.0, 0.0, 1.0)
        blf.color(0, *_HUD_WHITE)
        blf.size(0, 12)
        blf.position(0, px + 6, py + ph - 16, 0)
        blf.draw(0, "APF")
        blf.disable(0, blf.SHADOW)
    except Exception:   # noqa: BLE001
        pass


def _draw_lidar_radar():
    """Радар-коло 2D-лідара в кутку в'юпорта (праворуч угорі): напівпрозорий сірий
    диск із радіальним градієнтом, поділений на сектори за бінами лідара (СВІТОВА
    рамка — index 0 → +X, як і сам binned_lidar_2d); сектор, де лідар щось виявив
    (< lidar_range), тоне в червоний — тим сильніше, чим ближче перешкода. Тонке
    кільце-межа = максимальна дальність; жовта риска — курс дрона (де «ніс»)."""
    tel = _RUNTIME.get("telemetry")
    if tel is None or not tel.get("lidar"):
        return
    lidar = tel["lidar"]
    n = len(lidar)
    region = getattr(bpy.context, "region", None)
    if region is None:
        return
    w, h = region.width, region.height
    cfg = _RUNTIME.get("cfg")
    max_r = cfg.lidar_range if cfg is not None else 8.0

    R = _RADAR_RADIUS
    # Переносимо в лівий нижній кут над текстовими підказками
    cx = R + _RADAR_MARGIN + 14
    cy = R + _RADAR_MARGIN + 70

    def _lerp(a, b, f):
        return a + (b - a) * f

    def _lerp_color(c1, c2, f):
        return tuple(_lerp(c1[k], c2[k], f) for k in range(4))

    try:
        import gpu
        from gpu_extras.batch import batch_for_shader

        gpu.state.blend_set("ALPHA")
        sh = gpu.shader.from_builtin("SMOOTH_COLOR")
        verts, cols, idx = [], [], []
        yaw = tel.get("yaw", 0.0)

        for i, d in enumerate(lidar):
            # Обертаємо сектори відносно yaw дрона, щоб напрямок "вперед" на радарі був завжди вгору (+Y)
            a_center = 2.0 * math.pi * i / n - yaw + math.pi / 2
            a0 = a_center - math.pi / n
            a1 = a_center + math.pi / n
            red_mix = max(0.0, min(1.0, 1.0 - d / max_r)) if d < max_r - 1e-6 else 0.0
            center_c = _lerp_color(_RADAR_CENTER_GRAY, _RADAR_HIT, red_mix)
            rim_c = _lerp_color(_RADAR_RIM_GRAY, _RADAR_HIT, red_mix)
            p0 = (cx + R * math.cos(a0), cy + R * math.sin(a0))
            p1 = (cx + R * math.cos(a1), cy + R * math.sin(a1))
            b = len(verts)
            verts += [(cx, cy), p0, p1]
            cols += [center_c, rim_c, rim_c]
            idx.append((b, b + 1, b + 2))
        batch_for_shader(sh, "TRIS", {"pos": verts, "color": cols}, indices=idx).draw(sh)

        sh2 = gpu.shader.from_builtin("UNIFORM_COLOR")
        sh2.bind()
        sh2.uniform_float("color", (1.0, 1.0, 1.0, 0.25))
        ring = [(cx + R * math.cos(2.0 * math.pi * k / 48), cy + R * math.sin(2.0 * math.pi * k / 48))
               for k in range(49)]
        batch_for_shader(sh2, "LINE_STRIP", {"pos": ring}).draw(sh2)

        # Жовта стрілка носа завжди вказує вертикально вгору (+Y)
        nose = (cx, cy + R + 8)
        sh2.uniform_float("color", _HUD_DRONE_C)
        batch_for_shader(sh2, "LINES", {"pos": [(cx, cy), nose]}).draw(sh2)
        gpu.state.blend_set("NONE")
    except Exception:   # noqa: BLE001
        pass

    try:
        import blf
        blf.color(0, *_HUD_GREY)
        blf.size(0, 12)
        txt = "ЛІДАР"
        tw, _th = blf.dimensions(0, txt)
        blf.position(0, cx - tw / 2.0, cy + R + 10, 0)
        blf.draw(0, txt)
    except Exception:   # noqa: BLE001
        pass


# ── Статусний оверлей (дослівно за _draw_status_overlay коміту 2e2bc5d) ───────────

def _draw_status_overlay():
    status = _RUNTIME.get("status", STATUS_RUNNING)
    if status not in STATUS_MESSAGES:
        return
    region = getattr(bpy.context, "region", None)
    if region is None:
        return
    w, h = region.width, region.height

    if status == STATUS_DISQUALIFIED:
        try:
            import gpu
            import random as _r
            from gpu_extras.batch import batch_for_shader
            gpu.state.blend_set("ALPHA")
            sh = gpu.shader.from_builtin("SMOOTH_COLOR")
            step = max(22, w // 46)
            verts, cols, idx = [], [], []
            n = 0
            y = 0
            while y < h:
                x = 0
                while x < w:
                    g = _r.random()
                    c = (g, g, g, 0.30)
                    b = n * 4
                    verts += [(x, y), (x + step, y), (x + step, y + step), (x, y + step)]
                    cols += [c, c, c, c]
                    idx += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
                    n += 1
                    x += step
                y += step
            batch_for_shader(sh, "TRIS", {"pos": verts, "color": cols},
                             indices=idx).draw(sh)
            gpu.state.blend_set("NONE")
        except Exception:   # noqa: BLE001
            pass

    try:
        import gpu
        from gpu_extras.batch import batch_for_shader
        gpu.state.blend_set("ALPHA")
        sh = gpu.shader.from_builtin("UNIFORM_COLOR")
        bh = h * 0.26
        y0 = h * 0.5 - bh / 2.0
        sh.bind()
        sh.uniform_float("color", (0.03, 0.03, 0.05, 0.62))
        batch_for_shader(sh, "TRIS",
                         {"pos": [(0, y0), (w, y0), (w, y0 + bh), (0, y0 + bh)]},
                         indices=[(0, 1, 2), (0, 2, 3)]).draw(sh)
        gpu.state.blend_set("NONE")
    except Exception:   # noqa: BLE001
        pass

    try:
        import blf
        txt = (_RUNTIME.get("crash_text") if status == STATUS_COLLISION else None) \
            or STATUS_MESSAGES[status]
        col = {STATUS_COLLISION: (1.0, 0.30, 0.24, 1.0),
               STATUS_DISQUALIFIED: (1.0, 0.90, 0.20, 1.0),
               STATUS_FINISHED: (0.45, 1.0, 0.55, 1.0),
               STATUS_TIMEOUT: (0.92, 0.92, 0.92, 1.0)}.get(status, (1, 1, 1, 1))
        font = 0
        blf.enable(font, blf.SHADOW)
        blf.shadow(font, 5, 0.0, 0.0, 0.0, 1.0)
        blf.shadow_offset(font, 3, -3)
        blf.color(font, *col)
        blf.size(font, max(34, int(h * 0.085)))
        tw, th = blf.dimensions(font, txt)
        blf.position(font, (w - tw) / 2.0, h * 0.5 + th * 0.15, 0)
        blf.draw(font, txt)
        hint = "R — знову"
        blf.size(font, max(15, int(h * 0.028)))
        hw, hh = blf.dimensions(font, hint)
        blf.color(font, 0.92, 0.92, 0.92, 1.0)
        blf.position(font, (w - hw) / 2.0, h * 0.5 - hh * 2.4, 0)
        blf.draw(font, hint)
        blf.disable(font, blf.SHADOW)
    except Exception:   # noqa: BLE001
        pass


_DETECT_RANGE = 24.0     # дальність AI-виявлення, м


def _detect_targets():
    """[(world_pos, label, color)] — об'єкти, які «розпізнає» камера дрона."""
    out = []
    named = {
        "Person":   ("Людина",   (0.25, 0.60, 1.0)),
        "FuelTank": ("Бензобак",  (0.98, 0.55, 0.10)),
        "Medkit":   ("Аптечка",   (0.95, 0.20, 0.20)),
    }
    for name, (label, col) in named.items():
        o = bpy.data.objects.get(name)
        if o is not None:
            out.append((o.matrix_world.translation.copy(), label, col))
    for o in bpy.data.objects:                       # фури-чекпоінти = машини
        if o.name.endswith("_truck"):
            out.append((o.matrix_world.translation.copy(), "Машина", (0.95, 0.85, 0.25)))
    md = _RUNTIME.get("md")                           # уламок техніки
    wi = getattr(md, "wreck_index", None) if md is not None else None
    if wi is not None:
        w = bpy.data.objects.get("TREE_%03d" % wi)
        if w is not None:
            out.append((w.matrix_world.translation.copy(), "Техніка", (0.80, 0.45, 0.95)))
    return out


def _draw_object_detection():
    """AI-виявлення об'єктів: рамка + підпис над кожним об'єктом у полі зору дрона."""
    tel = _RUNTIME.get("telemetry")
    region = getattr(bpy.context, "region", None)
    rv3d = getattr(bpy.context, "region_data", None)
    if tel is None or region is None or rv3d is None:
        return
    try:
        import gpu
        import blf
        import math as _m
        from gpu_extras.batch import batch_for_shader
        from bpy_extras.view3d_utils import location_3d_to_region_2d

        drone = (tel["x"], tel["y"], tel["z"])
        sh = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.line_width_set(2.0)
        gpu.state.blend_set("ALPHA")
        font = 0
        n_det = 0
        for wpos, label, col in _detect_targets():
            d = _m.dist(drone, (wpos.x, wpos.y, wpos.z))
            if d > _DETECT_RANGE:
                continue
            p = location_3d_to_region_2d(region, rv3d, wpos)
            if p is None:                            # позаду камери
                continue
            s = max(26.0, 560.0 / max(1.5, d))       # рамка більша зблизька
            x0, y0, x1, y1 = p.x - s / 2, p.y - s / 2, p.x + s / 2, p.y + s / 2
            sh.bind()
            sh.uniform_float("color", (col[0], col[1], col[2], 0.95))
            batch_for_shader(sh, "LINE_STRIP",
                             {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]}).draw(sh)
            conf = 82 + (hash(label) % 17)           # псевдо-впевненість 82–98%
            blf.size(font, 13)
            blf.color(font, col[0], col[1], col[2], 1.0)
            blf.position(font, x0, y1 + 4, 0)
            blf.draw(font, "%s %d%%" % (label, conf))
            n_det += 1
        if n_det:                                    # лічильник угорі-праворуч
            blf.size(font, 15)
            blf.color(font, 0.2, 1.0, 0.4, 1.0)
            blf.position(font, region.width - 210, region.height - 40, 0)
            blf.draw(font, "AI DETECT: %d об'єктів" % n_det)
        gpu.state.line_width_set(1.0)
    except Exception:                                # noqa: BLE001 — оверлей не має ламати гру
        pass


def _draw_overlay():
    """Єдиний зареєстрований колбек: телеметрійна панель (з гіроскопом) угорі
    ліворуч — ОДНА й та сама і в Chase, і в FPV — + радар лідара, зверху —
    статусний оверлей (аварія/межі), якщо є термінальний статус.

    _enforce_camera_lock() викликається тут БЕЗУМОВНО (а не лише в _tick()
    модальних операторів) — інакше вільна орбіта мишею була б доступна в
    будь-яку мить, поки жоден політ не «running» (між раундами, одразу після
    старту Blender, після зміни режиму тощо): цей колбек — POST_PIXEL
    draw_handler, який Blender викликає щоразу під час перемальовування
    в'юпорта, тож блокування діє завжди, незалежно від стану польоту."""
    _enforce_camera_lock()
    _draw_flight_hud()
    _draw_lidar_radar()
    _draw_apf_map()
    _draw_object_detection()
    _draw_status_overlay()


def _register_hud():
    global _HUD_HANDLE
    if _HUD_HANDLE is None:
        _HUD_HANDLE = bpy.types.SpaceView3D.draw_handler_add(_draw_overlay, (), 'WINDOW', 'POST_PIXEL')


def _autostart():
    """Автостарт після реєстрації: запустити оператор у поточному режимі (_FLIGHT_MODE)
    і перейти в кіоск-вид (лише в'юпорт)."""
    try:
        if _FLIGHT_MODE == "manual":
            bpy.ops.wm.drone_manual('INVOKE_DEFAULT')
        else:
            # Автономний: якщо траєкторія ще не завантажена — завантажити тестову
            if _RUNTIME.get("trajectory") is None:
                _load_autonomous_map(SEED)
            bpy.ops.wm.drone_autonomous('INVOKE_DEFAULT')
    except Exception as exc:                  # noqa: BLE001
        print("autostart error:", exc)
    try:
        _enter_kiosk_view()                    # лише в'юпорт — без шапки/тулбара/N-панелі
    except Exception as exc:                   # noqa: BLE001
        print("kiosk view error:", exc)
    return None


def _selfcheck():
    """Headless: побудова середовища, мапа пози, і межі/зіткнення на синтетичних точках."""
    md = _RUNTIME["md"]
    n_tree = sum(1 for o in bpy.data.objects if o.name.startswith("TREE_"))
    n_truck = sum(1 for o in bpy.data.objects if o.name.endswith("_truck"))
    n_zone = sum(1 for o in bpy.data.objects if o.name.endswith("_zone"))
    n_obst = sum(1 for o in bpy.data.objects
                 if o.name.startswith("OBST_") or o.name.startswith("DECOR_"))
    terr = any(o.name.startswith("TERRAIN") for o in bpy.data.objects)
    print("SELFCHECK: дерев=%d, фур=%d (зон=%d), перешкод=%d, рельєф=%s, дрон=%s, старт=%s"
          % (n_tree, n_truck, n_zone, n_obst, terr, "Drone" in bpy.data.objects,
             tuple(round(v, 1) for v in md.start)))

    q = RealisticQuad(QuadParams())
    q.reset(_RUNTIME["start"])
    for _ in range(30):
        q.step(0.0, 1.0, 0.0, 0.0, 1.0 / KEY_HZ)
    e = pose_matrix(q).to_euler('XYZ')
    ok = abs(e.y - q.pitch) < 1e-4 and abs(e.x) < 1e-4 and abs(e.z) < 1e-4
    print("SELFCHECK: орієнтація (pitch→euler.y) →", "OK" if ok else "FAIL")

    cfg, terrain = _RUNTIME["cfg"], _RUNTIME["terrain"]
    s_run = collision_and_bounds_status(_RUNTIME["start"], md, terrain, cfg)
    s_bounds = collision_and_bounds_status((cfg.bounds + 5.0, 0.0, md.start[2]), md, terrain, cfg)
    s_ceiling = collision_and_bounds_status((0.0, 0.0, md.ceiling + 5.0), md, terrain, cfg)
    gz = terrain.height_at(0.0, 0.0)
    s_ground = collision_and_bounds_status((0.0, 0.0, gz), md, terrain, cfg)
    print("SELFCHECK: старт=%s (očік RUNNING), поза межею-x=%s, над стелею=%s, "
          "торкання землі=%s (усі очік. відповідно DISQUALIFIED/DISQUALIFIED/COLLISION)"
          % (s_run, s_bounds, s_ceiling, s_ground))
    ok2 = (s_run == STATUS_RUNNING and s_bounds == STATUS_DISQUALIFIED
           and s_ceiling == STATUS_DISQUALIFIED and s_ground == STATUS_COLLISION)
    print("SELFCHECK: межі/зіткнення →", "OK" if ok2 else "FAIL")

    # Автономний конвеєр: headless-симуляція → завантаження таблиці в сцену → replay-тік
    result = sim_headless.simulate(seed=SEED)
    print("SELFCHECK: автономний прогін статус=%s кадрів=%d"
          % (result["meta"]["final_status"], result["meta"]["n_frames"]))
    md2 = MapData.from_dict(result["map"])
    cfg2 = dataclasses.replace(CFG, n_trees=result["meta"]["n_trees"])
    build_scene_from_mapdata(md2, cfg2)
    _RUNTIME["trajectory"] = result["frames"]
    _RUNTIME["traj_hz"] = result["meta"]["sim_hz"]

    class _FakeOp:
        pass
    op = _FakeOp()
    for _ in range(5):
        DRONE_OT_autonomous._tick(op)
    tel = _RUNTIME["telemetry"]
    ok3 = (_RUNTIME["traj_frame"] == 5 and tel is not None
          and "lidar" in tel and len(tel["lidar"]) == cfg2.lidar_n_az)
    print("SELFCHECK: replay-тік (5 кадрів, телеметрія+лідар) →", "OK" if ok3 else "FAIL")


# ── ГОЛОСОВИЙ МІСТ: Blender слухає команди з файлу (їх пише веб-сервер voice/) ──
_VOICE_CMD_FILE = str(sim_headless.OUT_DIR / "voice_cmd.txt")


def _voice_execute(cmd: str):
    """Виконати голосову команду: лети / новий маршрут / стоп."""
    c = cmd.strip().lower()
    try:
        if c in ("fly", "лети", "старт", "вперед", "полетіли", "go"):
            if _RUNTIME.get("trajectory") is None:
                _load_autonomous_map(SEED)
            bpy.ops.wm.drone_autonomous('INVOKE_DEFAULT')
            print("ГОЛОС: лети →", c)
        elif c in ("route", "маршрут", "побудуй", "новий", "новий маршрут", "rebuild"):
            import random as _r
            _RUNTIME["running"] = False
            _load_autonomous_map(_r.randint(0, 100000))     # нова мапа + безпечний A*-маршрут
            bpy.ops.wm.drone_autonomous('INVOKE_DEFAULT')
            print("ГОЛОС: новий маршрут →", c)
        elif c in ("stop", "стоп", "стій", "зупинись"):
            _RUNTIME["running"] = False
            print("ГОЛОС: стоп")
        else:
            print("ГОЛОС: невідома команда:", c)
    except Exception as exc:                                 # noqa: BLE001
        print("ГОЛОС: помилка виконання:", exc)


def _voice_poll():
    """Таймер: раз на 0.4 с читає файл-команду й виконує, тоді видаляє його."""
    try:
        import os
        if os.path.exists(_VOICE_CMD_FILE):
            with open(_VOICE_CMD_FILE, encoding="utf-8") as f:
                cmd = f.read()
            os.remove(_VOICE_CMD_FILE)
            if cmd.strip():
                _voice_execute(cmd)
    except Exception as exc:                                 # noqa: BLE001
        print("ГОЛОС: помилка опитування:", exc)
    return 0.4


def main():
    if bpy.app.background:
        build_scene(seed=SEED)      # детермінований сід — відтворювана самоперевірка
        _selfcheck()
        return

    seed = _random_seed()           # кожен запуск учасника — нова випадкова мапа
    build_scene(seed=seed)          # ручний режим (сирий рушій) — типовий старт

    for cls in (DRONE_OT_manual, DRONE_OT_autonomous, DRONE_OT_reset,
                DRONE_OT_toggle_camera, DRONE_OT_switch_vehicle,
                DRONE_OT_load_test_map,
                DRONE_OT_load_random_map, DRONE_OT_toggle_apf_map, DRONE_OT_regen_manual_map,
                DRONE_OT_launch_autonomous, DRONE_OT_show_metrics,
                DRONE_OT_clear_metrics_history,
                DRONE_OT_set_mode,
                DRONE_OT_switch_role, DRONE_PT_panel):
        bpy.utils.register_class(cls)

    bpy.types.Scene.vision_mode = bpy.props.EnumProperty(
        name="Vision Mode",
        description="Select visual mode",
        items=[
            ("DAY", "День", "Standard day lighting"),
            ("NIGHT", "Нічне бачення", "Night vision with spotlight"),
            ("THERMAL", "Тепловізор", "Thermal imaging")
        ],
        default="DAY",
        update=update_vision_mode
    )

    _register_hud()
    bpy.app.timers.register(_autostart, first_interval=0.4)
    bpy.app.timers.register(_voice_poll, first_interval=1.0)   # слухаємо голосові команди


if __name__ == "__main__":
    main()
