# -*- coding: utf-8 -*-
"""
Реалістичний рушій квадрокоптера (angle-mode) — САМОДОСТАТНІЙ для пісочниці.

Це РОЗРОБНИЦЬКА пісочниця (не для учасників/адмінів): окремий рушій для власного
тестування «відчуття» польоту. Ані Blender, ані numpy — лише стандартний math, тож
запускається однією командою в терміналі разом із sandbox/engine_test.py.

Модель (як у справжнього FPV у режимі self-level):
  • стіки задають КУТ нахилу (тангаж/крен), а не швидкість;
  • нахил ВЕКТОРИЗУЄ тягу → горизонтальне прискорення (щоб летіти — нахились);
  • ГАЗ — ПЕРСИСТЕНТНИЙ важіль-колектив (як у War Thunder): клавіша/стік лише
    ЗМІНЮЄ % тяги, і він тримається (не миттєвий ривок);
  • діють гравітація й КВАДРАТИЧНИЙ опір повітря F_d = −k·|v|·v;
  • нахил має інерцію (стала часу tau_att).

Параметри — у QuadParams; підкручуй під бажане «відчуття». За замовчуванням —
ШВИДШІ, ніж у грі (більший кут нахилу й менший опір → вища гранична швидкість).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

Vec3 = Tuple[float, float, float]

_G = 9.81  # прискорення вільного падіння, м/с²


@dataclass
class QuadParams:
    """Параметри апарата (SI). Дефолти — «швидший» відгук, ніж у грі."""
    mass: float = 0.9            # маса, кг
    twr: float = 3.2             # тягоозброєність (макс. тяга = twr·m·g)
    drag: float = 0.03           # коеф. квадратичного опору (менше → вища швидкість)
    max_tilt_deg: float = 45.0   # макс. кут нахилу (більше → більше гориз. прискорення)
    tau_att: float = 0.10        # інерція нахилу, с (менше → чіткіше)
    max_yaw_rate: float = 2.8    # макс. кутова швидкість курсу, рад/с
    throttle_rate: float = 0.8   # швидкість зміни газу-важеля (частка/с)
    substep: float = 0.005       # крок інтегрування, с


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else float(v)


class RealisticQuad:
    """Квадрокоптер self-level. step(throttle_cmd, pitch_cmd, roll_cmd, yaw_rate_cmd,
    dt): throttle_cmd — ЗМІНА газу-важеля; решта — кути/курс ∈ [−1,1]."""

    def __init__(self, params: QuadParams = None):
        self.p = params or QuadParams()
        self.max_tilt = math.radians(self.p.max_tilt_deg)
        self.hover_throttle = 1.0 / self.p.twr
        self.reset((0.0, 0.0, 3.0))

    # ── Життєвий цикл ────────────────────────────────────────────────────────────
    def reset(self, pos: Vec3, yaw: float = 0.0) -> None:
        self.pos: List[float] = [float(pos[0]), float(pos[1]), float(pos[2])]
        self.vel: List[float] = [0.0, 0.0, 0.0]
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = float(yaw)
        self.throttle = self.hover_throttle   # старт із газу зависання (важіль тримається)

    # ── Крок фізики ──────────────────────────────────────────────────────────────
    def step(self, throttle_cmd: float, pitch_cmd: float, roll_cmd: float,
             yaw_rate_cmd: float, dt: float) -> None:
        throttle_cmd = _clamp(throttle_cmd)
        pitch_cmd = _clamp(pitch_cmd)
        roll_cmd = _clamp(roll_cmd)
        yaw_rate_cmd = _clamp(yaw_rate_cmd)
        n = max(1, int(math.ceil(dt / max(1e-4, self.p.substep))))
        h = dt / n
        for _ in range(n):
            self._integrate(throttle_cmd, pitch_cmd, roll_cmd, yaw_rate_cmd, h)

    def _integrate(self, throttle_cmd, pitch_cmd, roll_cmd, yaw_rate_cmd, h):
        p = self.p
        # 1) Курс
        self.yaw = (self.yaw + yaw_rate_cmd * p.max_yaw_rate * h
                    + math.pi) % (2.0 * math.pi) - math.pi
        # 2) Angle mode: цільові кути; апарат наздоганяє з інерцією (tau_att)
        a_att = 1.0 - math.exp(-h / max(1e-3, p.tau_att))
        self.pitch += (pitch_cmd * self.max_tilt - self.pitch) * a_att
        self.roll += (roll_cmd * self.max_tilt - self.roll) * a_att
        # 3) ПЕРСИСТЕНТНИЙ газ-важіль (колектив): клавіша лише змінює % газу
        self.throttle = min(1.0, max(0.0,
                                     self.throttle + throttle_cmd * p.throttle_rate * h))
        thrust = self.throttle * p.twr * p.mass * _G   # 0..T_max; зависання при 1/TWR
        # 4) Напрямок тяги (вісь «вгору» корпуса) у світі: R_zyx(ψ,θ,φ)·[0,0,1]
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        ux = cr * sp * cy + sr * sy
        uy = cr * sp * sy - sr * cy
        uz = cr * cp
        # 5) Прискорення: тяга + гравітація + квадратичний опір
        vx, vy, vz = self.vel
        spd = math.sqrt(vx * vx + vy * vy + vz * vz)
        km = p.drag / p.mass
        tm = thrust / p.mass
        ax = tm * ux - km * spd * vx
        ay = tm * uy - km * spd * vy
        az = tm * uz - _G - km * spd * vz
        # 6) Напів-неявний Ейлер
        self.vel[0] = vx + ax * h
        self.vel[1] = vy + ay * h
        self.vel[2] = vz + az * h
        self.pos[0] += self.vel[0] * h
        self.pos[1] += self.vel[1] * h
        self.pos[2] += self.vel[2] * h

    # ── Похідні величини ─────────────────────────────────────────────────────────
    @property
    def speed(self) -> float:
        vx, vy, vz = self.vel
        return math.sqrt(vx * vx + vy * vy + vz * vz)


@dataclass
class PlaneParams:
    """Параметри «Потужнольоту» — КАЗУАЛЬНА (аркадна) фізика «як у GTA»: жодних
    сил (підйомна/опір/вага) і жодного кута атаки/звалювання — апарат просто
    ЛЕТИТЬ ТУДИ, КУДИ ДИВИТЬСЯ НІС, зі швидкістю від важеля газу. Крен НАПРЯМУ
    задає швидкість повороту курсу (банкований віраж без жодної аеродинаміки).
    Найпростіша й найпрощавальніша модель із усіх — неможливо «розбити» через
    фізику: швидкість завжди в межах [min_speed, max_speed], зриву потоку нема."""
    min_speed: float = 5.0          # м/с — швидкість на нульовому газу (як планер, що завжди трохи летить)
    max_speed: float = 26.0         # м/с — швидкість на повному газу
    throttle_rate: float = 0.7      # швидкість зміни важеля газу, частка/с (дає плавний розгін/гальмування)
    pitch_rate_max: float = 1.2     # макс. кутова швидкість тангажу, рад/с
    roll_rate_max: float = 3.2      # макс. кутова швидкість крену, рад/с
    max_pitch_deg: float = 55.0     # обмежувач тангажу (щоб не крутило через голову)
    max_roll_deg: float = 55.0      # обмежувач крену — тримає керований, «пробачливий» банк
    roll_stability: float = 0.9     # сильне самовирівнювання крену — як тільки відпустив A/D, сам рівняється
    pitch_stability: float = 0.6    # самовирівнювання тангажу
    turn_gain: float = 1.1          # крен(рад)·turn_gain = швидкість повороту курсу (рад/с) — БАНК ⇒ ПОВОРОТ,
                                    # напряму, без жодної проміжної фізики (немає більше «флюгерної стійкості»
                                    # чи ковзання — курс МИТТЄВО й передбачувано слідує за креном)
    substep: float = 0.02           # крок інтегрування, с


class RealisticPlane:
    """«Потужноліт» — КІНЕМАТИЧНА (не силова) модель: активується автоматично,
    щойно пілот пересідає на нього (switch_to_plane у blender_manual.py).
    На відміну від першої (силової, з підйомною силою/опором/вагою) і другої
    (спрощеної, зі сталим CL) версій — тут ЖОДНИХ сил і жодного інтегрування
    прискорення: кожен тік просто СТАВИТЬ швидкість НАПРЯМУ з важеля газу й
    рухає апарат уздовж носа. Пряме й передбачуване, як в аркадній грі:
      • швидкість = f(газ) — без інерції маси/тяги/опору;
      • курс = f(крен) — банк одразу повертає, без ковзання чи флюгерної
        стійкості (їх просто нема — рухатись боком неможливо за конструкцією);
      • жодного зриву потоку/звалювання — швидкість завжди в безпечних межах.
    Публічний інтерфейс (step/reset/pos/vel/pitch/roll/yaw/throttle/speed) —
    той самий, що й у RealisticQuad, тож решта коду (поза/HUD/колізії) не бачить
    різниці між апаратами."""

    def __init__(self, params: PlaneParams = None):
        self.p = params or PlaneParams()
        self.max_roll = math.radians(self.p.max_roll_deg)
        self.max_pitch = math.radians(self.p.max_pitch_deg)
        # Для сумісності HUD (раніше показував «звалювання») — тут це просто
        # нижня межа швидкості, нижче якої апарат фізично опуститись не може.
        self.stall_speed = self.p.min_speed
        self.reset((0.0, 0.0, 3.0))

    # ── Життєвий цикл ────────────────────────────────────────────────────────────
    def reset(self, pos: Vec3, yaw: float = 0.0) -> None:
        self.pos: List[float] = [float(pos[0]), float(pos[1]), float(pos[2])]
        self.yaw = float(yaw)
        self.pitch = 0.0
        self.roll = 0.0
        self.throttle = 0.5
        speed = self.p.min_speed + self.throttle * (self.p.max_speed - self.p.min_speed)
        self.vel: List[float] = [speed * math.cos(self.yaw), speed * math.sin(self.yaw), 0.0]

    # ── Крок фізики (насправді — кінематика, без сил) ───────────────────────────
    def step(self, throttle_cmd: float, pitch_cmd: float, roll_cmd: float,
             yaw_rate_cmd: float, dt: float) -> None:
        throttle_cmd = _clamp(throttle_cmd)
        pitch_cmd = _clamp(pitch_cmd)
        roll_cmd = _clamp(roll_cmd)
        n = max(1, int(math.ceil(dt / max(1e-4, self.p.substep))))
        h = dt / n
        for _ in range(n):
            self._integrate(throttle_cmd, pitch_cmd, roll_cmd, h)

    def _integrate(self, throttle_cmd, pitch_cmd, roll_cmd, h):
        p = self.p

        # Крен/тангаж — керовані ШВИДКІСТЮ зміни + сильна автостабілізація
        # (відпустив клавішу — сам вирівнюється, «пробачливо», як в аркаді).
        self.roll += (roll_cmd * p.roll_rate_max - p.roll_stability * self.roll) * h
        self.roll = max(-self.max_roll, min(self.max_roll, self.roll))
        self.pitch += (pitch_cmd * p.pitch_rate_max - p.pitch_stability * self.pitch) * h
        self.pitch = max(-self.max_pitch, min(self.max_pitch, self.pitch))

        # Курс — НАПРЯМУ від крену (банк ⇒ поворот), без жодної аеродинаміки:
        # крен управо (D, додатний) повертає праворуч (від'ємна зміна курсу).
        self.yaw = (self.yaw - self.roll * p.turn_gain * h + math.pi) % (2.0 * math.pi) - math.pi

        # ПЕРСИСТЕНТНИЙ важіль газу (як і в квада) — клавіша лише змінює %;
        # швидкість — НАПРЯМУ з газу, без маси/тяги/опору/інерції розгону.
        self.throttle = min(1.0, max(0.0, self.throttle + throttle_cmd * p.throttle_rate * h))
        speed = p.min_speed + self.throttle * (p.max_speed - p.min_speed)

        # Ніс = напрямок руху НАПРЯМУ (кінематика: рухається туди, куди дивиться).
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.vel[0] = speed * cy * cp
        self.vel[1] = speed * sy * cp
        self.vel[2] = speed * -sp

        self.pos[0] += self.vel[0] * h
        self.pos[1] += self.vel[1] * h
        self.pos[2] += self.vel[2] * h

    # ── Похідні величини ─────────────────────────────────────────────────────────
    @property
    def speed(self) -> float:
        vx, vy, vz = self.vel
        return math.sqrt(vx * vx + vy * vy + vz * vz)
