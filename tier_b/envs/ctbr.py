"""CTBR: [колективна тяга, body-rate setpoints] → оберти моторів CF2X.

PURE NUMPY — тестується без pybullet. Конвеєр:

  a ∈ [-1,1]⁴ ──decode──▶ (c, ω_sp) ──rate P──▶ τ ──mixer──▶ f моторів ──▶ RPM

Знакова конвенція CF2X знята з ІНСТАЛЬОВАНОГО gym-pybullet-drones
(BaseAviary._dynamics, гілка DroneModel.CF2X, і DSLPIDControl.MIXER_MATRIX —
вони узгоджені між собою):

  τx (roll)  = −(f0 + f1 − f2 − f3)·d,  d = L/√2      → знаки (−,−,+,+)
  τy (pitch) =  (−f0 + f1 + f2 − f3)·d                → знаки (−,+,+,−)
  τz (yaw)   =  (−m0 + m1 − m2 + m3),  mᵢ = κ·fᵢ, κ=KM/KF → знаки (−,+,−,+)

Сатурація «тяга-пріоритет»: якщо запит не влазить у [0, f_max] на мотор,
зберігаємо T і масштабуємо торки максимальним допустимим α ∈ [0,1] —
через це c→2g деградує плавно (тане запас на торки), а не ламається.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# знакові рядки CF2X (див. docstring)
SIGN_ROLL = np.array([-1.0, -1.0, 1.0, 1.0])
SIGN_PITCH = np.array([-1.0, 1.0, 1.0, -1.0])
SIGN_YAW = np.array([-1.0, 1.0, -1.0, 1.0])

# --- дефолти = config.yaml["action"] + константи CF2X (step-0 аудит) ---------
C_MAX_G = 2.0
RATE_LIMITS = np.array([6.0, 6.0, 3.0])
TAU_C = 0.05


@dataclass(frozen=True)
class CTBRParams:
    """Фізичні константи дрона + ліміти CTBR. Дефолти — CF2X із URDF."""

    m: float = 0.027
    g: float = 9.8
    kf: float = 3.16e-10
    km: float = 7.94e-12
    arm_l: float = 0.0397
    max_rpm: float = 21702.6437752510
    j_diag: tuple = (1.4e-5, 1.4e-5, 2.17e-5)
    c_max_g: float = C_MAX_G
    rate_limits: tuple = (6.0, 6.0, 3.0)
    tau_c: float = TAU_C

    @property
    def f_motor_max(self) -> float:
        return self.kf * self.max_rpm**2

    @property
    def c_max(self) -> float:
        return self.c_max_g * self.g


def allocation_matrix(p: CTBRParams) -> np.ndarray:
    """A: тяги моторів f → [T, τx, τy, τz]. Обертаємо один раз."""
    d = p.arm_l / np.sqrt(2.0)
    kappa = p.km / p.kf
    return np.vstack([
        np.ones(4),
        d * SIGN_ROLL,
        d * SIGN_PITCH,
        kappa * SIGN_YAW,
    ])


def decode_action(a: np.ndarray, p: CTBRParams) -> tuple[float, np.ndarray]:
    """a ∈ [-1,1]⁴ → (c [м/с², mass-normalized], ω_sp [рад/с])."""
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    c = (a[0] + 1.0) / 2.0 * p.c_max
    omega_sp = a[1:4] * np.asarray(p.rate_limits)
    return float(c), omega_sp


def rate_p_torques(omega_sp: np.ndarray, omega: np.ndarray, p: CTBRParams) -> np.ndarray:
    """Betaflight-стиль P-закон (без I — stateless): τ = (J/τ_c)·(ω_sp − ω)."""
    k = np.asarray(p.j_diag) / p.tau_c
    return k * (np.asarray(omega_sp) - np.asarray(omega))


def mix(thrust_total: float, torques: np.ndarray, p: CTBRParams,
        _ainv_cache: dict = {}) -> np.ndarray:
    """[T, τ] → тяги моторів f (Н) із тяга-пріоритетною сатурацією."""
    key = (p.arm_l, p.km, p.kf)
    if key not in _ainv_cache:
        _ainv_cache[key] = np.linalg.inv(allocation_matrix(p))
    ainv = _ainv_cache[key]

    f_max = p.f_motor_max
    thrust_total = float(np.clip(thrust_total, 0.0, 4.0 * f_max))
    f = ainv @ np.concatenate([[thrust_total], np.asarray(torques, dtype=np.float64)])

    lo, hi = 0.0, f_max
    if np.all((f >= lo) & (f <= hi)):
        return f

    # тяга-пріоритет: f = f_T + α·df; знайти найбільше α ∈ [0,1], що влазить
    f_t = thrust_total / 4.0
    df = f - f_t
    alpha = 1.0
    for i in range(4):
        if df[i] > 1e-12:
            alpha = min(alpha, (hi - f_t) / df[i])
        elif df[i] < -1e-12:
            alpha = min(alpha, (lo - f_t) / df[i])
    alpha = max(0.0, alpha)
    return np.clip(f_t + alpha * df, lo, hi)


def rpm_from_thrusts(f: np.ndarray, p: CTBRParams) -> np.ndarray:
    return np.clip(np.sqrt(np.maximum(f, 0.0) / p.kf), 0.0, p.max_rpm)


def ctbr_to_rpm(action: np.ndarray, omega_gyro: np.ndarray, p: CTBRParams) -> np.ndarray:
    """Повний конвеєр: дія політики + гіроскоп → RPM чотирьох моторів."""
    c, omega_sp = decode_action(action, p)
    tau = rate_p_torques(omega_sp, omega_gyro, p)
    f = mix(c * p.m, tau, p)
    return rpm_from_thrusts(f, p)
