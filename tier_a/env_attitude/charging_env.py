"""ChargingStationEnv — двофазна обгортка над AttitudeEnv.

Реалізує state-machine польоту:

    FLY_TO_CHARGER  →  дрон летить по ref1 (START → CHARGER_XY)
         ↓ (досяг точки зарядки)
    CHARGING        →  дрон hovering на місці CHARGE_STEPS кроків
         ↓ (зарядка закінчена)
    FLY_TO_GOAL     →  дрон летить по ref2 (CHARGER_XY → GOAL_XY)
         ↓ (досяг цілі / термінал)
    DONE

Дрон ЗУПИНЯЄТЬСЯ на зарядній точці (не заряджається в процесі польоту):
CHARGING-фаза виконує нульові torque-дії (hover) поки лічильник не вичерпається.

Приклад використання
--------------------
    from tier_a.env_attitude.charging_env import ChargingStationEnv

    env = ChargingStationEnv(charger_xy=(5.0, 5.0), seed=0, charge_steps=50)
    obs = env.reset()
    done = False
    while not done:
        a = 13  # neutral hover action
        obs, reward, done, info = env.step(a)
        print(env.phase, info)
"""

from __future__ import annotations

import enum
import math

import numpy as np

from tier_a.env_attitude.constants import (
    ARRIVE_TOL,
    CRUISE_SPEED,
    N_TREES_ATTITUDE,
)
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.reference_trajectory import (
    DS,
    V_MIN_FRAC,
    build_local_reference,
    build_reference,
)

# --- Константи зарядки -------------------------------------------------------
CHARGE_STEPS: int = 50          # кроки hovering (50 × 0.02 s ≈ 1 секунда зарядки)
CHARGER_ARRIVE_TOL: float = 0.5  # м — допуск «досяг зарядної станції»
NEUTRAL_ACTION: int = 13         # (0, 0, 0) — нульові моменти, індекс у 27-action сітці


class Phase(enum.Enum):
    """Поточна фаза місії."""
    FLY_TO_CHARGER = "fly_to_charger"
    CHARGING       = "charging"
    FLY_TO_GOAL    = "fly_to_goal"
    DONE           = "done"


class ChargingStationEnv:
    """Двофазний польот з зупинкою на зарядній станції.

    Parameters
    ----------
    charger_xy : tuple[float, float]
        Координати зарядної станції у метрах.
        За замовчуванням — середина між START_XY і GOAL_XY.
    charge_steps : int
        Кількість кроків hovering на зарядній станції.
    seed : int
        Seed для побудови лісу та AttitudeEnv.
    n_trees : int
        Кількість дерев у лісі (передається в AttitudeEnv).
    kwargs :
        Додаткові параметри (wind_prob, body_aware_collision тощо),
        прокидаються в обидва внутрішніх AttitudeEnv.
    """

    def __init__(
        self,
        charger_xy: tuple[float, float] | None = None,
        charge_steps: int = CHARGE_STEPS,
        seed: int = 0,
        n_trees: int = N_TREES_ATTITUDE,
        **kwargs,
    ) -> None:
        self.seed = seed
        self.n_trees = n_trees
        self.charge_steps = charge_steps
        self._extra_kwargs = kwargs

        # Побудуємо базовий env щоб отримати дерева і START/GOAL
        _base_env = AttitudeEnv(n_trees=n_trees, seed=seed, **kwargs)
        self._trees = _base_env.trees
        self._start_xy: tuple[float, float] = _base_env._start_xy
        self._goal_xy:  tuple[float, float] = _base_env._goal_xy

        # Визначити координати зарядної станції
        if charger_xy is None:
            sx, sy = self._start_xy
            gx, gy = self._goal_xy
            charger_xy = ((sx + gx) / 2.0, (sy + gy) / 2.0)
        self.charger_xy: tuple[float, float] = charger_xy

        # Публічний інтерфейс (n_actions, obs_dim — ті самі, що в AttitudeEnv)
        self.n_actions = _base_env.n_actions
        self.obs_dim   = _base_env.obs_dim

        # Внутрішній стан (буде ініційований у reset())
        self._env1: AttitudeEnv | None = None   # START → CHARGER
        self._env2: AttitudeEnv | None = None   # CHARGER → GOAL
        self._active_env: AttitudeEnv | None = None
        self._phase: Phase = Phase.FLY_TO_CHARGER
        self._charge_counter: int = 0

        # Для агрегації метрик (tracking RMSE по фазах)
        self._sq_phase1: list[float] = []
        self._sq_phase2: list[float] = []

    # -- read-only accessors ---------------------------------------------------
    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def trees(self) -> np.ndarray:
        return self._trees

    @property
    def _t_wall(self) -> float:
        """Сумарний час польоту (для сумісності з greedy_rollout_attitude)."""
        t = 0.0
        if self._env1 is not None:
            t += self._env1._t_wall
        if self._env2 is not None:
            t += self._env2._t_wall
        return t

    # -- побудова сегментних env-ів -------------------------------------------
    def _build_env1(self) -> AttitudeEnv:
        """AttitudeEnv для сегменту START → CHARGER_XY."""
        ref1 = build_local_reference(
            self._start_xy,
            self.charger_xy,
            self._trees,
            seed=self.seed,
            speed=CRUISE_SPEED,
            ds=DS,
            v_min_frac=V_MIN_FRAC,
        )
        if ref1 is None:
            # Якщо пряма лінія перетинає дерево — використаємо базовий ref
            # (будуємо «глобальний» env і беремо з нього reference лише до половини)
            raise ValueError(
                f"Segment START{self._start_xy}→CHARGER{self.charger_xy} перетинає дерево. "
                "Виберіть іншу точку зарядки або збільшіть clearance."
            )
        return AttitudeEnv(
            n_trees=self.n_trees,
            seed=self.seed,
            custom_reference=ref1,
            v0=0.0,
            **self._extra_kwargs,
        )

    def _build_env2(self) -> AttitudeEnv:
        """AttitudeEnv для сегменту CHARGER_XY → GOAL_XY."""
        ref2 = build_local_reference(
            self.charger_xy,
            self._goal_xy,
            self._trees,
            seed=self.seed,
            speed=CRUISE_SPEED,
            ds=DS,
            v_min_frac=V_MIN_FRAC,
        )
        if ref2 is None:
            raise ValueError(
                f"Segment CHARGER{self.charger_xy}→GOAL{self._goal_xy} перетинає дерево. "
                "Виберіть іншу точку зарядки або збільшіть clearance."
            )
        return AttitudeEnv(
            n_trees=self.n_trees,
            seed=self.seed,
            custom_reference=ref2,
            v0=0.0,
            **self._extra_kwargs,
        )

    # -- gym-like API ----------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Скидає місію до початкового стану (фаза FLY_TO_CHARGER)."""
        self._env1 = self._build_env1()
        self._env2 = None
        self._active_env = self._env1
        self._phase = Phase.FLY_TO_CHARGER
        self._charge_counter = 0
        self._sq_phase1 = []
        self._sq_phase2 = []
        return self._env1.reset()

    def step(self, a: int) -> tuple[np.ndarray, float, bool, dict]:
        """Виконати один крок місії.

        Повертає (obs, reward, done, info) аналогічно AttitudeEnv.step().
        info містить додаткові ключі:
            phase          — поточна Phase
            charge_remain  — скільки кроків зарядки залишилось
            charger_reached — True якщо тільки що досягнута точка зарядки
        """
        if self._phase == Phase.FLY_TO_CHARGER:
            return self._step_fly_to_charger(a)
        elif self._phase == Phase.CHARGING:
            return self._step_charging()
        elif self._phase == Phase.FLY_TO_GOAL:
            return self._step_fly_to_goal(a)
        else:
            # DONE — повертаємо останній obs без змін
            obs = self._active_env._obs()
            return obs, 0.0, True, {
                "phase": self._phase,
                "goal": False, "collision": False, "departed": False,
                "truncated": False, "charge_remain": 0, "charger_reached": False,
            }

    def _step_fly_to_charger(self, a: int) -> tuple[np.ndarray, float, bool, dict]:
        obs2, r, done, info = self._env1.step(a)
        self._sq_phase1.append(float(obs2[0] ** 2 + obs2[1] ** 2))

        # Перевіряємо відстань до зарядної станції
        xyz = self._env1.xyz
        dist_to_charger = float(math.hypot(
            xyz[0] - self.charger_xy[0],
            xyz[1] - self.charger_xy[1],
        ))
        charger_reached = dist_to_charger < CHARGER_ARRIVE_TOL

        info["phase"] = self._phase
        info["charge_remain"] = self.charge_steps
        info["charger_reached"] = charger_reached

        if info.get("collision") or info.get("departed") or info.get("loss_of_control"):
            # Термінал: зіткнення або вихід за межі під час льоту до зарядки
            done = True
            self._phase = Phase.DONE
            return obs2, r, done, info

        if charger_reached or done:
            # Досягли зарядної точки — переходимо до зарядки
            self._phase = Phase.CHARGING
            self._charge_counter = 0
            info["charger_reached"] = True
            # Не вважаємо "done" — ще є фаза зарядки та фаза до цілі
            return obs2, r, False, info

        return obs2, r, False, info

    def _step_charging(self) -> tuple[np.ndarray, float, bool, dict]:
        """Виконати один крок зарядки (hover — нульові моменти)."""
        # Виконуємо нульову дію через env1 (дрон стоїть)
        obs2, r, done, info = self._env1.step(NEUTRAL_ACTION)
        self._charge_counter += 1
        remain = max(0, self.charge_steps - self._charge_counter)

        info["phase"] = self._phase
        info["charge_remain"] = remain
        info["charger_reached"] = False

        if self._charge_counter >= self.charge_steps:
            # Зарядка закінчена — будуємо env2 і переходимо до фази FLY_TO_GOAL
            self._env2 = self._build_env2()
            self._active_env = self._env2
            self._phase = Phase.FLY_TO_GOAL
            obs_start2 = self._env2.reset()
            info["phase"] = Phase.FLY_TO_GOAL
            info["charge_remain"] = 0
            return obs_start2, 0.0, False, info

        # Ще заряджається — повертаємо поточний obs без "done"
        return obs2, 0.0, False, info

    def _step_fly_to_goal(self, a: int) -> tuple[np.ndarray, float, bool, dict]:
        obs2, r, done, info = self._env2.step(a)
        self._sq_phase2.append(float(obs2[0] ** 2 + obs2[1] ** 2))

        info["phase"] = self._phase
        info["charge_remain"] = 0
        info["charger_reached"] = False

        if done:
            self._phase = Phase.DONE
        return obs2, r, done, info

    # -- метрики по фазах -----------------------------------------------------
    def phase_metrics(self) -> dict:
        """Агреговані метрики після завершення місії."""
        rmse1 = float(np.sqrt(np.mean(self._sq_phase1))) if self._sq_phase1 else float("nan")
        rmse2 = float(np.sqrt(np.mean(self._sq_phase2))) if self._sq_phase2 else float("nan")
        return {
            "tracking_rmse_phase1": rmse1,
            "tracking_rmse_phase2": rmse2,
            "tracking_rmse_total": float(np.sqrt(np.mean(
                self._sq_phase1 + self._sq_phase2
            ))) if (self._sq_phase1 or self._sq_phase2) else float("nan"),
            "charge_steps_done": self._charge_counter,
        }
