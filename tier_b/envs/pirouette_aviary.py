"""PirouetteAviary — підклас BaseRLAviary із CTBR-дією, 34-променевим LIDAR,
аналітичною еліпсоїд-колізією, воротами-курикулумом і shaping-винагородою.

Батьківському конструктору передаємо act=ActionType.RPM лише щоб не ламати
його сантехніку — _actionSpace/_preprocessAction ПОВНІСТЮ перевизначені
(CTBR у бібліотеці немає; DSLPIDControl не використовуємо взагалі).

Дисципліна terminal-vs-truncated (правило кіта):
  terminated  = колізія (аналітична) або фініш   → Φ(terminal)=0 у shaping
  truncated   = jam / таймаут / вихід за межі    → bootstrap зберігається
"""

from __future__ import annotations

import collections

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

from tier_b import rewards as rw
from tier_b.envs import forest_gen as fg
from tier_b.envs.collision_ellipsoid import SEMI_AXES, check_forest
from tier_b.envs.ctbr import CTBRParams, ctbr_to_rpm
from tier_b.envs.lidar import MAX_RANGE, N_RAYS, TREE_GROUP, scan
from tier_b.envs.trajectory import FINISH_RADIUS, GateTracker, build_waypoints

# дефолти = config.yaml (curriculum / jam / bounds / action.history_len / sim)
GAPS = (1.20, 0.80, 0.55, 0.40, 0.30, 0.26, 0.20, 0.16, 0.12)
JAM_V = 0.05
JAM_STEPS = 96          # 2 с × 48 Гц
STALL_EPS = 0.1         # м — мінімальне покращення залишкової довжини маршруту
STALL_STEPS = 240       # 5 с × 48 Гц (виміряне доповнення: ловить «кружляння»)
EPISODE_MAX_S = 12.0
HISTORY_LEN = 4
BOUNDS_X = (-2.0, 14.0)
BOUNDS_Y = (-4.0, 4.0)
BOUNDS_Z = (0.02, 3.0)

OBS_DIM = 3 + 4 + 3 + 3 + N_RAYS + 4 * HISTORY_LEN  # 63


class PirouetteAviary(BaseRLAviary):

    def __init__(self, level: int = 0, master_seed: int = 0,
                 gui: bool = False, record: bool = False,
                 gaps=GAPS, semi_axes=SEMI_AXES):
        self.level = int(level)
        self.master_seed = int(master_seed)
        self.gaps = tuple(gaps)
        self.semi_axes = tuple(semi_axes)
        self._episode_idx = -1
        self._act_hist = collections.deque(maxlen=HISTORY_LEN)
        self._tree_ids: list[int] = []
        self.forest = None       # створюється у _build_episode перед _addObstacles
        self.tracker = None
        self._build_episode()    # BaseAviary.__init__ викликає _addObstacles

        super().__init__(drone_model=DroneModel.CF2X, num_drones=1,
                         initial_xyzs=np.array([[fg.START_XY[0], fg.START_XY[1],
                                                 fg.CRUISE_Z]]),
                         physics=Physics.PYB, pyb_freq=240, ctrl_freq=48,
                         gui=gui, record=record,
                         obs=ObservationType.KIN, act=ActionType.RPM)

        # константи CF2X — з URDF, не з пам'яті
        self.params = CTBRParams(m=self.M, g=self.G, kf=self.KF, km=self.KM,
                                 arm_l=self.L, max_rpm=self.MAX_RPM,
                                 j_diag=tuple(np.diag(self.J)))
        self.EPISODE_LEN_SEC = EPISODE_MAX_S

    # -- курикулум ---------------------------------------------------------------
    def set_level(self, level: int) -> None:
        """Викликається callback-ом через vec_env.env_method; діє з наступного reset."""
        self.level = int(np.clip(level, 0, len(self.gaps) - 1))

    # -- епізод / ліс -------------------------------------------------------------
    def _build_episode(self) -> None:
        self._episode_idx += 1
        seed = fg.episode_seed(self.master_seed, self._episode_idx)
        self.forest = fg.generate_forest(self.gaps[self.level], seed)
        wps = build_waypoints(self.forest)
        self.tracker = GateTracker(self.forest, wps)
        self._jam_count = 0
        self._stall_count = 0
        self._best_remaining = self.tracker.remaining_length(wps[0])
        self._collided = False
        self._depth = 0.0
        self._finished = False
        self._gate_event = False
        self._out_of_bounds = False
        self._ctrl_ticks = 0     # власний лічильник кроків політики: step_counter
        self._events_step = -1   # батька інкрементується ПІСЛЯ хуків (виміряно)
        self._phi_prog = self.tracker.phi_prog(wps[0])
        self._phi_gate = self.tracker.phi_gate(wps[0])
        self._phi_prog_prev = self._phi_prog
        self._phi_gate_prev = self._phi_gate
        self._act_hist.clear()
        for _ in range(HISTORY_LEN):
            self._act_hist.append(np.zeros(4))

    def reset(self, seed=None, options=None):
        self._build_episode()
        return super().reset(seed=seed, options=options)

    def _addObstacles(self):
        """Циліндри-дерева; група TREE_GROUP, щоб LIDAR бачив ЛИШЕ їх.
        Викликається BaseAviary._housekeeping після p.resetSimulation."""
        self._tree_ids = []
        h = self.forest.tree_height
        for x, y, r in self.forest.trees:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                         physicsClientId=self.CLIENT)
            # візуальна форма потрібна і в DIRECT: getCameraImage (TinyRenderer)
            # рендерить лише visual shapes — без неї відео буде порожнім
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                      rgbaColor=[0.35, 0.25, 0.12, 1],
                                      physicsClientId=self.CLIENT)
            body = p.createMultiBody(0, col, vis, [x, y, h / 2.0],
                                     physicsClientId=self.CLIENT)
            p.setCollisionFilterGroupMask(body, -1, 1 | TREE_GROUP, -1,
                                          physicsClientId=self.CLIENT)
            self._tree_ids.append(body)

        # ВАЖЛИВО (виміряний нюанс pybullet): collisionFilterMask променя фільтрує
        # лише тіла з ЯВНО заданою групою — тіла з групою за замовчуванням промінь
        # влучає завжди. Тому дрон і землю явно ставимо у групу 1 (маска -1
        # зберігає фізичні контакти), і LIDAR бачить самі дерева.
        drone = self.DRONE_IDS[0]
        for link in range(-1, p.getNumJoints(drone, physicsClientId=self.CLIENT)):
            p.setCollisionFilterGroupMask(drone, link, 1, -1,
                                          physicsClientId=self.CLIENT)
        p.setCollisionFilterGroupMask(self.PLANE_ID, -1, 1, -1,
                                      physicsClientId=self.CLIENT)

    # -- CTBR --------------------------------------------------------------------
    def _actionSpace(self):
        # буфер дій батька (він його створює у своєму _actionSpace) не потрібен,
        # але підтримуємо сумісність: власна історія — у self._act_hist
        return spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

    def _preprocessAction(self, action):
        self._ctrl_ticks += 1  # рівно раз на ctrl-крок, ДО фізики та хуків
        a = np.asarray(action, dtype=np.float64).reshape(-1)[:4]
        self._act_hist.append(a.copy())
        rot = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        omega_body = rot.T @ self.ang_v[0]
        rpm = ctbr_to_rpm(a, omega_body, self.params)
        return rpm.reshape(1, 4)

    # -- спостереження -------------------------------------------------------------
    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def _computeObs(self):
        self._refresh_events()
        pos = self.pos[0]
        rot = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        wp = self.tracker.waypoints[min(self.tracker.next_wp,
                                        len(self.tracker.waypoints) - 1)]
        wp_body = np.clip(rot.T @ (wp - pos) / 5.0, -1.0, 1.0)
        vel_body = rot.T @ self.vel[0] / 5.0
        omega = rot.T @ self.ang_v[0] / 10.0
        lidar = scan(self.CLIENT, pos, self.quat[0], MAX_RANGE)
        hist = np.concatenate(list(self._act_hist))
        obs = np.concatenate([wp_body, self.quat[0], vel_body, omega, lidar, hist])
        return obs.astype(np.float32)

    # -- події кроку (рівно раз на ctrl-крок) ---------------------------------------
    def _refresh_events(self):
        if self._events_step == self._ctrl_ticks:
            return
        first_call = self._ctrl_ticks == 0  # _computeObs під час reset
        self._events_step = self._ctrl_ticks

        pos = self.pos[0]
        rot = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)

        self._gate_event = False
        if not first_call:
            self._collided, self._depth = check_forest(
                pos, rot, self.forest.trees, self.semi_axes,
                self.forest.tree_height)
            ev = self.tracker.update(pos)
            self._gate_event = ev["gate_passed"]
            if ev["finished"]:
                self._finished = True

            # jam: ‖v‖ < 0.05 м/с безперервно 2 с поза зоною фінішу
            near_finish = np.linalg.norm(pos - self.tracker.waypoints[-1]) < FINISH_RADIUS
            if np.linalg.norm(self.vel[0]) < JAM_V and not near_finish:
                self._jam_count += 1
            else:
                self._jam_count = 0

            # progress-stall: «кружляння» зі швидкістю > v_thresh velocity-jam
            # не ловить (виміряно на probe 300k: 532-крокові епізоди, 0 воріт)
            remaining = self.tracker.remaining_length(pos)
            if remaining < self._best_remaining - STALL_EPS:
                self._best_remaining = remaining
                self._stall_count = 0
            elif not near_finish:
                self._stall_count += 1

            self._out_of_bounds = not (
                BOUNDS_X[0] <= pos[0] <= BOUNDS_X[1]
                and BOUNDS_Y[0] <= pos[1] <= BOUNDS_Y[1]
                and BOUNDS_Z[0] <= pos[2] <= BOUNDS_Z[1])

        self._phi_prog_prev = self._phi_prog
        self._phi_gate_prev = self._phi_gate
        self._phi_prog = self.tracker.phi_prog(pos)
        self._phi_gate = self.tracker.phi_gate(pos)

    # -- reward / термінали ----------------------------------------------------------
    def _computeReward(self):
        self._refresh_events()
        terminal = self._collided or self._finished
        rot = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        omega_body = rot.T @ self.ang_v[0]
        return rw.step_reward(
            phi_prog_new=self._phi_prog, phi_prog_old=self._phi_prog_prev,
            phi_gate_new=self._phi_gate, phi_gate_old=self._phi_gate_prev,
            z=float(self.pos[0][2]), omega=omega_body, terminal=terminal,
            collided=self._collided, depth=self._depth,
            gate_passed=self._gate_event, finished=self._finished and terminal)

    def _computeTerminated(self):
        self._refresh_events()
        return bool(self._collided or self._finished)

    def _computeTruncated(self):
        self._refresh_events()
        timeout = self._ctrl_ticks / self.CTRL_FREQ >= EPISODE_MAX_S
        return bool(self._jam_count >= JAM_STEPS
                    or self._stall_count >= STALL_STEPS
                    or timeout or self._out_of_bounds)

    def _computeInfo(self):
        return {
            "success": bool(self._finished),
            "is_success": bool(self._finished),
            "collision": bool(self._collided),
            "jam": bool(self._jam_count >= JAM_STEPS
                        or self._stall_count >= STALL_STEPS),
            "out_of_bounds": bool(self._out_of_bounds),
            "gates_passed": int(self.tracker.gates_passed),
            "level": int(self.level),
        }
