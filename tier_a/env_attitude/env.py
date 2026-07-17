"""AttitudeEnv: the RL-facing continuous 6-DOF flight-tracking environment.

Wraps ``env_attitude.dynamics`` (pure physics) and
``env_attitude.reference_trajectory`` (the pre-baked path) into a
``step()``/``reset()`` API shaped like ``env.gridworld.GridWorld``, but
continuous. The agent's observation contains only errors relative to the
reference -- never the raw ``(x, y, z)`` position or the tree map (INVARIANT
2's spirit; see ``tests/test_invariants_attitude.py``'s ``AttitudeBlindEnv``).
"""

from __future__ import annotations

import math

import numpy as np

from tier_a.env_attitude import dynamics
from tier_a.env_attitude.constants import (
    ACTIONS_ATTITUDE,
    ARRIVE_TOL,
    CRUISE_SPEED,
    DT,
    MAX_DEVIATION,
    N_ACTIONS_ATTITUDE,
    N_TREES_ATTITUDE,
    ONLINE_TIME_MARGIN,
    PHI_MAX,
    PHI_MAX_PIROUETTE,
    R_COLLISION,
    R_DEPARTED,
    R_GOAL,
    R_STEP,
    TAU_WIND_MAX,
    TIME_MARGIN,
)
from tier_a.env_attitude.lidar2d import scan2d
from tier_a.env_attitude.lidar3d import (
    LIDAR_RANGE_3D,
    N_CIRCLES_DENSE,
    N_RAYS_3D,
    RAYS_PER_CIRCLE_DENSE,
    scan3d,
    scan3d_dense,
)
from tier_a.env_attitude.mapping import BeliefMap
from tier_a.env_attitude.occupancy import collides, collides_oriented
from tier_a.env_attitude.online_planner import OnlinePlanner, PlannerConfig
from tier_a.env_attitude.reference_trajectory import DS, V_MIN_FRAC, _trajectory_from_waypoints, build_reference

OBS_DIM = 10


def _wrap(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


class AttitudeEnv:
    """6-DOF attitude-tracking env over a fixed, pre-baked reference trajectory.

    Deterministic by default (``wind_prob=0.0``) -- mirrors ``env.GridWorld``
    being deterministic while ``env_wind.GridWorldWind`` is the explicit,
    opt-in harder variant. Pass ``wind_prob=env_attitude.constants.WIND_PROB``
    (or any nonzero value) to add stochastic gust torques back in once the
    deterministic case trains reliably.

    ``body_aware_collision=False`` by default, matching the same pattern:
    every existing Tier-A result was measured against the original point-mass
    collision test (``env_attitude.occupancy.collides``). Passing ``True``
    switches to ``collides_oriented`` -- the drone becomes a rollable
    ellipsoid whose presented width shrinks as it banks toward
    ``PHI_MAX_PIROUETTE`` (85 deg) instead of the cruise ``PHI_MAX`` (60 deg)
    limit, a deliberately different and riskier operating envelope for
    threading a gap narrower than the level-flight footprint.
    """

    n_actions = N_ACTIONS_ATTITUDE
    obs_dim = OBS_DIM

    def __init__(self, n_trees: int = N_TREES_ATTITUDE, seed: int = 0,
                 dt: float = DT, wind_prob: float = 0.0,
                 max_time: float | None = None,
                 custom_trees: np.ndarray | None = None,
                 custom_reference=None, v0: float = 0.0,
                 online_lidar_planning: bool = False,
                 planner_config: PlannerConfig | None = None,
                 body_aware_collision: bool = False) -> None:
        self.seed = seed
        self.n_trees = n_trees
        self.dt = dt
        self.wind_prob = wind_prob
        self.v0 = v0
        self.online_lidar_planning = online_lidar_planning
        self._planner_config = planner_config
        self.body_aware_collision = body_aware_collision

        if custom_reference is not None:
            base_ref = custom_reference
        else:
            base_ref = build_reference(n_trees, seed, trees=custom_trees)
        if base_ref is None:
            raise ValueError(
                f"seed {seed} with {n_trees} trees is infeasible -- filter with "
                "oracle.astar.is_feasible before constructing AttitudeEnv"
            )
        self._trees = base_ref.trees
        self._start_xy = tuple(base_ref.xy[0])
        self._start_psi0 = float(base_ref.psi_ref[0])
        self._goal_xy = tuple(base_ref.waypoints_xy[-1])

        if online_lidar_planning:
            ref = self._plan_initial_ref()
            self.max_time = max_time if max_time is not None else ONLINE_TIME_MARGIN * ref.T_total
        else:
            ref = base_ref
            self.max_time = max_time if max_time is not None else TIME_MARGIN * ref.T_total
        self._ref = ref

        self._rng = np.random.default_rng(seed)
        self._state = dynamics.make_initial_state(tuple(ref.xy[0]), ref.z_ref,
                                                    psi0=float(ref.psi_ref[0]), v0=v0)
        self._t = 0.0
        self._t_wall = 0.0

    def _plan_initial_ref(self):
        """Build the very first reference from a fully-UNKNOWN belief -- since
        ``BeliefMap.planning_grid()`` treats unknown cells as passable, this
        trivially yields the straight line from the start to the goal cell,
        giving "fly straight until LIDAR proves otherwise" for free."""
        self._belief = BeliefMap()
        self._planner = OnlinePlanner(goal_xy=self._goal_xy, config=self._planner_config)
        waypoints = self._planner.maybe_replan(self._belief, self._start_xy, self._start_psi0)
        if waypoints is None:
            waypoints = np.array([self._start_xy, self._goal_xy])
        return _trajectory_from_waypoints(waypoints, self._trees, self.seed, self.n_trees,
                                          CRUISE_SPEED, DS, v_min_frac=V_MIN_FRAC)

    # -- read-only physical accessors: viz/tests only, never the agent's obs --
    @property
    def xyz(self) -> np.ndarray:
        return self._state[0:3].copy()

    @property
    def attitude(self) -> tuple[float, float, float]:
        return float(self._state[6]), float(self._state[7]), float(self._state[8])

    @property
    def trees(self) -> np.ndarray:
        return self._ref.trees

    @property
    def reference(self):
        return self._ref

    def lidar_scan(self, n_rays: int = N_RAYS_3D, max_range: float = LIDAR_RANGE_3D) -> np.ndarray:
        """Body-frame-rotating 3D LIDAR reading at the current true state.

        This is a *sensor* read, not map access: it costs the caller a scan
        and returns only range, the same trust boundary the base kit already
        grants Tier 3's 2D LIDAR (``tier_d/env/lidar.py``). Used by
        ``agent.qtable_attitude``'s discretized state; the continuous DQN
        variant (``agent/qnet_attitude.py``) never calls this.
        """
        phi, theta, psi = self.attitude
        return scan3d(self.xyz, phi, theta, psi, self._ref.trees, n_rays=n_rays, max_range=max_range)

    def lidar_scan_dense(self, n_circles: int = N_CIRCLES_DENSE,
                        rays_per_circle: int = RAYS_PER_CIRCLE_DENSE,
                        max_range: float = LIDAR_RANGE_3D) -> np.ndarray:
        """Dense multi-great-circle 3D LIDAR reading (64 rays by default) at
        the current true state -- same sensor trust boundary as
        ``lidar_scan()`` (a range return, never the tree coordinates
        themselves), just a denser ray set. Used by
        ``agent.pirouette_attitude``.
        """
        phi, theta, psi = self.attitude
        return scan3d_dense(self.xyz, phi, theta, psi, self._ref.trees,
                            n_circles=n_circles, rays_per_circle=rays_per_circle, max_range=max_range)

    # -- observation --------------------------------------------------------
    def _obs(self) -> np.ndarray:
        x, y, z = self._state[0:3]
        vz = self._state[5]
        phi, theta, psi = self._state[6:9]
        p, q, r = self._state[9:12]

        ref_now = self._ref.at(self._t)
        _, e_lat, _ = self._ref.nearest((x, y))
        e_alt = z - self._ref.z_ref
        e_vz = vz  # reference vertical speed is 0 at constant altitude

        return np.array([
            e_lat, e_alt, e_vz,
            _wrap(phi - ref_now["phi"]), _wrap(theta - ref_now["theta"]), _wrap(psi - ref_now["psi"]),
            p, q, r,
            ref_now["kappa"],
        ])

    def reset(self) -> np.ndarray:
        if self.online_lidar_planning:
            self._ref = self._plan_initial_ref()
        self._state = dynamics.make_initial_state(tuple(self._ref.xy[0]), self._ref.z_ref,
                                                    psi0=float(self._ref.psi_ref[0]), v0=self.v0)
        self._t = 0.0
        self._t_wall = 0.0
        return self._obs()

    def step(self, a: int) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one control tick. Returns (obs, reward, done, info).

        ``done`` covers real terminals (collision/departed/loss_of_control/goal);
        ``info["truncated"]`` marks the timeout, which bootstraps rather than
        zeroing the potential -- do not conflate the two (see CLAUDE.md).
        """
        tau = np.array(ACTIONS_ATTITUDE[a], dtype=float)
        wind = bool(self._rng.random() < self.wind_prob)
        if wind:
            tau = tau + self._rng.uniform(-TAU_WIND_MAX, TAU_WIND_MAX, size=3)

        self._state = dynamics.step(self._state, tuple(tau), self.dt)
        self._t += self.dt
        self._t_wall += self.dt

        replanned = False
        if self.online_lidar_planning:
            xy_now = (float(self._state[0]), float(self._state[1]))
            psi_now = float(self._state[8])
            d = scan2d(xy_now, psi_now, self._trees)
            self._belief.update_from_scan(xy_now, d, psi_now)
            new_waypoints = self._planner.maybe_replan(self._belief, xy_now, psi_now)
            if new_waypoints is not None:
                self._ref = _trajectory_from_waypoints(
                    new_waypoints, self._trees, self.seed, self.n_trees,
                    CRUISE_SPEED, DS, v_min_frac=1.0,  # already cruising -- no relaunch ramp
                )
                self._t = 0.0
                replanned = True

        x, y, z = self._state[0:3]
        phi, theta, _ = self._state[6:9]
        _, e_lat, _ = self._ref.nearest((x, y))
        e_alt = float(z - self._ref.z_ref)

        if self.body_aware_collision:
            collision = collides_oriented((x, y), z, phi, self._ref.trees)
            phi_limit = PHI_MAX_PIROUETTE  # deliberately different envelope -- see class docstring
        else:
            collision = collides((x, y), z, self._ref.trees)
            phi_limit = PHI_MAX
        departed = abs(e_lat) > MAX_DEVIATION or abs(e_alt) > MAX_DEVIATION
        loss_of_control = abs(phi) > phi_limit or abs(theta) > PHI_MAX
        dist_to_final = float(np.hypot(
            x - self._ref.waypoints_xy[-1, 0], y - self._ref.waypoints_xy[-1, 1]
        ))
        goal = (not (collision or departed or loss_of_control)) and dist_to_final < ARRIVE_TOL

        info = {
            "collision": collision, "departed": departed, "loss_of_control": loss_of_control,
            "goal": goal, "truncated": False, "wind": wind, "e_lat": e_lat, "e_alt": e_alt,
            "replanned": replanned,
        }

        if collision:
            return self._obs(), R_COLLISION, True, info
        if loss_of_control or departed:
            return self._obs(), R_DEPARTED, True, info
        if goal:
            return self._obs(), R_GOAL, True, info

        info["truncated"] = self._t_wall >= self.max_time
        return self._obs(), R_STEP, info["truncated"], info
