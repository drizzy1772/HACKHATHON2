"""Simplified 6-DOF rigid-body dynamics for a hover-thrust quadrotor-like body.

Pure integration: no RNG, no environment/trajectory dependency, no I/O. Thrust
is fixed at hover magnitude (``T = mass*g``) rather than being a control
input, so the only commanded quantities are the three body-axis torques. This
bounds the learning problem to attitude-tracking + altitude compensation
(tilting away from level reduces the *vertical* component of thrust, which is
exactly the coupling that makes attitude control matter for position), rather
than full 3D trajectory optimisation -- see the plan's time-boxing section.

State layout (12D, world frame position/velocity + Euler attitude + body rates):
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

Wind is injected by the caller (``env_attitude.env.AttitudeEnv``) by adding a
disturbance torque to the commanded torque before calling ``step`` -- this
module only ever sees the combined torque.
"""

from __future__ import annotations

import math

import numpy as np

from tier_a.env_attitude.constants import DRAG_COEFF, GRAVITY, INERTIA, MASS

STATE_DIM = 12


def hover_thrust(mass: float = MASS, g: float = GRAVITY) -> float:
    return mass * g


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body -> world rotation, ``R = Rz(psi) @ Ry(theta) @ Rx(phi)`` (ZYX Euler)."""
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cpsi, spsi = math.cos(psi), math.sin(psi)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
    ry = np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
    rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def euler_rates(phi: float, theta: float, p: float, q: float, r: float) -> tuple[float, float, float]:
    """Standard ZYX Euler-angle kinematics from body rates.

    Singular at ``theta = +-90deg`` -- out of scope: this track's reference
    trajectory keeps pitch well inside that range (see reference_trajectory.py).
    """
    tth = math.tan(theta)
    cphi, sphi = math.cos(phi), math.sin(phi)
    phidot = p + sphi * tth * q + cphi * tth * r
    thetadot = cphi * q - sphi * r
    psidot = (sphi / math.cos(theta)) * q + (cphi / math.cos(theta)) * r
    return phidot, thetadot, psidot


def step(
    state: np.ndarray,
    torque: tuple[float, float, float],
    dt: float,
    mass: float = MASS,
    inertia: tuple[float, float, float] = INERTIA,
    g: float = GRAVITY,
    c_d: float = DRAG_COEFF,
) -> np.ndarray:
    """One semi-implicit Euler integration tick.

    Rotational dynamics are integrated first (Euler's rigid-body equations,
    ``tau = I*alpha``), then translational dynamics use the *updated* attitude
    to rotate the fixed hover thrust into the world frame -- this is the
    "semi-implicit" part: it is what couples a torque command to an altitude
    change within the same tick, rather than lagging by one step.
    """
    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = (float(v) for v in state)
    Ixx, Iyy, Izz = inertia
    tau_phi, tau_theta, tau_psi = torque

    pdot = (tau_phi - (Izz - Iyy) * q * r) / Ixx
    qdot = (tau_theta - (Ixx - Izz) * p * r) / Iyy
    rdot = (tau_psi - (Iyy - Ixx) * p * q) / Izz

    p2, q2, r2 = p + pdot * dt, q + qdot * dt, r + rdot * dt
    phidot, thetadot, psidot = euler_rates(phi, theta, p2, q2, r2)
    phi2 = phi + phidot * dt
    theta2 = theta + thetadot * dt
    psi2 = psi + psidot * dt

    thrust_mag = mass * g  # fixed at hover -- not a control input
    R = rotation_matrix(phi2, theta2, psi2)
    thrust_world = (thrust_mag / mass) * (R @ np.array([0.0, 0.0, 1.0]))
    v = np.array([vx, vy, vz])
    dv = thrust_world - np.array([0.0, 0.0, g]) - (c_d / mass) * v
    v2 = v + dv * dt
    pos2 = np.array([x, y, z]) + v2 * dt

    return np.array([
        pos2[0], pos2[1], pos2[2],
        v2[0], v2[1], v2[2],
        phi2, theta2, psi2,
        p2, q2, r2,
    ])


def make_initial_state(xy0: tuple[float, float], z0: float, psi0: float = 0.0,
                        v0: float = 0.0) -> np.ndarray:
    """``v0`` is a forward (heading-aligned) initial speed -- 0.0 for a
    standing launch (the default track), or e.g. ``CRUISE_SPEED`` for a
    local segment meant to represent the middle of an already-cruising
    flight (see ``env_attitude.scenarios.build_local_gauntlet_env``)."""
    x, y = xy0
    vx, vy = v0 * math.cos(psi0), v0 * math.sin(psi0)
    return np.array([x, y, z0, vx, vy, 0.0, 0.0, 0.0, psi0, 0.0, 0.0, 0.0])
