# -*- coding: utf-8 -*-
"""
КОНТРАКТ стану дрона для автономного режиму (ХАКАТОН: сам крок керування —
step_autopilot() — реалізують учасники в solution.py, не тут).

AutopilotState — ЄДИНА форма стану, яку читають рендер (blender_manual.py
виставляє позу дрона з x/y/z/yaw/pitch/roll) і колізії (collision_and_bounds_
status у sim_headless.py); учасники повертають НОВИЙ AutopilotState із тими
самими полями з кожного виклику step_autopilot(), значення — вільні."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutopilotState:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0    # ВІЗУАЛЬНИЙ (не обов'язково впливає на рух)
    roll: float = 0.0     # ВІЗУАЛЬНИЙ (не обов'язково впливає на рух)
