"""Tier-A: flight attitude control.

A separate, self-contained track. The 2D forest and its A* oracle path are
reused as-is (via ``oracle.astar`` and ``env.forest``/``env.constants``); the
new problem is flying a 6-DOF rigid body along that fixed path by controlling
roll/pitch/yaw torques under a stochastic wind disturbance. Never imported by
the core 2D kit, and never imports ``torch`` itself (see ``agent/qnet_attitude.py``
for the only place torch is allowed to appear).
"""

from __future__ import annotations
