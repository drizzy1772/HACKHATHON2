"""Tier-B "CTBR Pirouette" — 6DOF RL-стек на gym-pybullet-drones.

ВАЖЛИВО (правило репозиторію): жодних top-level імпортів torch /
gym_pybullet_drones / pybullet тут — ядро хакатон-кіта не має цих залежностей,
і корневий pytest мусить збиратися без conda-середовища `drones`.
Усі важкі імпорти — ліниві, всередині функцій.
"""

from __future__ import annotations

import pathlib

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def load_config(path=CONFIG_PATH) -> dict:
    """Читає config.yaml. yaml імпортується ліниво: у головному venv його нема,
    і це нормально — чисті numpy-модулі несуть власні дефолти."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
