"""MeteoGym: satellite availability control with Gymnasium."""

from gymnasium.envs.registration import register, registry

from .battery import BatteryConfig
from .config import SimulationConfig
from .env import MeteoGymEnv

if "MeteoGym-v0" not in registry:
    register(id="MeteoGym-v0", entry_point="meteogym.env:MeteoGymEnv")

__all__ = ["BatteryConfig", "MeteoGymEnv", "SimulationConfig"]
