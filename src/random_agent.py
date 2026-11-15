# random_agent.py
from __future__ import annotations
import numpy as np


class RandomAgent:
    """Agente de referencia que elige acciones binarias con distribución uniforme."""

    def __init__(self, num_satellites: int, seed: int | None = None) -> None:
        self.num_satellites = num_satellites
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: dict | None = None) -> np.ndarray:
        """Retorna un vector MultiBinary(S) aleatorio uniforme."""
        return self.rng.integers(0, 2, size=self.num_satellites, dtype=np.int8)