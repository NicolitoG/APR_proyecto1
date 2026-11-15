# state_discretizer.py
from __future__ import annotations
import numpy as np


def discretize_observation(obs: dict, urgent_threshold_s: float = 30.0) -> tuple:
    """Convierte el diccionario obs de MeteoGym en una tupla discreta por satélite."""
    num_sats = len(obs["satellite_ids"])
    state_per_sat = []

    for s in range(num_sats):
        # 1. Energía: combina bloqueo (available) y nivel de batería (battery_soc)
        if not obs["available"][s]:
            energy_state = 0
        else:
            soc = obs["battery_soc"][s]
            if soc <= 0.40:
                energy_state = 1
            elif soc <= 0.70:
                energy_state = 2
            else:
                energy_state = 3

        # 2. Sol: 0 en eclipse, 1 iluminado
        sun_state = int(obs["sunlit"][s])

        # 3. Carga de trabajo: filtra solo usuarios visibles con tarea activa
        task_mask = obs["task_present"][s].astype(bool)
        num_tasks = int(np.sum(task_mask))

        if num_tasks == 0:
            workload_state = 0
        else:
            min_time_left = float(np.min(obs["task_time_left_s"][s][task_mask]))
            if min_time_left <= urgent_threshold_s:
                workload_state = 2  # Urgente
            elif num_tasks <= 2:
                workload_state = 1  # Carga baja holgada
            else:
                workload_state = 3  # Alta congestión holgada

        state_per_sat.append((energy_state, sun_state, workload_state))

    return tuple(state_per_sat)