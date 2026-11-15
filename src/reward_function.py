# reward_function.py
from __future__ import annotations
import numpy as np


def compute_custom_reward(
    obs: dict,
    info: dict,
    alpha: float = 1.1,
    beta: float = 1.1,
    gamma: float = 1.3,
    delta: float = 0.4,
    epsilon: float = 0.7,
) -> tuple[float, dict]:
    """
    Calcula R_t = alpha*N_exitosas - beta*N_fallidas_tierra - gamma*N_fallidas_satelite
                  - delta*N_descargas - epsilon*N_ocioso
    Retorna la recompensa escalar y un diccionario con el desglose de conteos.
    """
    events = info["task_events"]

    n_exitosas = sum(1 for e in events if e["status"] == "completed")
    n_fallidas_tierra = sum(
        1 for e in events if e["status"] != "completed" and e["satellite_id"] is None
    )
    n_fallidas_sat = sum(
        1 for e in events if e["status"] != "completed" and e["satellite_id"] is not None
    )

    # Satélites descargándose en el intervalo actual (corriente > 0 indica descarga en battery.py)
    n_descargas = int(np.sum(obs["battery_current_a"] > 0))

    # Satélites encendidos efectivamente pero sin gastar energía en tareas (encendidos en vano)
    effective_active = info["effective_active"].astype(bool)
    working = info["task_energy_j"] > 0
    n_ocioso = int(np.sum(effective_active & ~working))

    custom_r = (
        alpha * n_exitosas
        - beta * n_fallidas_tierra
        - gamma * n_fallidas_sat
        - delta * n_descargas
        - epsilon * n_ocioso
    )

    breakdown = {
        "n_exitosas": n_exitosas,
        "n_fallidas_tierra": n_fallidas_tierra,
        "n_fallidas_sat": n_fallidas_sat,
        "n_descargas": n_descargas,
        "n_ocioso": n_ocioso,
    }
    return float(custom_r), breakdown