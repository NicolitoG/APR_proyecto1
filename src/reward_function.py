# reward_function.py
from __future__ import annotations
import numpy as np

def compute_custom_reward(
    obs: dict, 
    info: dict, 
    alpha: float = 1.0, 
    beta: float = 1.0, 
    gamma: float = 1.5, 
    delta: float = 0.5, 
    epsilon: float = 0.2
) -> tuple[float, dict[str, int]]:
    """
    Calcula la recompensa moldeada (reward shaping) para un paso de simulación.
    
    Fórmula:
    R_t = alpha * N_tasks_exitosas 
        - beta * N_tasks_fallidas_no_asignadas 
        - gamma * N_tasks_fallidas_asignadas 
        - delta * N_satelites_descargados 
        - epsilon * N_satelites_ociosos
    """
    
    # 1. Contadores de tareas
    n_tasks_exitosas = 0
    n_tasks_fallidas_no_asignadas = 0
    n_tasks_fallidas_asignadas = 0
    
    # Procesar eventos de tareas en el paso actual
    for event in info.get("task_events", []):
        if event["status"] == "completed":
            n_tasks_exitosas += 1
        elif event["status"] == "failed":
            # Si satellite_id es None, ningún satélite la tomó antes de fallar/expirar
            if event.get("satellite_id") is None:
                n_tasks_fallidas_no_asignadas += 1
            else:
                n_tasks_fallidas_asignadas += 1
                
    # 2. Contadores de estado del satélite
    
    # N_satelites_descargados: 
    # Un satélite se considera descargado si available == False (entró en bloqueo por batería baja)
    available_array = np.array(obs.get("available", []))
    n_satelites_descargados = int(np.sum(~available_array))
    
    # N_satelites_ociosos:
    # Encendidos (effective_active == True) pero sin procesar datos (task_energy_j == 0.0)
    effective_active = np.array(info.get("effective_active", []))
    task_energy = np.array(info.get("task_energy_j", []))
    
    ociosos_mask = (effective_active == True) & (task_energy == 0.0)
    n_satelites_ociosos = int(np.sum(ociosos_mask))
    
    # 3. Cálculo de la recompensa total
    reward = (
        alpha * n_tasks_exitosas
        - beta * n_tasks_fallidas_no_asignadas
        - gamma * n_tasks_fallidas_asignadas
        - delta * n_satelites_descargados
        - epsilon * n_satelites_ociosos
    )
    
    # 4. Desglose para guardado en logs/CSVs
    breakdown = {
        "n_tasks_exitosas": n_tasks_exitosas,
        "n_tasks_fallidas_no_asignadas": n_tasks_fallidas_no_asignadas,
        "n_tasks_fallidas_asignadas": n_tasks_fallidas_asignadas,
        "n_satelites_descargados": n_satelites_descargados,
        "n_satelites_ociosos": n_satelites_ociosos,
    }
    
    return float(reward), breakdown