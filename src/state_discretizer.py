# state_discretizer.py
from __future__ import annotations
import numpy as np

def get_energy_level(soc: float, available: bool) -> int:
    """
    Clasifica el estado energético del satélite en 4 niveles.
    0: Bloqueado por seguridad (corte de batería).
    1: Riesgo / Batería baja (SoC <= 40%).
    2: Batería media (40% < SoC <= 70%).
    3: Batería alta (SoC > 70%).
    """
    if not available:
        return 0
    if soc <= 0.40:
        return 1
    elif soc <= 0.70:
        return 2
    else:
        return 3

def get_solar_level(sunlit: float | bool) -> int:
    """
    Determina la condición solar del satélite.
    0: En eclipse (sin carga).
    1: Iluminado (cargando).
    """
    return 1 if sunlit else 0

def get_congestion_level(num_tasks: int, high_congestion_threshold: int = 3) -> int:
    """
    Clasifica la congestión según la cantidad de tareas visibles.
    0: Sin tareas.
    1: Congestión baja (1 a threshold-1 tareas).
    2: Congestión alta (>= threshold tareas).
    """
    if num_tasks == 0:
        return 0
    if num_tasks < high_congestion_threshold:
        return 1
    return 2

def get_urgency_level(deadlines: np.ndarray, urgency_threshold_s: float = 30.0) -> int:
    """
    Evalúa la urgencia basándose en el menor tiempo restante de las tareas visibles.
    0: Sin tareas visibles.
    1: Sin urgencia (la tarea más próxima tiene > 30s).
    2: Urgencia alta (al menos una tarea vence en <= 30s).
    """
    # Filtrar solo los deadlines válidos (mayores a 0)
    valid_deadlines = deadlines[deadlines > 0]
    
    if len(valid_deadlines) == 0:
        return 0
        
    min_deadline = np.min(valid_deadlines)
    if min_deadline <= urgency_threshold_s:
        return 2
    return 1

def discretize_satellite_state(
    obs: dict, 
    sat_idx: int, 
    congestion_threshold: int, 
    urgency_threshold_s: float
) -> tuple[int, int, int, int]:
    """
    Extrae y discretiza las cuatro variables de estado para un solo satélite.
    Retorna una tupla (energia, solar, congestion, urgencia).
    """
    # 1. Batería
    soc = float(obs["battery_soc"][sat_idx])
    available = bool(obs["available"][sat_idx])
    e_s = get_energy_level(soc, available)
    
    # 2. Sol
    sunlit = bool(obs["sunlit"][sat_idx])
    c_s = get_solar_level(sunlit)
    
    # 3. Tareas y Deadlines
    tasks_array = obs["task_present"][sat_idx]
    num_tasks = int(np.sum(tasks_array > 0))
    
    deadlines_array = obs["task_time_left_s"][sat_idx]
    
    cong_s = get_congestion_level(num_tasks, congestion_threshold)
    urg_s = get_urgency_level(deadlines_array, urgency_threshold_s)
    
    return (e_s, c_s, cong_s, urg_s)

def discretize_observation(
    obs: dict, 
    congestion_threshold: int = 3, 
    urgency_threshold_s: float = 30.0
) -> tuple:
    """
    Punto de entrada principal. Convierte el diccionario de observación de meteogym
    en una tupla de tuplas inmutable, lista para ser usada como llave en la tabla Q.
    
    Ejemplo de salida para 2 satélites:
    ((2, 1, 1, 1), (3, 0, 2, 2))
    """
    num_satellites = len(obs["battery_soc"])
    global_state = []
    
    for sat_idx in range(num_satellites):
        sat_tuple = discretize_satellite_state(
            obs=obs, 
            sat_idx=sat_idx, 
            congestion_threshold=congestion_threshold,
            urgency_threshold_s=urgency_threshold_s
        )
        global_state.append(sat_tuple)
        
    return tuple(global_state)