# estimate_congestion.py
from __future__ import annotations
import random
import gymnasium as gym
import numpy as np

from meteogym import SimulationConfig
from random_agent import RandomAgent

def run_congestion_estimation(num_episodes: int = 3) -> None:
    num_satellites = 10
    random_satellite_ids = tuple(random.sample(range(1, 1081), num_satellites))
    
    print(f"Iniciando estimación de congestión para {num_episodes} episodios...")
    config = SimulationConfig(
        satellite_ids=random_satellite_ids,
        num_users=100,
        duration_s=5400.0,
        step_s=10.0,
    )
    env = gym.make("MeteoGym-v0", config=config)
    num_sats = len(config.satellite_ids)
    agent = RandomAgent(num_satellites=num_sats, seed=42)
    
    # Almacenará la cantidad de tareas visibles (cuando sea > 0)
    active_tasks_counts = []
    total_steps = 0

    print(f"Configuración: satélites={config.satellite_ids}, usuarios={config.num_users}, duración={config.duration_s}s, paso={config.step_s}s")

    for ep in range(num_episodes):
        print(f"\n[episodio {ep + 1}/{num_episodes}] reiniciando entorno...")
        obs, _ = env.reset(seed=100 + ep)
        terminated = truncated = False
        ep_steps = 0
        ep_active_tasks_count = 0
        
        while not (terminated or truncated):
            action = agent.select_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            ep_steps += 1
            total_steps += 1
            
            for sat_idx in range(num_sats):
                # Usamos la llave real de meteogym
                tasks_array = obs["task_present"][sat_idx]
                num_tasks = int(np.sum(tasks_array))
                
                if num_tasks > 0:
                    active_tasks_counts.append(num_tasks)
                    ep_active_tasks_count += 1

        print(f"[episodio {ep + 1}/{num_episodes}] terminado. Pasos: {ep_steps}, muestras registradas en este episodio: {ep_active_tasks_count}")
        print(f"[episodio {ep + 1}/{num_episodes}] total acumulado de muestras >0: {len(active_tasks_counts)}, pasos totales: {total_steps}")
    
    print("\nSimulación terminada. Cerrando entorno...")
    env.close()
    
    if not active_tasks_counts:
        print("No se registraron tareas en toda la simulación.")
        return

    counts = np.array(active_tasks_counts)
    print(f"--- Resultados de Congestión ({num_episodes} episodios) ---")
    print(f"Muestras útiles (instantes con >0 tareas): {len(counts)}")
    print(f"Media de tareas simultáneas: {np.mean(counts):.2f}")
    print(f"Mediana de tareas simultáneas: {np.median(counts):.2f}")
    print(f"Percentil 75: {np.percentile(counts, 75):.2f}")
    print(f"Máximo observado: {np.max(counts)}")
    print("---------------------------------------------------------")
    print("Sugerencia: Usa el valor de la Media o del Percentil 75 redondeado")
    print("como tu 'high_congestion_threshold' en state_discretizer.py.")

if __name__ == "__main__":
    run_congestion_estimation()