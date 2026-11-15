from __future__ import annotations

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import meteogym
from meteogym import SimulationConfig


# =====================================================================
# 1. FUNCIÓN DE DISCRETIZACIÓN DE ESTADOS (Para Q-Learning)
# =====================================================================
def discretize_observation(obs: dict, urgent_threshold_s: float = 30.0) -> tuple:
    """
    Transforma el diccionario continuo de observaciones de MeteoGym en una
    tupla discreta e inmutable (hashable) para usar como llave en la Q-Table.
    
    Retorna:
        Una tupla de largo S: ((e_0, c_0, w_0), (e_1, c_1, w_1), ...)
    """
    num_sats = len(obs["satellite_ids"])
    state_per_sat = []

    for s in range(num_sats):
        # 1. Estado energético (0: bloqueado, 1: bajo, 2: medio, 3: alto)
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

        # 2. Condición solar (0: eclipse, 1: iluminado por el sol)
        sun_state = int(obs["sunlit"][s])

        # 3. Carga de trabajo y urgencia de tareas visibles
        task_mask = obs["task_present"][s].astype(bool)
        num_tasks = int(np.sum(task_mask))

        if num_tasks == 0:
            workload_state = 0
        else:
            # Mínimo tiempo restante entre las tareas visibles por el satélite s
            min_time_left = float(np.min(obs["task_time_left_s"][s][task_mask]))
            if min_time_left <= urgent_threshold_s:
                workload_state = 2  # Tarea urgente presente
            elif num_tasks <= 2:
                workload_state = 1  # Poca carga y holgada
            else:
                workload_state = 3  # Alta congestión (>= 3 tareas) y holgada

        state_per_sat.append((energy_state, sun_state, workload_state))

    return tuple(state_per_sat)


# =====================================================================
# 2. AGENTE DE REFERENCIA ALEATORIO (Punto 4)
# =====================================================================
class RandomAgent:
    """
    Agente que selecciona acciones siguiendo una distribución uniforme
    sobre el espacio de acciones MultiBinary(S), equivalente a elegir
    cada componente en {0, 1} con p = 0.5 de forma independiente.
    """

    def __init__(self, num_satellites: int, seed: int | None = None) -> None:
        self.num_satellites = num_satellites
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: dict | tuple | None = None) -> np.ndarray:
        """Retorna un vector binario de dimensión (S,) con dtype int8."""
        return self.rng.integers(0, 2, size=self.num_satellites, dtype=np.int8)


# =====================================================================
# 3. EVALUACIÓN MULTI-SEMILLA Y DEMOSTRACIÓN (Puntos 1, 4 y 6)
# =====================================================================
def evaluate_random_baseline(
    config: SimulationConfig,
    num_episodes: int = 20,
    base_seed: int = 100,
) -> dict:
    """
    Ejecuta múltiples episodios independientes variando tanto la semilla del
    entorno (posiciones de usuarios, llegadas y tamaños de tareas) como la
    del agente aleatorio.
    """
    env = gym.make("MeteoGym-v0", config=config)
    num_sats = len(config.satellite_ids)

    episode_rewards = []
    episode_completed = []
    episode_failed = []
    steps_rewards_matrix = []  # Para graficar recompensa promedio vs step

    try:
        for ep in range(num_episodes):
            env_seed = base_seed + ep
            agent_seed = base_seed + 1000 + ep

            obs, info = env.reset(seed=env_seed)
            agent = RandomAgent(num_satellites=num_sats, seed=agent_seed)

            # Verificación de discretización en el primer paso del primer episodio
            if ep == 0:
                discrete_s = discretize_observation(obs)
                print(f"[Demo Interacción] Estado inicial discretizado: {discrete_s}")

            terminated = truncated = False
            step_rewards = []

            while not (terminated or truncated):
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                step_rewards.append(reward)

            episode_rewards.append(info["total_reward"])
            episode_completed.append(info["completed_tasks"])
            episode_failed.append(info["failed_tasks"])
            steps_rewards_matrix.append(step_rewards)

            print(
                f"Episodio {ep + 1:02d}/{num_episodes:02d} (seed={env_seed}) | "
                f"Recompensa Total: {info['total_reward']:7.1f} | "
                f"Completadas: {info['completed_tasks']:4d} | "
                f"Fallidas: {info['failed_tasks']:4d} | "
                f"SoC Final: { np.round(obs['battery_soc'], 3) }"
            )
    finally:
        env.close()

    steps_rewards_arr = np.array(steps_rewards_matrix)  # Shape: (num_episodes, num_steps)
    results = {
        "rewards": np.array(episode_rewards),
        "completed": np.array(episode_completed),
        "failed": np.array(episode_failed),
        "mean_step_reward": np.mean(steps_rewards_arr, axis=0),
        "std_step_reward": np.std(steps_rewards_arr, axis=0),
        "mean_cum_reward": np.mean(np.cumsum(steps_rewards_arr, axis=1), axis=0),
        "std_cum_reward": np.std(np.cumsum(steps_rewards_arr, axis=1), axis=0),
    }

    print("\n--- Resumen Estadístico Agente Aleatorio ---")
    print(f"Recompensa Total : {results['rewards'].mean():.2f} ± {results['rewards'].std():.2f}")
    print(f"Tareas Completadas: {results['completed'].mean():.2f} ± {results['completed'].std():.2f}")
    print(f"Tareas Fallidas   : {results['failed'].mean():.2f} ± {results['failed'].std():.2f}")

    return results


def plot_baseline_results(results: dict, step_s: float = 10.0) -> None:
    """Genera el gráfico de recompensa vs steps de simulación (con banda de desviación)."""
    num_steps = len(results["mean_cum_reward"])
    steps = np.arange(1, num_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Gráfico 1: Recompensa promedio instantánea por step
    axes[0].plot(steps, results["mean_step_reward"], color="tab:gray", label="Media por step")
    axes[0].fill_between(
        steps,
        results["mean_step_reward"] - results["std_step_reward"],
        results["mean_step_reward"] + results["std_step_reward"],
        color="tab:gray",
        alpha=0.25,
        label="±1 Desv. Estándar",
    )
    axes[0].set_title("Recompensa Instantánea Promedio vs. Step (Agente Aleatorio)")
    axes[0].set_xlabel(f"Step de simulación (Δt = {step_s:.0f}s)")
    axes[0].set_ylabel("Recompensa por Step")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Gráfico 2: Recompensa acumulada promedio a lo largo del episodio
    axes[1].plot(steps, results["mean_cum_reward"], color="tab:blue", label="Media acumulada")
    axes[1].fill_between(
        steps,
        results["mean_cum_reward"] - results["std_cum_reward"],
        results["mean_cum_reward"] + results["std_cum_reward"],
        color="tab:blue",
        alpha=0.25,
        label="±1 Desv. Estándar",
    )
    axes[1].set_title("Recompensa Acumulada Promedio vs. Step (Agente Aleatorio)")
    axes[1].set_xlabel(f"Step de simulación (Δt = {step_s:.0f}s)")
    axes[1].set_ylabel("Recompensa Acumulada")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("random_baseline_performance.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Configuración base del experimento (puedes ajustar satellite_ids según defina tu grupo)
    sim_config = SimulationConfig(
        satellite_ids=(1, 2),
        num_users=100,
        duration_s=5400.0,
        step_s=10.0,
    )

    stats = evaluate_random_baseline(sim_config, num_episodes=20, base_seed=42)
    plot_baseline_results(stats, step_s=sim_config.step_s)