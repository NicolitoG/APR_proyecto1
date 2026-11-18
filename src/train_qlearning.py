# train_qlearning.py
from __future__ import annotations
import csv
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import random
import gc 

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import meteogym
from meteogym import SimulationConfig
from q_learning_agent import QLearningAgent
from state_discretizer import discretize_observation
from reward_function import compute_custom_reward


# =====================================================================
# 1. GESTIÓN DE CARPETAS Y REGISTROS
# =====================================================================
def get_next_train_dir(base_dir: Path = Path("resultados_entrega1")) -> Path:
    """Crea automáticamente la siguiente carpeta de entrenamiento (ej. qlearning_ejecucion_1)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_nums = []

    for item in base_dir.iterdir():
        if item.is_dir():
            match = re.match(r"^qlearning_ejecuci[oó]n_(\d+)$", item.name)
            if match:
                existing_nums.append(int(match.group(1)))

    next_num = max(existing_nums, default=0) + 1
    run_dir = base_dir / f"qlearning_ejecución_{next_num}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_training_hyperparameters(
    run_dir: Path,
    config: SimulationConfig,
    reward_weights: dict[str, float],
    agent_params: dict,
    num_episodes: int,
    base_seed: int,
    congestion_threshold: float,
    urgency_threshold_s: float,
) -> None:
    """Guarda un archivo de texto con todos los parámetros del entrenamiento."""
    txt_path = run_dir / "hiperparametros.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "==========================================================",
        f" REGISTRO DE ENTRENAMIENTO Q-LEARNING - {run_dir.name.upper()}",
        f" Fecha: {timestamp}",
        "==========================================================",
        "",
        "[1. Parámetros de Q-Learning]",
        f"Episodios de train   : {num_episodes}",
        f"Alpha (Tasa aprend.) : {agent_params['alpha']}",
        f"Gamma (Descuento)    : {agent_params['gamma']}",
        f"Epsilon inicial      : {agent_params['epsilon_start']}",
        f"Epsilon mínimo       : {agent_params['epsilon_min']}",
        f"Epsilon decay        : {agent_params['epsilon_decay']}",
        "",
        "[2. Parámetros del Espacio de Estados]",
        f"Umbral congestión    : {congestion_threshold}",
        f"Umbral urgencia (s)  : {urgency_threshold_s}",
        "",
        "[3. Pesos de la Recompensa Custom]",
        *[f"{k:20s} : {v}" for k, v in reward_weights.items()],
        "",
        "[4. Configuración del Simulador]",
        f"Satélites            : {len(config.satellite_ids)} (IDs: {config.satellite_ids})",
        f"Usuarios             : {config.num_users}",
        f"Duración (s)         : {config.duration_s}",
        f"Semilla base         : {base_seed}",
        "==========================================================",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =====================================================================
# 2. MOTOR DEL EPISODIO
# =====================================================================
def run_single_episode(
    env: gym.Env,
    agent: QLearningAgent,
    env_seed: int,
    reward_weights: dict[str, float],
    congestion_threshold: float,
    urgency_threshold_s: float,
) -> dict:
    """Ejecuta un solo episodio completo actualizando la tabla Q en cada paso."""
    obs, info = env.reset(seed=env_seed)
    
    # Discretizar estado inicial
    state = discretize_observation(obs, congestion_threshold, urgency_threshold_s)
    
    terminated = truncated = False
    ep_native_reward = 0.0
    ep_custom_reward = 0.0
    
    while not (terminated or truncated):
        # 1. Elegir acción (Comportamiento b: epsilon-greedy)
        action_array = agent.select_action(state, training=True)
        
        # 2. Interactuar con el simulador
        next_obs, native_reward, terminated, truncated, next_info = env.step(action_array)
        
        # 3. Calcular la recompensa custom (Shaping)
        custom_reward, breakdown = compute_custom_reward(next_obs, next_info, **reward_weights)
        
        # 4. Discretizar el nuevo estado
        next_state = discretize_observation(next_obs, congestion_threshold, urgency_threshold_s)
        
        # 5. Ecuación de Bellman: Actualizar la tabla Q
        agent.update(state, action_array, custom_reward, next_state)
        
        # 6. Avanzar el tiempo
        state = next_state
        ep_native_reward += native_reward
        ep_custom_reward += custom_reward
        
    # 7. Decaimiento de exploración al final del episodio
    agent.decay_epsilon()
    
    return {
        "native_total_reward": float(next_info["total_reward"]),
        "custom_total_reward": ep_custom_reward,
        "completed_tasks": int(next_info["completed_tasks"]),
        "failed_tasks": int(next_info["failed_tasks"]),
        "final_epsilon": agent.epsilon
    }


# =====================================================================
# 3. GRAFICADOR
# =====================================================================
def plot_learning_curve(episodes_data: list[dict], output_dir: Path) -> None:
    """Genera y guarda las gráficas de aprendizaje (con media móvil para suavizar)."""
    episodes = np.arange(1, len(episodes_data) + 1)
    custom_rewards = [d["custom_total_reward"] for d in episodes_data]
    native_rewards = [d["native_total_reward"] for d in episodes_data]
    
    # Función lambda para calcular la media móvil (ventana de 20)
    window = min(20, len(episodes_data))
    moving_avg = lambda data: np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # Gráfico 1: Recompensa Custom (la que el agente usa para aprender)
    axes[0].plot(episodes, custom_rewards, alpha=0.3, color="tab:orange", label="Por episodio")
    if len(episodes) >= window:
        axes[0].plot(episodes[window-1:], moving_avg(custom_rewards), color="darkorange", linewidth=2, label=f"Media móvil (n={window})")
    axes[0].set_title("Evolución de la Recompensa Custom")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Recompensa")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    # Gráfico 2: Recompensa Nativa (la del simulador puro)
    axes[1].plot(episodes, native_rewards, alpha=0.3, color="tab:blue", label="Por episodio")
    if len(episodes) >= window:
        axes[1].plot(episodes[window-1:], moving_avg(native_rewards), color="darkblue", linewidth=2, label=f"Media móvil (n={window})")
    axes[1].set_title("Desempeño Real (Recompensa Nativa)")
    axes[1].set_xlabel("Episodio")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "curva_entrenamiento.png", dpi=150)
    plt.close(fig)

# =====================================================================
# 4. GUARDADO DE CHECKPOINTS
# =====================================================================
def save_checkpoints(output_dir: Path, agent: QLearningAgent, episodes_summary: list[dict]) -> None:
    """Sobrescribe los archivos de salida con el progreso hasta el episodio actual."""
    # 1. Guardar tabla Q
    agent.save(output_dir / "q_table.pkl")
    
    # 2. Guardar CSV de historial
    if episodes_summary:
        csv_path = output_dir / "historial_episodios.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episodes_summary[0].keys()))
            writer.writeheader()
            writer.writerows(episodes_summary)
            
    # 3. Actualizar gráficas (solo si hay más de 1 episodio para evitar warnings de matplotlib)
    if len(episodes_summary) > 1:
        plot_learning_curve(episodes_summary, output_dir)

# =====================================================================
# 5. ORQUESTADOR PRINCIPAL
# =====================================================================
def train_agent(
    config: SimulationConfig,
    reward_weights: dict[str, float],
    agent_params: dict,
    num_episodes: int = 400,
    base_seed: int = 42,
    congestion_threshold: float = 2.5,
    urgency_threshold_s: float = 30.0
) -> None:
    """Inicializa el entorno y el agente, corre el entrenamiento completo y guarda resultados."""
    output_dir = get_next_train_dir()
    start_time = time.perf_counter()
    start_datetime = datetime.now()
    print(f"\n[Inicio] Entrenando Q-Learning por {num_episodes} episodios.")
    print(f"[Inicio] Hora de inicio: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Ruta] Guardando en: {output_dir}")
    
    # 1. Guardar metadatos
    save_training_hyperparameters(
        run_dir=output_dir,
        config=config,
        reward_weights=reward_weights,
        agent_params=agent_params,
        num_episodes=num_episodes,
        base_seed=base_seed,
        congestion_threshold=congestion_threshold,
        urgency_threshold_s=urgency_threshold_s
    )
    
    env = gym.make("MeteoGym-v0", config=config)
    num_sats = len(config.satellite_ids)
    
    # 2. Inicializar agente
    agent = QLearningAgent(num_satellites=num_sats, **agent_params)
    
    episodes_summary = []
    
    try:
        # 3. Bucle de episodios
        for ep in range(num_episodes):
            env_seed = base_seed + ep  # Nueva semilla cada episodio para generalizar
            episode_start = time.perf_counter()
            
            ep_stats = run_single_episode(
                env=env,
                agent=agent,
                env_seed=env_seed,
                reward_weights=reward_weights,
                congestion_threshold=congestion_threshold,
                urgency_threshold_s=urgency_threshold_s
            )
            
            ep_stats["episode"] = ep + 1
            episodes_summary.append(ep_stats)

            # ---------------------------------------------------------
            # Checkpoint y limpieza de memoria cada 10 episodios
            # ---------------------------------------------------------
            if (ep + 1) % 10 == 0:
                save_checkpoints(output_dir, agent, episodes_summary)
                gc.collect()  # Fuerza a Python a liberar memoria RAM huérfana

            elapsed_total = time.perf_counter() - start_time
            elapsed_episode = time.perf_counter() - episode_start
            progress_pct = ((ep + 1) / num_episodes) * 100
            avg_episode_time = elapsed_total / (ep + 1) if ep + 1 else 0
            remaining_time = avg_episode_time * (num_episodes - (ep + 1))
            eta = datetime.now() + timedelta(seconds=remaining_time)
            
            # Imprimir progreso cada 10 episodios (o al inicio/fin)
            if (ep + 1) % 10 == 0 or ep == 0 or (ep + 1) == num_episodes:
                print(f"[Progreso] Ep {ep + 1:03d}/{num_episodes} ({progress_pct:5.1f}%) | "
                      f"R_Custom: {ep_stats['custom_total_reward']:7.1f} | "
                      f"R_Nativa: {ep_stats['native_total_reward']:7.1f} | "
                      f"Éxitos: {ep_stats['completed_tasks']:3d} | "
                      f"Epsilon: {ep_stats['final_epsilon']:.3f} | "
                      f"Tiempo: {elapsed_total:.1f}s | "
                      f"ETA: {eta.strftime('%H:%M:%S')} | "
                      f"Ep. actual: {elapsed_episode:.1f}s")
    finally:
        env.close()

    # 4. Guardar resultados
    # 4a. Exportar la tabla Q entrenada
    agent.save(output_dir / "q_table.pkl")
    
    # 4b. Guardar CSV de historial
    csv_path = output_dir / "historial_episodios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes_summary[0].keys()))
        writer.writeheader()
        writer.writerows(episodes_summary)
        
    # 4c. Generar gráficas
    plot_learning_curve(episodes_summary, output_dir)
    
    # Imprimir resumen de convergencia (estados descubiertos)
    total_elapsed = time.perf_counter() - start_time
    num_estados_descubiertos = len(agent.q_table)
    print(f"\n[OK] Entrenamiento finalizado con éxito.")
    print(f"[Tiempo] Duración total: {total_elapsed:.1f} segundos ({total_elapsed / 60:.1f} minutos)")
    print(f"[Stats] El agente descubrió {num_estados_descubiertos} estados únicos.")
    print(f"[Stats] Archivos exportados a: {output_dir.resolve()}")


if __name__ == "__main__":
    # Selección aleatoria de satélites
    num_satellites = 10
    random_satellite_ids = tuple(random.sample(range(1, 1081), num_satellites))

    # A. Configuración del simulador
    sim_config = SimulationConfig(
        satellite_ids=random_satellite_ids,
        num_users=100,
        duration_s=5400.0,
        step_s=10.0,
    )
    
    # B. Pesos de la recompensa
    pesos_recompensa = {
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 1.5,
        "delta": 0.1,
        "epsilon": 0.2,
    }
    
    # C. Hiperparámetros matemáticos del Q-Learning
    parametros_agente = {
        "alpha": 0.1,             # Tasa de aprendizaje
        "gamma": 0.95,            # Factor de descuento
        "epsilon_start": 1.0,     # Exploración inicial (100%)
        "epsilon_min": 0.05,      # Exploración mínima remanente (5%)
        "epsilon_decay": 0.99,    # Multiplicador por episodio
        "seed": 42                # Semilla para el RNG interno del agente
    }
    
    # Ejecutar el entrenamiento
    train_agent(
        config=sim_config,
        reward_weights=pesos_recompensa,
        agent_params=parametros_agente,
        num_episodes=100,          # 400 episodios es un buen balance para empezar
        base_seed=1000,           # Semilla base para generalizar usuarios
        congestion_threshold=2.5,
        urgency_threshold_s=30.0
    )