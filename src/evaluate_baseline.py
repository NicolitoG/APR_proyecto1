# evaluate_baseline.py
from __future__ import annotations
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import meteogym
from meteogym import SimulationConfig
from random_agent import RandomAgent
from state_discretizer import discretize_observation
from reward_function import compute_custom_reward


def get_next_run_dir(base_dir: Path = Path("resultados_entrega1")) -> Path:
    """
    Busca carpetas con el formato 'ejecucion_n' (o 'ejecución_n') dentro de base_dir
    y crea la siguiente disponible para no sobrescribir experimentos previos.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_nums = []

    for item in base_dir.iterdir():
        if item.is_dir():
            match = re.match(r"^ejecuci[oó]n_(\d+)$", item.name)
            if match:
                existing_nums.append(int(match.group(1)))

    next_num = max(existing_nums, default=0) + 1
    run_dir = base_dir / f"ejecución_{next_num}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_hyperparameters_txt(
    run_dir: Path,
    config: SimulationConfig,
    reward_weights: dict[str, float],
    num_episodes: int,
    base_seed: int,
    urgent_threshold_s: float = 30.0,
) -> None:
    """Genera un archivo de texto legible con todos los hiperparámetros de la corrida."""
    txt_path = run_dir / "hiperparametros.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "==========================================================",
        f" REGISTRO DE HIPERPARÁMETROS - {run_dir.name.upper()}",
        f" Fecha de ejecución: {timestamp}",
        "==========================================================",
        "",
        "[1. Parámetros de Evaluación]",
        f"Agente               : RandomAgent (Distribución Uniforme)",
        f"Número de episodios  : {num_episodes}",
        f"Semilla base (seed)  : {base_seed}",
        f"Umbral urgencia (s)  : {urgent_threshold_s}",
        "",
        "[2. Pesos de la Función de Recompensa Custom]",
        "Fórmula: R_t = alpha*N_exitosas - beta*N_fallidas_tierra - gamma*N_fallidas_sat - delta*N_descargas - epsilon*N_ocioso",
        f"alpha (exitosas)     : {reward_weights['alpha']}",
        f"beta (fallo tierra)  : {reward_weights['beta']}",
        f"gamma (fallo sat)    : {reward_weights['gamma']}",
        f"delta (descargas)    : {reward_weights['delta']}",
        f"epsilon (ocioso)     : {reward_weights['epsilon']}",
        "",
        "[3. Configuración del Simulador (SimulationConfig)]",
        f"Satélites (IDs)      : {config.satellite_ids}",
        f"Número de usuarios   : {config.num_users}",
        f"Duración episodio (s): {config.duration_s}",
        f"Paso de tiempo dt (s): {config.step_s}",
        f"Modo llegada tareas  : {config.task_arrival_mode}",
        f"Tasa tareas (/min)   : {config.task_rate_per_minute}",
        f"Tamaño tareas (MB)   : {config.task_size_mb}",
        f"Deadline tareas (s)  : {config.task_deadline_s}",
        f"Ancho de banda (Mbps): {config.uplink_mbps}",
        f"Energía tarea (J/MB) : {config.task_energy_j_per_mb}",
        f"SoC inicial batería  : {config.initial_soc}",
        f"SoC corte / reinicio : {config.battery.cutoff_soc} / {config.battery.resume_soc}",
        "==========================================================",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_random_evaluation(
    config: SimulationConfig,
    reward_weights: dict[str, float],
    num_episodes: int = 4,
    base_seed: int = 42,
    base_output_dir: Path = Path("resultados_entrega1"),
) -> dict:
    # Crear carpeta automática: resultados_entrega1/ejecución_n
    output_dir = get_next_run_dir(base_output_dir)
    print(f"[Inicio] Guardando resultados en: {output_dir}")

    # Guardar el archivo .txt con los hiperparámetros antes de empezar
    save_hyperparameters_txt(
        run_dir=output_dir,
        config=config,
        reward_weights=reward_weights,
        num_episodes=num_episodes,
        base_seed=base_seed,
    )

    env = gym.make("MeteoGym-v0", config=config)
    num_sats = len(config.satellite_ids)

    episodes_summary = []
    native_steps_matrix = []
    custom_steps_matrix = []

    try:
        for ep in range(num_episodes):
            env_seed = base_seed + ep
            agent_seed = base_seed + 1000 + ep

            obs, info = env.reset(seed=env_seed)
            agent = RandomAgent(num_satellites=num_sats, seed=agent_seed)

            if ep == 0:
                print(f"[Check] Estado inicial discretizado: {discretize_observation(obs)}")

            terminated = truncated = False
            ep_native_rewards = []
            ep_custom_rewards = []
            totals = {
                "n_exitosas": 0,
                "n_fallidas_tierra": 0,
                "n_fallidas_sat": 0,
                "n_descargas": 0,
                "n_ocioso": 0,
            }

            while not (terminated or truncated):
                action = agent.select_action(obs)
                obs, native_reward, terminated, truncated, info = env.step(action)

                # Pasamos los pesos desempacando el diccionario **reward_weights
                custom_reward, breakdown = compute_custom_reward(
                    obs, info, **reward_weights
                )

                ep_native_rewards.append(native_reward)
                ep_custom_rewards.append(custom_reward)
                for k in totals:
                    totals[k] += breakdown[k]

            native_steps_matrix.append(ep_native_rewards)
            custom_steps_matrix.append(ep_custom_rewards)

            ep_row = {
                "episode": ep + 1,
                "env_seed": env_seed,
                "agent_seed": agent_seed,
                "native_total_reward": float(info["total_reward"]),
                "custom_total_reward": float(np.sum(ep_custom_rewards)),
                "completed_tasks": int(info["completed_tasks"]),
                "failed_tasks": int(info["failed_tasks"]),
                "failed_ground": totals["n_fallidas_tierra"],
                "failed_satellite": totals["n_fallidas_sat"],
                "discharge_steps": totals["n_descargas"],
                "idle_active_steps": totals["n_ocioso"],
                "final_mean_soc": float(np.mean(obs["battery_soc"])),
            }
            episodes_summary.append(ep_row)

            print(
                f"Ep {ep + 1:02d}/{num_episodes:02d} | "
                f"R_Nativa: {ep_row['native_total_reward']:7.1f} | "
                f"R_Custom: {ep_row['custom_total_reward']:7.1f} | "
                f"Éxitos: {ep_row['completed_tasks']:4d} | "
                f"Fallos(Tierra/Sat): {ep_row['failed_ground']}/{ep_row['failed_satellite']}"
            )
    finally:
        env.close()

    # 1. Guardar tabla de episodios en CSV
    episodes_csv = output_dir / "resumen_episodios.csv"
    with episodes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes_summary[0].keys()))
        writer.writeheader()
        writer.writerows(episodes_summary)

    # 2. Calcular curvas paso a paso y guardarlas en CSV
    native_arr = np.array(native_steps_matrix)
    custom_arr = np.array(custom_steps_matrix)
    native_cum = np.cumsum(native_arr, axis=1)
    custom_cum = np.cumsum(custom_arr, axis=1)

    steps_csv = output_dir / "curvas_por_step.csv"
    with steps_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "time_s",
            "native_step_mean", "native_step_std",
            "native_cum_mean", "native_cum_std",
            "custom_step_mean", "custom_step_std",
            "custom_cum_mean", "custom_cum_std",
        ])
        for s_idx in range(native_arr.shape[1]):
            writer.writerow([
                s_idx + 1,
                (s_idx + 1) * config.step_s,
                float(np.mean(native_arr[:, s_idx])),
                float(np.std(native_arr[:, s_idx])),
                float(np.mean(native_cum[:, s_idx])),
                float(np.std(native_cum[:, s_idx])),
                float(np.mean(custom_arr[:, s_idx])),
                float(np.std(custom_arr[:, s_idx])),
                float(np.mean(custom_cum[:, s_idx])),
                float(np.std(custom_cum[:, s_idx])),
            ])

    # 3. Guardar resumen global en JSON (incluyendo los pesos usados)
    native_totals = [r["native_total_reward"] for r in episodes_summary]
    custom_totals = [r["custom_total_reward"] for r in episodes_summary]
    summary_json = output_dir / "estadisticas_globales.json"
    summary_data = {
        "run_folder": output_dir.name,
        "num_episodes": num_episodes,
        "base_seed": base_seed,
        "reward_weights": reward_weights,
        "native_reward_mean": float(np.mean(native_totals)),
        "native_reward_std": float(np.std(native_totals)),
        "custom_reward_mean": float(np.mean(custom_totals)),
        "custom_reward_std": float(np.std(custom_totals)),
    }
    summary_json.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # 4. Guardar gráfico PNG
    steps = np.arange(1, native_arr.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(steps, np.mean(native_cum, axis=0), color="tab:blue", label="Recompensa Nativa")
    axes[0].fill_between(
        steps,
        np.mean(native_cum, axis=0) - np.std(native_cum, axis=0),
        np.mean(native_cum, axis=0) + np.std(native_cum, axis=0),
        color="tab:blue", alpha=0.25
    )
    axes[0].set_title(f"Recompensa Acumulada Nativa ({output_dir.name})")
    axes[0].set_xlabel("Step de simulación")
    axes[0].set_ylabel("Recompensa Acumulada")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(steps, np.mean(custom_cum, axis=0), color="tab:orange", label="Recompensa Custom")
    axes[1].fill_between(
        steps,
        np.mean(custom_cum, axis=0) - np.std(custom_cum, axis=0),
        np.mean(custom_cum, axis=0) + np.std(custom_cum, axis=0),
        color="tab:orange", alpha=0.25
    )
    axes[1].set_title(f"Recompensa Acumulada Custom ({output_dir.name})")
    axes[1].set_xlabel("Step de simulación")
    axes[1].set_ylabel("Recompensa Acumulada")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plot_path = output_dir / "grafico_recompensa_steps.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Corrida finalizada. Todos los archivos están en: {output_dir.resolve()}")
    return summary_data


if __name__ == "__main__":
    # 1. Configuración física del entorno
    sim_config = SimulationConfig(
        satellite_ids=(1, 2),
        num_users=100,
        duration_s=5400.0,
        step_s=10.0,
    )

    # 2. Hiperparámetros de la función de recompensa (modifícalos aquí entre corridas)
    pesos_recompensa = {
        "alpha": 1.0,    # Premio por tarea exitosa
        "beta": 1.0,     # Castigo por tarea fallida en tierra
        "gamma": 1.5,    # Castigo por tarea fallida en satélite
        "delta": 0.1,    # Castigo por paso descargando batería
        "epsilon": 0.7,  # Castigo por encender satélite sin procesar tareas (ocioso)
    }

    run_random_evaluation(
        config=sim_config,
        reward_weights=pesos_recompensa,
        num_episodes=4,
        base_seed=42,
    )