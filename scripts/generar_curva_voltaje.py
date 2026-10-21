"""Genera una curva ilustrativa de descarga; no representa mediciones reales."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "images"
TASK_START = 40.0
TASK_END = 54.0
BASE_SLOPE = 0.0014
TASK_DROP = 0.14


def voltage(time):
    """Caída inicial, descarga suave, consumo de una tarea y caída final."""
    initial_drop = 0.13 * np.exp(-time / 1.1)
    # El consumo adicional se acumula durante la tarea y permanece después.
    task_drop = TASK_DROP * np.clip(
        (time - TASK_START) / (TASK_END - TASK_START), 0.0, 1.0
    )
    final_drop = 0.48 * (np.clip(time - 86.0, 0.0, None) / 14.0) ** 4
    return 3.30 + initial_drop - BASE_SLOPE * time - task_drop - final_drop


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
            "font.size": 13,
            "text.color": "#394049",
            "axes.labelcolor": "#394049",
            "axes.edgecolor": "#59636D",
            "xtick.color": "#59636D",
            "ytick.color": "#59636D",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.8), layout="constrained")
    ax.axvspan(TASK_START, TASK_END, color="#7C3581", alpha=0.10, lw=0)

    # Tres tramos conectados de una sola curva, sin saltos en sus uniones.
    for start, end, color in (
        (0.0, TASK_START, "#006B63"),
        (TASK_START, TASK_END, "#7C3581"),
        (TASK_END, 100.0, "#006B63"),
    ):
        time = np.linspace(start, end, 400)
        ax.plot(time, voltage(time), color=color, lw=2.6, solid_capstyle="round")

    ax.text(
        (TASK_START + TASK_END) / 2,
        3.47,
        "Procesamiento\nde una tarea",
        ha="center",
        va="top",
        fontsize=12,
        color="#7C3581",
        linespacing=1.05,
    )
    ax.set(xlim=(0, 100), ylim=(2.4, 3.5))
    ax.set_xlabel("Tiempo normalizado (%)", labelpad=6)
    ax.set_ylabel("Voltaje (V)", labelpad=6)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(2.4, 3.41, 0.2))
    ax.tick_params(labelsize=11.5, length=3.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EBEBEB", lw=0.7)

    for extension in ("pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"curva_voltaje.{extension}")
    plt.close(fig)


if __name__ == "__main__":
    main()
