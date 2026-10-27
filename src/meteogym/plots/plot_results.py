"""Matplotlib world-map playback of saved MeteoGym output, with no runtime downloads."""

from __future__ import annotations

import argparse
import json
import math
from importlib.resources import files
from pathlib import Path

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
except ImportError as error:
    raise ImportError(
        "Plotting requires Matplotlib and Pillow. Install the optional dependencies with "
        "python -m pip install 'meteogym[plot]' (or -e './src[plot]' from the repository)."
    ) from error

from .data import ReplayDataset, load_dataset

ACTIVE_COLOR = "#167341"
INACTIVE_COLOR = "#c23736"
DISCHARGED_COLOR = "#808080"
USER_COLOR = "#394957"
OCEAN_COLOR = "#edf4f8"
VOLTAGE_CMAP = LinearSegmentedColormap.from_list(
    "meteogym_voltage", ["#d93d43", "#ad6eac", "#2467c7"]
).with_extremes(bad=DISCHARGED_COLOR)


def _land_background(ax):
    """Draw bundled Natural Earth polygons, including their interior holes."""
    resource = files("meteogym.plots").joinpath("data/land_110m.geojson")
    features = json.loads(resource.read_text(encoding="utf-8"))["features"]
    for feature in features:
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        )
        for polygon in polygons:
            vertices, codes = [], []
            for index, ring in enumerate(polygon):
                points = np.asarray(ring, dtype=float)
                # Nonzero winding: exterior CCW, holes CW, independent of source orientation.
                signed_area = np.sum(
                    points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]
                )
                if (signed_area > 0) != (index == 0):
                    points = points[::-1]
                vertices.extend(points)
                codes.extend(
                    [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 2) + [MplPath.CLOSEPOLY]
                )
            ax.add_patch(
                PathPatch(
                    MplPath(vertices, codes),
                    facecolor="#dce4df",
                    edgecolor="#aab9af",
                    linewidth=0.45,
                    zorder=1,
                )
            )


class MapPlayback:
    """One reusable map and a fixed set of artists; Enter advances in step mode."""

    def __init__(self, dataset: ReplayDataset, *, interval_ms: float = 200, frame: int = 0):
        if not math.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("interval_ms must be finite and positive")
        if not 0 <= frame < len(dataset):
            raise ValueError(f"frame must be between 0 and {len(dataset) - 1}")
        initial = dataset.frame(frame)
        self.dataset = dataset
        self.interval_ms = interval_ms
        self.index = frame
        self.animation = None
        self.paused = False
        self._next_index = frame
        self.figure, self.ax = plt.subplots(figsize=(13.6, 7.8))
        self.figure.set_facecolor("#ffffff")
        self.figure.canvas.manager.set_window_title("MeteoGym | orbital replay")
        self.ax.set_facecolor(OCEAN_COLOR)
        _land_background(self.ax)
        self.ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (°)", ylabel="Latitude (°)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xticks(np.arange(-180, 181, 60))
        self.ax.set_yticks(np.arange(-90, 91, 30))
        self.ax.grid(color="#b9c7ce", linewidth=0.55, alpha=0.7, zorder=2)
        self.ax.tick_params(colors="#52636e", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#b9c7ce")

        users = dataset.user_locations_deg
        self.users = self.ax.scatter(
            users[:, 1],
            users[:, 0],
            s=14,
            c=USER_COLOR,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        self.offline_users = self.ax.scatter(
            [], [], s=14, marker="x", c="#b5bec4", linewidths=0.7, zorder=3
        )
        self.satellites = self.ax.scatter(
            initial.satellite_locations_deg[:, 1],
            initial.satellite_locations_deg[:, 0],
            s=175,
            c=initial.voltage_v,
            cmap=VOLTAGE_CMAP,
            norm=Normalize(*dataset.voltage_limits, clip=True),
            plotnonfinite=True,
            edgecolors="#ffffff",
            linewidths=1.6,
            zorder=5,
        )
        self.labels = [
            self.ax.annotate(
                "",
                (0, 0),
                xytext=(12, 9),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                zorder=6,
            )
            for _ in dataset.satellite_ids
        ]
        self.figure.suptitle(
            "MeteoGym · satellite availability",
            x=0.06,
            ha="left",
            fontsize=18,
            weight="bold",
            color="#263b49",
        )
        self.subtitle = self.ax.set_title("", loc="left", fontsize=10, color="#52636e", pad=12)
        self.footer = self.figure.supxlabel(
            "Enter / →  next step    ←  previous    Space  pause / play    Q  close",
            fontsize=9,
            color="#52636e",
        )
        colorbar = self.figure.colorbar(
            self.satellites, ax=self.ax, fraction=0.025, pad=0.025, shrink=0.8, extend="both"
        )
        colorbar.set_label("Battery voltage (V) · red: low / blue: full", fontsize=9)
        colorbar.ax.tick_params(labelsize=9)
        self.ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color=USER_COLOR,
                    markersize=4,
                    label="Ground user",
                ),
                Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="none",
                    color="#b5bec4",
                    markersize=4,
                    label="Offline user",
                ),
                Line2D([], [], color=ACTIVE_COLOR, linewidth=3, label="Active label"),
                Line2D([], [], color=INACTIVE_COLOR, linewidth=3, label="Inactive label"),
                Line2D(
                    [], [], marker="o", linestyle="none", color=DISCHARGED_COLOR,
                    markersize=8, label="Discharged battery",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncols=5,
            frameon=False,
            fontsize=9,
            handlelength=1.2,
            columnspacing=2,
        )
        self._key_connection = self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.update(frame)

    def update(self, index: int):
        """Update start-state artists and outcomes for the selected interval."""
        if not 0 <= index < len(self.dataset):
            raise IndexError("frame index is outside the saved episode")
        current = self.dataset.frame(index)
        charging = current.charging
        self.index = index
        self.satellites.set_offsets(current.satellite_locations_deg[:, ::-1])
        suspended = current.battery_suspended
        if suspended is None:
            # Older runs may omit suspension; only empty voltage is then observable.
            suspended = current.voltage_v <= self.dataset.voltage_limits[0]
        self.satellites.set_array(np.ma.masked_where(suspended, current.voltage_v))
        users = self.dataset.user_locations_deg
        self.users.set_offsets(users[current.user_online, ::-1])
        self.offline_users.set_offsets(users[~current.user_online, ::-1])
        for sat, label in enumerate(self.labels):
            latitude, longitude = current.satellite_locations_deg[sat]
            label.xy = (longitude, latitude)
            # Keep labels within the map near its edges.
            label.set_position((-12 if longitude >= 120 else 12, -32 if latitude > 65 else 9))
            label.set_ha("right" if longitude >= 120 else "left")
            charge_label = (
                ("Charging" if charging[sat] else "Not charging")
                if (charging is not None)
                else "Charging status unavailable"
            )
            label.set_text(
                f"S{self.dataset.satellite_ids[sat]} · {current.contact_count[sat]} users\n"
                f"{charge_label}"
            )
            label.set_color(ACTIVE_COLOR if current.active[sat] else INACTIVE_COLOR)
        utc = current.time_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        totals = "  ·  ".join(
            f"{label} {value:g}" if value is not None else f"{label} unavailable"
            for label, value in (
                ("generated", current.generated_tasks),
                ("failed", current.failed_tasks),
                ("completed", current.completed_tasks),
                ("reward", current.total_reward),
            )
        )
        self.subtitle.set_text(
            f"Step {current.step}  ·  t = {current.elapsed_s:g} s  ·  {utc}\n"
            f"{np.count_nonzero(current.active)}/{len(current.active)} satellites active  ·  "
            f"{np.count_nonzero(current.user_online)}/{len(current.user_online)} users online"
            "  ·  position, contacts and voltage at interval start\n"
            f"Totals through step end: {totals}"
        )
        return (self.satellites, self.users, self.offline_users, self.subtitle, *self.labels)

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.figure)
        elif event.key == " " and self.animation is not None:
            self.paused = not self.paused
            if self.paused:
                self.animation.pause()
            else:
                self._next_index = (self.index + 1) % len(self.dataset)
                self.animation.resume()
        elif event.key in ("enter", "return", "right", "left", "backspace", "home", "end"):
            if self.animation is not None:
                self.animation.pause()
                self.paused = True
            if event.key == "home":
                index = 0
            elif event.key == "end":
                index = len(self.dataset) - 1
            else:
                direction = -1 if event.key in ("left", "backspace") else 1
                index = min(max(self.index + direction, 0), len(self.dataset) - 1)
            self.update(index)
            self.figure.canvas.draw_idle()

    def animate(self):
        """Start looping playback while retaining the animation object."""
        if self.animation is not None:
            return self.animation

        def tick(_):
            if not self.paused:
                self.update(self._next_index)
                self._next_index = (self.index + 1) % len(self.dataset)
            return (self.satellites, self.users, self.offline_users, self.subtitle, *self.labels)

        self.animation = FuncAnimation(
            self.figure,
            tick,
            init_func=lambda: self.update(self.index),
            interval=self.interval_ms,
            cache_frame_data=False,
        )
        return self.animation

    def save(self, path: str | Path, *, stride: int = 1):
        """Export the current static frame or an animated GIF, without external encoders."""
        path = Path(path)
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        suffix = path.suffix.lower()
        if suffix not in (".png", ".pdf", ".svg", ".gif"):
            raise ValueError("save path must end in .png, .pdf, .svg, or .gif")
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix != ".gif":
            self.figure.savefig(path, dpi=140)
            return
        first = self.index
        indices = list(range(0, len(self.dataset), stride))
        if indices[-1] != len(self.dataset) - 1:
            indices.append(len(self.dataset) - 1)
        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=indices,
            init_func=lambda: self.update(indices[0]),
            interval=self.interval_ms,
            repeat=False,
            cache_frame_data=False,
        )
        try:
            animation.save(path, writer=PillowWriter(fps=1000 / self.interval_ms), dpi=90)
        finally:
            self.update(first)


def plot_dataset(
    dataset_dir: str | Path,
    *,
    mode: str = "animate",
    interval_ms: float = 200,
    save: str | Path | None = None,
    show: bool = True,
    frame: int = 0,
    stride: int = 1,
) -> MapPlayback:
    """Load an output directory, optionally export it, and display interactive playback."""
    if mode not in ("animate", "step"):
        raise ValueError("mode must be 'animate' or 'step'")
    if not show and save is None:
        raise ValueError("--no-show requires --save")
    if not show:
        plt.switch_backend("Agg")
    viewer = MapPlayback(load_dataset(dataset_dir), interval_ms=interval_ms, frame=frame)
    try:
        if save is not None:
            viewer.save(save, stride=stride)
        if show:
            backend = matplotlib.get_backend().lower()
            if backend in ("agg", "pdf", "pgf", "ps", "svg", "template") or "inline" in backend:
                raise RuntimeError(
                    f"Backend {matplotlib.get_backend()} cannot open an interactive window. "
                    "Use a GUI backend (for example MPLBACKEND=TkAgg) or export with "
                    "--save map.gif --no-show."
                )
            if mode == "animate":
                viewer.animate()
            plt.show(block=True)
        else:
            plt.close(viewer.figure)
    except Exception:
        plt.close(viewer.figure)
        raise
    return viewer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory with summary.json and steps.jsonl")
    parser.add_argument("--mode", choices=("animate", "step"), default="animate")
    parser.add_argument(
        "--interval",
        type=float,
        default=200,
        metavar="MS",
        help="Animation frame interval in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Initial frame index, starting at zero"
    )
    parser.add_argument(
        "--save", type=Path, help="Export .gif animation or .png/.pdf/.svg snapshot"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Save every Nth GIF frame, always including the last (default: 1)",
    )
    parser.add_argument("--no-show", action="store_true", help="Export without opening a window")
    args = parser.parse_args(argv)
    try:
        plot_dataset(
            args.dataset,
            mode=args.mode,
            interval_ms=args.interval,
            save=args.save,
            show=not args.no_show,
            frame=args.frame,
            stride=args.stride,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
