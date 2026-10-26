"""Run reproducible baseline episodes or export sampled ground contacts."""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .env import MeteoGymEnv
from .orbits import OrbitModel, export_contact_table
from .workload import Workload


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="SimulationConfig JSON; CLI values override it")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--satellites",
        nargs="+",
        type=int,
        help="Explicit source IDs, including for a single satellite",
    )
    selection.add_argument(
        "--all-satellites", action="store_true", help="Select every satellite in the constellation"
    )
    parser.add_argument("--users", type=int, help="Number of randomly positioned ground users")
    parser.add_argument(
        "--user-location",
        nargs=2,
        type=float,
        action="append",
        metavar=("LAT", "LON"),
        help="Repeat for fixed users; overrides --users",
    )
    parser.add_argument(
        "--user-region", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX")
    )
    parser.add_argument(
        "--task-rate",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Uniform per-user rate range, tasks/minute",
    )
    parser.add_argument(
        "--task-size",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Task size range, decimal MB",
    )
    parser.add_argument(
        "--task-arrival-mode", choices=("single", "queued"),
        help="single waits for completion/failure; queued accumulates independent arrivals",
    )
    parser.add_argument(
        "--deadline",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Relative task deadline range, seconds",
    )
    parser.add_argument("--duration", type=float, help="Episode duration, seconds")
    parser.add_argument("--step", type=float, help="Decision/contact sampling interval, seconds")
    parser.add_argument("--seed", type=int, default=10, help="Random seed (default: 10)")
    parser.add_argument("--start-time", help="ISO 8601 time with UTC offset; defaults to TLE epoch")
    parser.add_argument("--tle-path", help="Custom TLE file or original TLE directory")
    parser.add_argument("--min-elevation", type=float, help="Minimum ground elevation, degrees")
    parser.add_argument("--uplink-mbps", type=float, help="Total upload bandwidth per satellite")
    parser.add_argument("--initial-soc", type=float, help="Initial battery charge fraction [0, 1]")
    parser.add_argument(
        "--minimum-voltage", type=float,
        help="Queued-mode terminal voltage cutoff, V (default: battery empty voltage)",
    )
    parser.add_argument(
        "--initial-tasks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start with one pending task per user",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meteogym", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run an episode using a simple baseline action policy")
    _add_simulation_arguments(run)
    run.add_argument(
        "--policy",
        choices=("active", "passive", "random", "visible"),
        default="active",
        help="Binary actions; random is seeded (default: active)",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/run"),
        help="Directory for summary.json and steps.jsonl (default: outputs/run)",
    )
    run.add_argument(
        "--plot",
        nargs="?",
        choices=("animate", "step"),
        const="animate",
        help="Open the results viewer after the run (default mode: animate)",
    )
    run.add_argument(
        "--plot-interval",
        type=float,
        default=200,
        metavar="MS",
        help="Positive animation frame interval in milliseconds (default: 200)",
    )
    run.add_argument(
        "--plot-save", type=Path, metavar="PATH", help="Save the plot as PNG, PDF, SVG or GIF"
    )
    run.add_argument(
        "--plot-no-show",
        action="store_true",
        help="Export without opening a window; requires a save",
    )
    contacts = commands.add_parser("contacts", help="Export visible satellite/user edges as CSV")
    _add_simulation_arguments(contacts)
    contacts.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/contacts.csv"),
        help="CSV path; a matching .metadata.json records users and settings",
    )
    return parser


def _validate_plot_arguments(args: argparse.Namespace) -> None:
    if not math.isfinite(args.plot_interval) or args.plot_interval <= 0:
        raise ValueError("--plot-interval must be finite and positive")
    if args.plot is None and (args.plot_save is not None or args.plot_no_show):
        raise ValueError("--plot-save and --plot-no-show require --plot")
    if args.plot_no_show and args.plot_save is None:
        raise ValueError("--plot-no-show requires --plot-save")
    if args.plot_save is not None and args.plot_save.suffix.lower() not in (
        ".png",
        ".pdf",
        ".svg",
        ".gif",
    ):
        raise ValueError("--plot-save must use a .png, .pdf, .svg or .gif extension")


def _configuration(args: argparse.Namespace) -> SimulationConfig:
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    values = {}
    if args.config is not None:
        with args.config.open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("config JSON must contain a SimulationConfig object")
    fields = {
        "users": "num_users",
        "user_region": "user_region_deg",
        "task_rate": "task_rate_per_minute",
        "task_size": "task_size_mb",
        "task_arrival_mode": "task_arrival_mode",
        "deadline": "task_deadline_s",
        "duration": "duration_s",
        "step": "step_s",
        "start_time": "start_time",
        "tle_path": "tle_path",
        "min_elevation": "min_elevation_deg",
        "uplink_mbps": "uplink_mbps",
        "initial_soc": "initial_soc",
        "initial_tasks": "initial_tasks",
    }
    for argument, field in fields.items():
        value = getattr(args, argument)
        if value is not None:
            values[field] = value
    if args.minimum_voltage is not None:
        values["battery"] = {**values.get("battery", {}), "minimum_voltage_v": args.minimum_voltage}
    if args.all_satellites:
        values["satellite_ids"] = None
    elif args.satellites is not None:
        values["satellite_ids"] = args.satellites
    if args.user_location is not None:
        values["user_locations_deg"] = args.user_location
        values["num_users"] = len(args.user_location)
    elif args.users is not None or args.user_region is not None:
        # An explicit request for random users supersedes coordinates in JSON.
        values["user_locations_deg"] = None
    return SimulationConfig.from_dict(values)


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("meteogym", "numpy", "gymnasium", "sgp4", "astropy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "uninstalled source checkout"
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _action(policy: str, observation: dict, rng: np.random.Generator) -> np.ndarray:
    count = len(observation["satellite_ids"])
    if policy == "random":
        return rng.integers(0, 2, size=count, dtype=np.int8)
    if policy == "visible":
        return (observation["contact_count"] > 0).astype(np.int8)
    return np.full(count, policy == "active", dtype=np.int8)


def _run(args: argparse.Namespace, config: SimulationConfig) -> None:
    env = MeteoGymEnv(config)
    try:
        observation, info = env.reset(seed=args.seed)
        policy_rng = np.random.default_rng(args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        step_count = 0
        terminated = truncated = False
        with (args.output_dir / "steps.jsonl").open("w", encoding="utf-8") as stream:
            while not (terminated or truncated):
                action = _action(args.policy, observation, policy_rng)
                start_s = float(observation["time_s"][0])
                sunlit = observation["sunlit"].tolist()
                contact_count = observation["contact_count"].tolist()
                satellite_geographic = observation["geographic_deg_km"].tolist()
                user_online = observation["user_online"].tolist()
                battery_voltage_start = observation["battery_voltage_v"].tolist()
                observation, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                row = {
                    "step": step_count,
                    "start_s": start_s,
                    "end_s": info["elapsed_s"],
                    "time_utc": info["time_utc"],
                    "step_s": info["step_s"],
                    "actions": action.tolist(),
                    "effective_active": info["effective_active"].astype(int).tolist(),
                    "active_duration_s": info["active_duration_s"].tolist(),
                    "sunlit": sunlit,
                    "contact_count": contact_count,
                    "satellite_geographic_deg_km": satellite_geographic,
                    "user_online": user_online,
                    "battery_voltage_start_v": battery_voltage_start,
                    "reward": reward,
                    "completed": info["completed_this_step"],
                    "failed": info["failed_this_step"],
                    "generated_tasks": info["generated_tasks"],
                    "pending_tasks": info["pending_tasks"],
                    "pending_tasks_per_user": info["pending_tasks_per_user"].tolist(),
                    "task_events": info["task_events"],
                    "battery_failure_events": info["battery_failure_events"],
                    "battery_suspended": observation["battery_suspended"].tolist(),
                    "battery_soc": observation["battery_soc"].tolist(),
                    "battery_voltage_v": observation["battery_voltage_v"].tolist(),
                    "battery_current_a": observation["battery_current_a"].tolist(),
                    "charging": observation["charging"].tolist(),
                    "task_energy_j": info["task_energy_j"].tolist(),
                }
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        results = {
            key: info[key]
            for key in (
                "elapsed_s",
                "generated_tasks",
                "completed_tasks",
                "failed_tasks",
                "pending_tasks",
                "total_reward",
                "total_task_energy_j",
            )
        }
        results.update(steps=step_count, terminated=terminated, truncated=truncated)
        summary = {
            "output_schema_version": 2,
            "snapshot_timing": (
                "Satellite geographic coordinates, user presence, sunlight, contact counts and "
                "battery_voltage_start_v describe start_s. Battery state fields describe end_s; "
                "battery_current_a is the interval average."
            ),
            "config": config.to_dict(),
            "seed": args.seed,
            "policy": args.policy,
            "versions": _versions(),
            "start_time_utc": env.start_time.isoformat(),
            "end_time_utc": info["time_utc"],
            "satellite_ids": list(env.orbits.satellite_ids),
            "constellation_size": env.orbits.constellation_size,
            "user_locations_deg": env.workload.locations_deg.tolist(),
            "user_task_rates_per_minute": (env.workload.rates * 60).tolist(),
            "results": results,
            "final_soc": observation["battery_soc"].tolist(),
        }
        _write_json(args.output_dir / "summary.json", summary)
        print(
            f"Completed {step_count} steps: {info['completed_tasks']} tasks completed, "
            f"{info['failed_tasks']} failed, reward {info['total_reward']:g}."
        )
        print(f"Summary: {args.output_dir / 'summary.json'}")
    finally:
        env.close()


def _contacts(args: argparse.Namespace, config: SimulationConfig) -> None:
    model = OrbitModel(config.tle_path, config.satellite_ids)
    raw_start = config.start_time or model.default_start_time
    start = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise ValueError("start_time must include a UTC offset, for example ...Z")
    start = start.astimezone(timezone.utc)
    workload = Workload(config, np.random.default_rng(args.seed))
    steps = math.floor(config.duration_s / config.step_s)
    rows = export_contact_table(
        model,
        workload.locations_deg,
        start,
        config.step_s,
        steps,
        config.min_elevation_deg,
        args.output,
    )
    metadata = {
        "config": config.to_dict(),
        "seed": args.seed,
        "versions": _versions(),
        "start_time_utc": start.isoformat(),
        "satellite_ids": list(model.satellite_ids),
        "constellation_size": model.constellation_size,
        "user_locations_deg": workload.locations_deg.tolist(),
        "user_task_rates_per_minute": (workload.rates * 60).tolist(),
        "user_id_convention": "Zero-based index into user_locations_deg",
        "sampling": "t = 0, step, ..., floor(duration / step) * step; geometric visibility only",
        "sample_count": steps + 1,
        "sampled_end_s": steps * config.step_s,
        "visible_rows": rows,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    _write_json(metadata_path, metadata)
    print(f"Exported {rows} visible contact rows across {steps + 1} samples: {args.output}")
    print(f"Metadata: {metadata_path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; invalid inputs produce argparse errors without tracebacks."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _validate_plot_arguments(args)
        config = _configuration(args)
        if args.command == "run":
            _run(args, config)
            if args.plot is not None:
                try:
                    from .plots.plot_results import plot_dataset

                    plot_dataset(
                        args.output_dir,
                        mode=args.plot,
                        interval_ms=args.plot_interval,
                        save=args.plot_save,
                        show=not args.plot_no_show,
                    )
                except ImportError as error:
                    parser.error(
                        "Unable to load plotting support; install MeteoGym's plot dependencies "
                        f"with python -m pip install 'meteogym[plot]': {error}"
                    )
                except RuntimeError as error:
                    parser.error(f"Unable to display or export the plot: {error}")
        else:
            _contacts(args, config)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        parser.error(str(error))
    return 0
