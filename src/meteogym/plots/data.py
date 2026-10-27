"""Load saved runs without confusing interval-start and interval-end state."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..config import SimulationConfig

if TYPE_CHECKING:
    from ..orbits import OrbitModel


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """Start state, effective activity, and outcomes through the interval end.

    Charging describes net charging over this interval; task counts and total
    reward are cumulative through its end. Missing historical values stay None.
    """

    step: int
    elapsed_s: float
    time_utc: datetime
    satellite_locations_deg: NDArray[np.float64]
    user_online: NDArray[np.bool_]
    contact_count: NDArray[np.int64]
    active: NDArray[np.bool_]
    voltage_v: NDArray[np.float64]
    reward: float | None
    generated_tasks: int | None = None
    completed_tasks: int | None = None
    failed_tasks: int | None = None
    total_reward: float | None = None
    charging: NDArray[np.bool_] | None = None
    battery_suspended: NDArray[np.bool_] | None = None

    @property
    def start_s(self) -> float:
        """Alias making the interval-start timestamp explicit."""
        return self.elapsed_s


class ReplayDataset:
    """Validated replay metadata with cached, lazily propagated legacy frames."""

    def __init__(
        self,
        satellite_ids: tuple[int, ...],
        user_locations_deg: NDArray[np.float64],
        voltage_limits: tuple[float, float],
        rows: list[dict],
        orbits: OrbitModel | None = None,
    ) -> None:
        self.satellite_ids = satellite_ids
        self.user_locations_deg = user_locations_deg
        self.voltage_limits = voltage_limits
        self._rows = rows
        self._orbits = orbits
        self._cache: dict[int, ReplayFrame] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def frame(self, index: int) -> ReplayFrame:
        """Return a frame by zero-based index, accepting Python negative indices."""
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("frame index must be an integer")
        if not -len(self) <= index < len(self):
            raise IndexError("frame index out of range")
        index = int(index) % len(self)
        if index not in self._cache:
            values = self._rows[index].copy()
            if values["satellite_locations_deg"] is None:
                assert self._orbits is not None
                try:
                    locations = self._orbits.snapshot(values["time_utc"]).geographic[:, :2]
                except (ValueError, OSError) as exc:
                    raise ValueError(
                        f"Cannot reconstruct satellite positions for step {values['step']}: "
                        f"{exc}. Restore the original TLE or generate a new run."
                    ) from exc
                values["satellite_locations_deg"] = _locations(
                    locations, (len(self.satellite_ids), 2), "reconstructed satellite positions"
                )
            self._cache[index] = ReplayFrame(**values)
        return self._cache[index]


def _required(values: dict, key: str, context: str):
    if key not in values:
        raise ValueError(f"{context}: missing {key}")
    return values[key]


def _array(values, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    try:
        result = np.asarray(values)
        if result.dtype.kind not in "biuf":
            raise ValueError("expected numeric values")
        result = result.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite numeric array of shape {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite numeric array of shape {shape}")
    result.setflags(write=False)
    return result


def _locations(values, shape: tuple[int, int], label: str) -> NDArray[np.float64]:
    result = _array(values, shape, label)
    if np.any(np.abs(result[:, 0]) > 90) or np.any(np.abs(result[:, 1]) > 180):
        raise ValueError(f"{label} must contain latitude in [-90, 90] and longitude in [-180, 180]")
    return result


def _binary(values, count: int, label: str) -> NDArray[np.bool_]:
    numbers = _array(values, (count,), label)
    if not np.isin(numbers, [0, 1]).all():
        raise ValueError(f"{label} must contain only boolean or binary values")
    result = numbers.astype(np.bool_)
    result.setflags(write=False)
    return result


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _task_count(value, label: str) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _json(text: str, label: str) -> dict:
    try:
        result = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: invalid JSON ({exc})") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return result


def load_dataset(path: str | Path) -> ReplayDataset:
    """Read a directory containing ``summary.json`` and ``steps.jsonl``.

    Schema 2 records the complete start state needed by the viewer.
    Older runs reconstruct coordinates from the original TLE and obtain start
    voltage from the previous end voltage, or initial open-circuit voltage.
    Historical user presence cannot be recovered when departures were enabled.
    """
    path = Path(path)
    try:
        summary = _json((path / "summary.json").read_text(encoding="utf-8"), "summary.json")
        lines = (path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(
            f"Cannot read dataset {path}: {exc}. Expected summary.json and steps.jsonl."
        ) from exc
    schema = summary.get("output_schema_version", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("summary.json: output_schema_version must be 1 or 2")
    config_values = _required(summary, "config", "summary.json")
    if not isinstance(config_values, dict):
        raise ValueError("summary.json: config must be an object")
    try:
        config = SimulationConfig.from_dict(config_values)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"summary.json: invalid simulation config ({exc})") from exc
    ids = _required(summary, "satellite_ids", "summary.json")
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(value) is not int or value <= 0 for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("summary.json: satellite_ids must contain unique positive integers")
    satellite_ids = tuple(ids)
    if config.satellite_ids is not None and satellite_ids != config.satellite_ids:
        raise ValueError("summary.json: satellite_ids do not match config selection/order")
    users = _locations(
        _required(summary, "user_locations_deg", "summary.json"),
        (config.num_users, 2),
        "summary.json user_locations_deg",
    )
    try:
        start = datetime.fromisoformat(_required(summary, "start_time_utc", "summary.json"))
        if start.tzinfo is None:
            raise ValueError("UTC offset is required")
        start = start.astimezone(timezone.utc)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"summary.json: invalid start_time_utc ({exc})") from exc
    voltage_limits = (config.battery.voltage_empty_v, config.battery.voltage_full_v)
    previous_voltage = np.full(
        len(ids), voltage_limits[0] + config.initial_soc * (voltage_limits[1] - voltage_limits[0])
    )
    previous_suspended = _binary(
        previous_voltage <= config.battery.effective_minimum_voltage_v
        if config.task_arrival_mode == "queued"
        else np.full(len(ids), config.initial_soc <= config.battery.cutoff_soc),
        len(ids),
        "initial battery_suspended",
    )
    previous_end = 0.0
    previous_step = 0
    generated_tasks = None
    completed_tasks = failed_tasks = 0
    total_reward = 0.0
    rows = []
    reconstruct_positions = False
    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        context = f"steps.jsonl line {line_number}"
        row = _json(text, context)
        step = _required(row, "step", context)
        if type(step) is not int or step <= previous_step:
            raise ValueError(f"{context}: step must be a positive, strictly increasing integer")
        elapsed = _number(_required(row, "start_s", context), f"{context} start_s")
        end = _number(_required(row, "end_s", context), f"{context} end_s")
        if elapsed < 0 or end <= elapsed:
            raise ValueError(f"{context}: times must satisfy 0 <= start_s < end_s")
        if not math.isclose(elapsed, previous_end, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"{context}: interval times must be contiguous and start at zero")
        if schema == 2:
            for name in ("satellite_geographic_deg_km", "user_online", "battery_voltage_start_v"):
                _required(row, name, context)
        positions = None
        if "satellite_geographic_deg_km" in row:
            positions = _locations(
                row["satellite_geographic_deg_km"],
                (len(ids), 3),
                f"{context} satellite_geographic_deg_km",
            )[:, :2]
        else:
            reconstruct_positions = True
        if "user_online" in row:
            online = _binary(row["user_online"], config.num_users, f"{context} user_online")
        elif config.user_departure_rate_per_minute == 0:
            online = _binary(np.ones(config.num_users), config.num_users, "user_online")
        else:
            raise ValueError(
                f"{context}: user_online was not saved and user departures were enabled. "
                "Historical presence cannot be recovered; generate a new run with current MeteoGym."
            )
        counts = _array(
            _required(row, "contact_count", context), (len(ids),), f"{context} contact_count"
        )
        if (
            np.any(counts < 0)
            or np.any(counts > online.sum())
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError(
                f"{context}: contact_count must be integers between zero and online users"
            )
        counts = counts.astype(np.int64)
        counts.setflags(write=False)
        voltage = _array(
            row.get("battery_voltage_start_v", previous_voltage),
            (len(ids),),
            f"{context} battery_voltage_start_v",
        )
        previous_voltage = _array(
            _required(row, "battery_voltage_v", context),
            (len(ids),),
            f"{context} battery_voltage_v",
        )
        if np.any(voltage <= 0) or np.any(previous_voltage <= 0):
            raise ValueError(f"{context}: battery voltages must be positive")
        reward = row.get("reward")
        if reward is not None:
            reward = _number(reward, f"{context} reward")
        total_reward = (
            total_reward + reward if total_reward is not None and reward is not None else None
        )
        generated = _task_count(row.get("generated_tasks"), f"{context} generated_tasks")
        if generated is not None and generated_tasks is not None and generated < generated_tasks:
            raise ValueError(f"{context}: generated_tasks cannot decrease")
        generated_tasks = generated
        completed = _task_count(row.get("completed"), f"{context} completed")
        failed = _task_count(row.get("failed"), f"{context} failed")
        completed_tasks = (
            completed_tasks + completed
            if completed_tasks is not None and completed is not None
            else None
        )
        failed_tasks = (
            failed_tasks + failed if failed_tasks is not None and failed is not None else None
        )
        charging = (
            _binary(row["charging"], len(ids), f"{context} charging")
            if row.get("charging") is not None
            else None
        )
        try:
            time_utc = start + timedelta(seconds=elapsed)
        except OverflowError as exc:
            raise ValueError(f"{context}: start_s exceeds the supported datetime range") from exc
        rows.append(
            dict(
                step=step,
                elapsed_s=elapsed,
                time_utc=time_utc,
                satellite_locations_deg=positions,
                user_online=online,
                contact_count=counts,
                active=_binary(
                    _required(row, "effective_active", context),
                    len(ids),
                    f"{context} effective_active",
                ),
                voltage_v=voltage,
                reward=reward,
                generated_tasks=generated_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                total_reward=total_reward,
                charging=charging,
                battery_suspended=previous_suspended,
            )
        )
        # Saved suspension is interval-end state; use it at the next frame's start.
        previous_suspended = (
            _binary(row["battery_suspended"], len(ids), f"{context} battery_suspended")
            if row.get("battery_suspended") is not None
            else None
        )
        previous_step, previous_end = step, end
    if not rows:
        raise ValueError("steps.jsonl contains no frames; run an episode before plotting")
    orbits = None
    if reconstruct_positions:
        from ..orbits import OrbitModel

        try:
            orbits = OrbitModel(config.tle_path, satellite_ids)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot reconstruct legacy satellite positions: {exc}. "
                "Restore the original TLE at config.tle_path, or generate a new run that saves "
                "satellite_geographic_deg_km."
            ) from exc
        warnings.warn(
            "Legacy dataset: satellite positions are reconstructed from the configured TLE "
            "using the currently installed astronomy libraries; they may differ from the "
            "original run. New runs save coordinates directly.",
            UserWarning,
            stacklevel=2,
        )
    return ReplayDataset(satellite_ids, users, voltage_limits, rows, orbits)
