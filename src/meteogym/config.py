"""Validated, serializable experiment settings. Time is measured in seconds."""

import math
from dataclasses import asdict, dataclass, field

from .battery import BatteryConfig


@dataclass(frozen=True)
class SimulationConfig:
    satellite_ids: tuple[int, ...] | None = (1, 2)
    tle_path: str | None = None
    start_time: str | None = None
    duration_s: float = 5400.0
    step_s: float = 10.0
    num_users: int = 100
    # Bounds: latitude min/max, longitude min/max; uniform surface-area sampling.
    user_region_deg: tuple[float, float, float, float] = (-90.0, 90.0, -180.0, 180.0)
    user_locations_deg: tuple[tuple[float, float], ...] | None = None
    task_rate_per_minute: tuple[float, float] = (1.0, 3.0)
    task_size_mb: tuple[float, float] = (0.1, 2.0)
    task_deadline_s: tuple[float, float] = (60.0, 180.0)
    initial_tasks: bool = False
    min_elevation_deg: float = 0.0
    uplink_mbps: float = 1.0
    task_energy_j_per_mb: float = 2.0
    initial_soc: float = 0.8
    # Optional user presence transitions; zero keeps every user present.
    user_departure_rate_per_minute: float = 0.0
    user_return_rate_per_minute: float = 0.0
    battery: BatteryConfig = field(default_factory=BatteryConfig)

    def __post_init__(self):
        if self.satellite_ids is not None:
            ids = tuple(self.satellite_ids)
            if not ids or any(type(i) is not int or i <= 0 for i in ids):
                raise ValueError("satellite_ids must contain positive integer source IDs")
            if len(set(ids)) != len(ids):
                raise ValueError("satellite_ids must be unique")
            object.__setattr__(self, "satellite_ids", ids)
        if type(self.num_users) is not int or self.num_users < 1:
            raise ValueError("num_users must be a positive integer")
        for name in ("duration_s", "step_s", "uplink_mbps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "task_energy_j_per_mb",
            "user_departure_rate_per_minute",
            "user_return_rate_per_minute",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("task_rate_per_minute", "task_size_mb", "task_deadline_s"):
            bounds = tuple(getattr(self, name))
            minimum = 0 if name == "task_rate_per_minute" else 1e-12
            if (
                len(bounds) != 2
                or not all(math.isfinite(x) for x in bounds)
                or not minimum <= bounds[0] <= bounds[1]
            ):
                raise ValueError(f"{name} must be an ordered, finite pair >= {minimum}")
            object.__setattr__(self, name, bounds)
        if not math.isfinite(self.initial_soc) or not 0 <= self.initial_soc <= 1:
            raise ValueError("initial_soc must be in [0, 1]")
        if not math.isfinite(self.min_elevation_deg) or not 0 <= self.min_elevation_deg <= 90:
            raise ValueError("min_elevation_deg must be in [0, 90]")
        region = tuple(self.user_region_deg)
        if (
            len(region) != 4
            or not all(math.isfinite(x) for x in region)
            or not -90 <= region[0] <= region[1] <= 90
            or not -180 <= region[2] <= region[3] <= 180
        ):
            raise ValueError("user_region_deg must be (lat_min, lat_max, lon_min, lon_max)")
        object.__setattr__(self, "user_region_deg", region)
        if self.user_locations_deg is not None:
            locations = tuple(tuple(x) for x in self.user_locations_deg)
            if len(locations) != self.num_users:
                raise ValueError("user_locations_deg must contain num_users coordinate pairs")
            for location in locations:
                if (
                    len(location) != 2
                    or not all(math.isfinite(x) for x in location)
                    or not -90 <= location[0] <= 90
                    or not -180 <= location[1] <= 180
                ):
                    raise ValueError("user locations must be valid latitude/longitude pairs")
            object.__setattr__(self, "user_locations_deg", locations)
        if not isinstance(self.battery, BatteryConfig):
            raise TypeError("battery must be a BatteryConfig")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "SimulationConfig":
        values = dict(values)
        if "battery" in values:
            values["battery"] = BatteryConfig(**values["battery"])
        return cls(**values)
