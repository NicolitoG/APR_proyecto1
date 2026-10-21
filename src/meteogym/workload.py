"""Ground users and gated exponential task arrivals, without a network emulator."""

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass
class Task:
    task_id: int
    size_mb: float
    remaining_mb: float
    created_s: float
    deadline_s: float
    satellite_index: int | None = None


class Workload:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        count = config.num_users
        if config.user_locations_deg is None:
            lat_min, lat_max, lon_min, lon_max = config.user_region_deg
            sin_lat = rng.uniform(np.sin(np.deg2rad(lat_min)), np.sin(np.deg2rad(lat_max)), count)
            self.locations_deg = np.column_stack(
                (np.rad2deg(np.arcsin(sin_lat)), rng.uniform(lon_min, lon_max, count))
            )
        else:
            self.locations_deg = np.array(config.user_locations_deg, dtype=float)
        self.rates = rng.uniform(*config.task_rate_per_minute, count) / 60.0
        self.online = np.ones(count, dtype=bool)
        self.tasks: list[Task | None] = [None] * count
        self.next_arrival_s = np.array([self._wait(i) for i in range(count)])
        self.generated = 0
        if config.initial_tasks:
            self.next_arrival_s[:] = 0
        self.generate_due(0.0)

    def _wait(self, user: int) -> float:
        return float(self.rng.exponential(1.0 / self.rates[user])) if self.rates[user] else np.inf

    def generate_due(self, now_s: float) -> int:
        """Admit arrivals at a step boundary; deadlines start at admission."""
        generated = 0
        for user in np.flatnonzero(self.online & (self.next_arrival_s <= now_s)):
            if self.tasks[user] is not None:
                continue
            size = float(self.rng.uniform(*self.config.task_size_mb))
            deadline = now_s + float(self.rng.uniform(*self.config.task_deadline_s))
            self.tasks[user] = Task(self.generated, size, size, now_s, deadline)
            self.generated += 1
            generated += 1
            self.next_arrival_s[user] = np.inf
        return generated

    def finish(self, user: int, now_s: float) -> Task:
        task = self.tasks[user]
        if task is None:
            raise RuntimeError("cannot finish a user without a pending task")
        self.tasks[user] = None
        self.next_arrival_s[user] = now_s + self._wait(user) if self.online[user] else np.inf
        return task

    def update_presence(self, now_s: float, dt_s: float) -> list[tuple[int, Task]]:
        """At most one presence transition per user per step; departures fail tasks."""
        old_online = self.online.copy()
        leave_p = -np.expm1(-self.config.user_departure_rate_per_minute * dt_s / 60)
        return_p = -np.expm1(-self.config.user_return_rate_per_minute * dt_s / 60)
        leaving = old_online & (self.rng.random(len(old_online)) < leave_p)
        returning = ~old_online & (self.rng.random(len(old_online)) < return_p)
        self.online[leaving] = False
        failed = []
        for user in np.flatnonzero(leaving):
            if self.tasks[user] is not None:
                failed.append((int(user), self.finish(user, now_s)))
            self.next_arrival_s[user] = np.inf
        self.online[returning] = True
        for user in np.flatnonzero(returning):
            self.next_arrival_s[user] = now_s + self._wait(user)
        return failed
