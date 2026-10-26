"""Ground users with gated or independently queued exponential task arrivals."""

from collections import deque
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
        self.queues: list[deque[Task]] = [deque() for _ in range(count)]
        self.next_arrival_s = np.array([self._wait(i) for i in range(count)])
        self.generated = 0
        if config.initial_tasks:
            if config.task_arrival_mode == "single":
                self.next_arrival_s[:] = 0
            else:
                for user in range(count):
                    self._append_task(user, 0.0)
        self.generate_due(0.0)

    @property
    def tasks(self) -> list[Task | None]:
        """The first pending task for each user, including single-task workloads."""
        return [queue[0] if queue else None for queue in self.queues]

    @property
    def pending_count(self) -> int:
        return sum(len(queue) for queue in self.queues)

    def _wait(self, user: int) -> float:
        return float(self.rng.exponential(1.0 / self.rates[user])) if self.rates[user] else np.inf

    def _append_task(self, user: int, created_s: float) -> None:
        size = float(self.rng.uniform(*self.config.task_size_mb))
        deadline = created_s + float(self.rng.uniform(*self.config.task_deadline_s))
        self.queues[user].append(Task(self.generated, size, size, created_s, deadline))
        self.generated += 1

    def generate_due(self, now_s: float) -> int:
        """Admit due arrivals; queued tasks retain their sampled creation time."""
        generated = 0
        for user in np.flatnonzero(self.online & (self.next_arrival_s <= now_s)):
            if self.config.task_arrival_mode == "single":
                if self.queues[user]:
                    continue
                self._append_task(user, now_s)
                generated += 1
                self.next_arrival_s[user] = np.inf
            else:
                while self.next_arrival_s[user] <= now_s:
                    created_s = float(self.next_arrival_s[user])
                    self._append_task(user, created_s)
                    generated += 1
                    self.next_arrival_s[user] = max(
                        created_s + self._wait(user), np.nextafter(created_s, np.inf)
                    )
        return generated

    def finish(self, user: int, now_s: float, task: Task | None = None) -> Task:
        """Remove a task; only single-task mode restarts the arrival clock."""
        queue = self.queues[user]
        if not queue:
            raise RuntimeError("cannot finish a user without a pending task")
        if task is None:
            task = queue.popleft()
        else:
            queue.remove(task)
        if self.config.task_arrival_mode == "single":
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
            while self.queues[user]:
                failed.append((int(user), self.finish(user, now_s)))
            self.next_arrival_s[user] = np.inf
        self.online[returning] = True
        for user in np.flatnonzero(returning):
            self.next_arrival_s[user] = now_s + self._wait(user)
        return failed
