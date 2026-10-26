"""Vectorized internal-resistance batteries with linear voltage and charge counting.

Positive battery current means discharge. The defaults describe an illustrative
2S1P CubeSat pack, not a calibrated model of SUCHAI-3. See docs/energy-model.md.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BatteryConfig:
    """Pack-level values; active current includes the passive bus load."""

    capacity_ah: float = 2.6
    voltage_empty_v: float = 5.5
    voltage_full_v: float = 8.4
    internal_resistance_ohm: float = 0.14
    passive_current_a: float = 0.12
    active_current_a: float = 0.45
    solar_current_a: float = 0.65
    max_discharge_current_a: float = 2.5
    charge_efficiency: float = 0.95
    cutoff_soc: float = 0.20
    resume_soc: float = 0.30
    minimum_voltage_v: float | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "minimum_voltage_v" and value is None:
                continue
            if isinstance(value, (bool, str)) or not np.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
        if self.capacity_ah <= 0:
            raise ValueError("capacity_ah must be positive")
        if not 0 < self.voltage_empty_v < self.voltage_full_v:
            raise ValueError("voltage limits must satisfy 0 < empty < full")
        if self.internal_resistance_ohm < 0:
            raise ValueError("internal_resistance_ohm must be nonnegative")
        if not 0 <= self.passive_current_a <= self.active_current_a:
            raise ValueError("currents must satisfy 0 <= passive <= active")
        if self.solar_current_a < 0:
            raise ValueError("solar_current_a must be nonnegative")
        if not 0 < self.max_discharge_current_a or (
            self.active_current_a > self.max_discharge_current_a
        ):
            raise ValueError("max_discharge_current_a must cover the active load")
        if self.internal_resistance_ohm * self.max_discharge_current_a >= (
            self.voltage_empty_v / 2
        ):
            raise ValueError("resistance/current limit would cause excessive voltage sag")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0 <= self.cutoff_soc < self.resume_soc <= 1:
            raise ValueError("SoC thresholds must satisfy 0 <= cutoff < resume <= 1")
        if not 0 < self.effective_minimum_voltage_v <= self.voltage_full_v:
            raise ValueError("minimum_voltage_v must be positive and no greater than full voltage")

    @property
    def capacity_coulomb(self) -> float:
        return self.capacity_ah * 3600.0

    @property
    def effective_minimum_voltage_v(self) -> float:
        """Loaded-voltage cutoff for queued transfers; defaults to empty voltage."""
        return self.voltage_empty_v if self.minimum_voltage_v is None else self.minimum_voltage_v


@dataclass(frozen=True)
class BatteryStep:
    """Independent snapshots of the interval outcome, with one entry per pack.

    ``active`` describes the mode during this interval. A pack that reaches its
    cutoff at the end is locked out for the next interval. ``current_a`` is the
    interval-average battery current, accounting for saturation at 0/100% SoC.
    ``voltage_v`` uses ending SoC and that reported current.
    """

    active: NDArray[np.bool_]
    charging: NDArray[np.bool_]
    soc: NDArray[np.float64]
    voltage_v: NDArray[np.float64]
    current_a: NDArray[np.float64]
    task_energy_j: NDArray[np.float64]


@dataclass(frozen=True)
class TransferStep:
    """One scalar transfer interval, possibly ending early at a voltage cutoff."""

    duration_s: float
    task_energy_j: float
    active: bool
    failed: bool


class BatteryPack:
    """A collection of independent batteries sharing one configuration.

    Admission sequence: call ``plan_active`` and ``task_energy_budget_j``, admit
    at most that much task work, then call ``advance`` with exactly that work.
    Planning does not mutate state. A failed advance does not consume charge.
    """

    def __init__(self, config: BatteryConfig, count: int, initial_soc: ArrayLike = 0.8) -> None:
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("count must be a positive integer")
        self.config = config
        self.count = int(count)
        self.soc = self._vector(initial_soc, "initial_soc")
        if np.any((self.soc < 0) | (self.soc > 1)):
            raise ValueError("initial_soc must be in [0, 1]")
        self.locked_out = self.soc <= config.cutoff_soc
        self.current_a = np.zeros(self.count, dtype=np.float64)
        self.voltage_v = self.open_circuit_voltage_v()
        self.charging = np.zeros(self.count, dtype=np.bool_)

    def _vector(self, value: ArrayLike, name: str) -> NDArray[np.float64]:
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:
            result = np.full(self.count, result.item(), dtype=np.float64)
        if result.shape != (self.count,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite and scalar or shape ({self.count},)")
        return result.copy()

    def _flags(self, value: ArrayLike, name: str) -> NDArray[np.bool_]:
        result = self._vector(value, name)
        if np.any((result != 0) & (result != 1)):
            raise ValueError(f"{name} must contain only booleans or 0/1")
        return result.astype(np.bool_)

    @staticmethod
    def _validate_dt(dt_s: float) -> None:
        if isinstance(dt_s, bool) or not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")

    def open_circuit_voltage_v(self) -> NDArray[np.float64]:
        c = self.config
        return c.voltage_empty_v + (c.voltage_full_v - c.voltage_empty_v) * self.soc

    def available(self, requested_active: ArrayLike) -> NDArray[np.bool_]:
        """Return instantaneous availability; use plan_active for a full step."""
        return (
            self._flags(requested_active, "requested_active")
            & ~self.locked_out
            & (self.soc > self.config.cutoff_soc)
        )

    def _trajectory(
        self, dt_s: float, current_a: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Ending SoC, exact mean SoC, and actual mean current with clipping."""
        c = self.config
        efficiency = np.where(current_a < 0, c.charge_efficiency, 1.0)
        delta = -current_a * efficiency * dt_s / c.capacity_coulomb
        end = np.clip(self.soc + delta, 0.0, 1.0)
        # A constant current makes SoC affine until it reaches a capacity bound.
        fraction = np.ones(self.count, dtype=np.float64)
        np.divide(end - self.soc, delta, out=fraction, where=delta != 0)
        fraction = np.clip(fraction, 0.0, 1.0)
        mean_soc = (self.soc + end) * 0.5 * fraction + end * (1.0 - fraction)
        return end, mean_soc, current_a * fraction

    def plan_active(
        self, dt_s: float, sunlit: ArrayLike, requested_active: ArrayLike
    ) -> NDArray[np.bool_]:
        """Only enable packs able to afford the entire active baseline interval."""
        self._validate_dt(dt_s)
        solar = self._flags(sunlit, "sunlit") * self.config.solar_current_a
        baseline = self.config.active_current_a - solar
        efficiency = np.where(baseline < 0, self.config.charge_efficiency, 1.0)
        # Check the unconstrained trajectory: clipping to zero would otherwise
        # admit an unaffordable step when cutoff_soc is configured as zero.
        end_soc = self.soc - baseline * efficiency * dt_s / self.config.capacity_coulomb
        return self.available(requested_active) & (end_soc >= self.config.cutoff_soc)

    def _task_current_limit(
        self, dt_s: float, sunlit: NDArray[np.bool_], active: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        c = self.config
        # Reserve the baseline load and allow task work only above the cutoff.
        reserve_current = (self.soc - c.cutoff_soc) * c.capacity_coulomb / dt_s
        max_net_current = np.minimum(reserve_current, c.max_discharge_current_a)
        baseline = c.active_current_a - sunlit * c.solar_current_a
        limit = np.maximum(0.0, max_net_current - baseline)
        # Keep energy strictly increasing in task current, including large dt.
        voltage_slope = c.voltage_full_v - c.voltage_empty_v
        sag_slope = c.internal_resistance_ohm + voltage_slope * dt_s / (2.0 * c.capacity_coulomb)
        min_voltage = c.voltage_empty_v - (c.internal_resistance_ohm * c.max_discharge_current_a)
        limit = np.minimum(limit, min_voltage / (2.0 * sag_slope))
        return np.where(active, limit, 0.0)

    def _task_energy(
        self, dt_s: float, baseline_a: NDArray[np.float64], task_current_a: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        c = self.config
        _, mean_soc, mean_current = self._trajectory(dt_s, baseline_a + task_current_a)
        mean_voltage = (
            c.voltage_empty_v
            + (c.voltage_full_v - c.voltage_empty_v) * mean_soc
            - c.internal_resistance_ohm * mean_current
        )
        return task_current_a * mean_voltage * dt_s

    def task_energy_budget_j(
        self, dt_s: float, sunlit: ArrayLike, requested_active: ArrayLike
    ) -> NDArray[np.float64]:
        """Maximum additional task work in joules, after baseline reservation."""
        active = self.plan_active(dt_s, sunlit, requested_active)
        sun = self._flags(sunlit, "sunlit")
        limit = self._task_current_limit(dt_s, sun, active)
        baseline = self.config.active_current_a - sun * self.config.solar_current_a
        return self._task_energy(dt_s, baseline, limit)

    def advance(
        self,
        dt_s: float,
        sunlit: ArrayLike,
        requested_active: ArrayLike,
        task_energy_j: ArrayLike | None = None,
    ) -> BatteryStep:
        """Advance one interval, rejecting work that exceeds the prior budget.

        Task energy is terminal electrical work. A constant extra task current
        is found using the exact mean voltage along the linear SoC trajectory.
        Solar generation feeds the bus before charging the battery. At full
        charge excess solar current is curtailed; passive bus power at empty
        charge is unserved. Processing always reserves the configured cutoff.
        """
        active = self.plan_active(dt_s, sunlit, requested_active)
        sun = self._flags(sunlit, "sunlit")
        energy = self._vector(0.0 if task_energy_j is None else task_energy_j, "task_energy_j")
        if np.any(energy < 0):
            raise ValueError("task_energy_j must be nonnegative")
        c = self.config
        baseline = np.where(active, c.active_current_a, c.passive_current_a)
        baseline = baseline - sun * c.solar_current_a
        high = self._task_current_limit(dt_s, sun, active)
        budget = self._task_energy(dt_s, baseline, high)
        tolerance = 1e-9 * np.maximum(1.0, budget)
        if np.any(energy > budget + tolerance) or np.any((~active) & (energy > 0)):
            raise ValueError("task energy exceeds the available battery budget")
        energy = np.minimum(energy, budget)
        low = np.zeros(self.count, dtype=np.float64)
        if np.any(energy > 0):
            # All satellites solve together; 48 iterations exceed float32 observation precision.
            for _ in range(48):
                middle = (low + high) * 0.5
                below = self._task_energy(dt_s, baseline, middle) < energy
                low = np.where(below, middle, low)
                high = np.where(below, high, middle)
        task_current = np.where(energy > 0, (low + high) * 0.5, 0.0)
        old_soc = self.soc.copy()
        self.soc, _, self.current_a = self._trajectory(dt_s, baseline + task_current)
        self.charging = self.soc > old_soc
        self.voltage_v = self.open_circuit_voltage_v() - (
            self.current_a * c.internal_resistance_ohm
        )
        self.locked_out |= self.soc <= c.cutoff_soc + 1e-12
        self.locked_out &= self.soc < c.resume_soc
        return BatteryStep(
            active=active.copy(),
            charging=self.charging.copy(),
            soc=self.soc.copy(),
            voltage_v=self.voltage_v.copy(),
            current_a=self.current_a.copy(),
            task_energy_j=energy.copy(),
        )

    def advance_transfer(
        self,
        dt_s: float,
        sunlit: bool,
        requested_active: bool,
        task_power_w: float,
    ) -> TransferStep:
        """Advance a single pack until the requested end or its loaded-voltage cutoff.

        Queued transfers bypass the legacy SoC reserve and lockout: the caller
        controls suspension and recovery after ``failed``. The requested task
        power is represented by a constant extra current whose integrated work
        matches that power over the actual interval, including early cutoff.
        A hardware current cap (and the monotone branch of the work/current
        relation) can throttle it.
        If cutoff occurs first, only work up to that instant is delivered.

        Terminal voltage includes the instantaneous resistive sag, so a newly
        applied load can fail at time zero. Complete depletion also stops work
        when a custom cutoff lies below the model's loaded empty voltage.
        Passive intervals always run to their end without reporting failure.
        """
        if self.count != 1:
            raise ValueError("advance_transfer requires a single battery pack")
        self._validate_dt(dt_s)
        sun = bool(self._flags(sunlit, "sunlit")[0])
        active = bool(self._flags(requested_active, "requested_active")[0])
        if (
            isinstance(task_power_w, (bool, str))
            or not np.isfinite(task_power_w)
            or task_power_w < 0
        ):
            raise ValueError("task_power_w must be finite and nonnegative")
        if not active:
            if task_power_w > 0:
                raise ValueError("task power must be zero when processing is passive")
            self.advance(dt_s, sun, False)
            return TransferStep(float(dt_s), 0.0, False, False)

        c = self.config
        baseline = np.array([c.active_current_a - sun * c.solar_current_a])
        slope = c.voltage_full_v - c.voltage_empty_v
        # Unlike task_energy_budget_j, do not reserve charge above cutoff_soc.
        # This conservative derivative bound keeps bisection on the increasing
        # work/current branch even across capacity saturation or large steps.
        sag_slope = c.internal_resistance_ohm + slope * dt_s / (2 * c.capacity_coulomb)
        min_voltage = c.voltage_empty_v - c.internal_resistance_ohm * c.max_discharge_current_a
        limit = min(
            max(0.0, c.max_discharge_current_a - baseline[0]),
            min_voltage / (2 * sag_slope),
        )
        initial_voltage = float(self.open_circuit_voltage_v()[0])

        def cutoff_time(extra_current: float) -> float:
            net_current = baseline[0] + extra_current
            starting_voltage = initial_voltage - net_current * c.internal_resistance_ohm
            if starting_voltage <= c.effective_minimum_voltage_v:
                return 0.0
            if net_current <= 0:
                return float("inf")
            voltage_time = (
                (starting_voltage - c.effective_minimum_voltage_v)
                * c.capacity_coulomb / (slope * net_current)
            )
            empty_time = self.soc[0] * c.capacity_coulomb / net_current
            return float(min(voltage_time, empty_time))

        def mean_task_power(extra_current: float) -> float:
            duration = min(dt_s, cutoff_time(extra_current))
            if duration == 0:
                # Use the instantaneous limit to solve a load that causes an
                # immediate sag; that failed interval still delivers no work.
                voltage = initial_voltage - (baseline[0] + extra_current) * (
                    c.internal_resistance_ohm
                )
                return extra_current * voltage
            work = self._task_energy(duration, baseline, np.array([extra_current]))[0]
            return float(work / duration)

        low, high = 0.0, limit
        if task_power_w > 0:
            target_power = min(float(task_power_w), mean_task_power(high))
            # Couple the cutoff time to each candidate current. Solving over
            # the original interval would overdeliver power before early cutoff.
            for _ in range(48):
                middle = (low + high) * 0.5
                if mean_task_power(middle) <= target_power:
                    low = middle
                else:
                    high = middle
        task_current = np.array([low])
        current = baseline + task_current
        failure_time = cutoff_time(low)
        failed = failure_time <= dt_s
        duration = float(min(dt_s, failure_time))
        old_soc = self.soc.copy()
        if duration > 0:
            energy = float(self._task_energy(duration, baseline, task_current)[0])
            self.soc, _, self.current_a = self._trajectory(duration, current)
        else:
            energy = 0.0
            # A zero-duration sag consumes no charge, but exposes its voltage.
            self.current_a = current.copy()
        self.charging = self.soc > old_soc
        self.voltage_v = self.open_circuit_voltage_v() - self.current_a * c.internal_resistance_ohm
        self.locked_out |= self.soc <= c.cutoff_soc + 1e-12
        self.locked_out &= self.soc < c.resume_soc
        return TransferStep(duration, energy, True, bool(failed))
