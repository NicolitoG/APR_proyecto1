"""Physical and admission invariants for the CubeSat energy model."""

from dataclasses import replace

import numpy as np
import pytest

from meteogym.battery import BatteryConfig, BatteryPack


def test_constant_mode_gives_linear_discharge_and_voltage_sag():
    c = BatteryConfig()
    battery = BatteryPack(c, 2, initial_soc=0.8)
    first = battery.advance(60.0, [False, False], [False, True])
    second = battery.advance(60.0, [False, False], [False, True])
    current = np.array([c.passive_current_a, c.active_current_a])
    expected_drop = current * 60.0 / c.capacity_coulomb
    np.testing.assert_allclose(first.soc, 0.8 - expected_drop)
    np.testing.assert_allclose(first.soc - second.soc, expected_drop)
    np.testing.assert_allclose(second.current_a, current)
    np.testing.assert_allclose(
        second.voltage_v,
        battery.open_circuit_voltage_v() - current * c.internal_resistance_ohm,
    )
    assert second.soc[0] > second.soc[1]


def test_solar_offsets_load_before_charge_efficiency():
    # Both modes must charge in this scenario, regardless of experiment defaults.
    c = BatteryConfig(capacity_ah=2.6, active_current_a=0.45, solar_current_a=0.65)
    battery = BatteryPack(c, 2, initial_soc=0.5)
    step = battery.advance(100.0, [True, True], [False, True])
    net = np.array([c.passive_current_a, c.active_current_a]) - c.solar_current_a
    np.testing.assert_allclose(step.current_a, net)
    np.testing.assert_allclose(step.soc, 0.5 - net * c.charge_efficiency * 100 / c.capacity_coulomb)
    assert np.all(step.charging)


def test_sunlight_can_still_discharge_under_heavy_load():
    c = replace(BatteryConfig(), solar_current_a=0.2)
    step = BatteryPack(c, 1, 0.5).advance(100.0, True, True)
    assert step.soc[0] < 0.5
    assert not step.charging[0]


def test_task_joules_equal_chemical_energy_minus_resistive_loss():
    c = replace(BatteryConfig(), passive_current_a=0.0, active_current_a=0.0)
    battery = BatteryPack(c, 1, 0.8)
    step = battery.advance(60.0, False, True, 400.0)
    s1 = step.soc[0]
    slope = c.voltage_full_v - c.voltage_empty_v
    chemical_energy = c.capacity_coulomb * (
        c.voltage_empty_v * (0.8 - s1) + 0.5 * slope * (0.8**2 - s1**2)
    )
    resistive_loss = step.current_a[0] ** 2 * c.internal_resistance_ohm * 60.0
    assert chemical_energy == pytest.approx(400.0 + resistive_loss, abs=1e-8)
    assert step.task_energy_j[0] == 400.0


def test_task_joules_match_solar_and_baseline_bus_work():
    c = BatteryConfig()
    battery = BatteryPack(c, 1, 0.8)
    step = battery.advance(60.0, True, True, 300.0)
    task_current = step.current_a[0] - c.active_current_a + c.solar_current_a
    mean_soc = (0.8 + step.soc[0]) / 2
    mean_voltage = c.voltage_empty_v + (c.voltage_full_v - c.voltage_empty_v) * mean_soc
    mean_voltage -= step.current_a[0] * c.internal_resistance_ohm
    assert task_current * mean_voltage * 60.0 == pytest.approx(300.0, abs=1e-8)


def test_full_task_budget_reaches_cutoff_and_blocks_next_step():
    # Leave enough headroom above 20% for a 60-second active baseline plus task work.
    c = BatteryConfig(capacity_ah=2.6, active_current_a=0.45)
    battery = BatteryPack(c, 1, 0.21)
    assert battery.plan_active(60.0, False, True)[0]
    budget = battery.task_energy_budget_j(60.0, False, True)
    assert budget[0] > 0
    step = battery.advance(60.0, False, True, budget)
    assert step.active[0]
    assert step.soc[0] == pytest.approx(c.cutoff_soc, abs=1e-12)
    assert battery.locked_out[0]
    assert not battery.plan_active(60.0, False, True)[0]
    assert battery.task_energy_budget_j(60.0, False, True)[0] == 0


def test_unaffordable_baseline_denied_before_task_admission():
    battery = BatteryPack(BatteryConfig(), 1, 0.2001)
    assert not battery.plan_active(60.0, False, True)[0]
    assert battery.task_energy_budget_j(60.0, False, True)[0] == 0
    step = battery.advance(60.0, False, True)
    assert not step.active[0]
    assert step.current_a[0] == pytest.approx(battery.config.passive_current_a)


def test_zero_cutoff_still_requires_full_interval_baseline_energy():
    c = replace(BatteryConfig(), cutoff_soc=0.0)
    battery = BatteryPack(c, 1, 0.0001)
    assert not battery.plan_active(60.0, False, True)[0]
    assert battery.task_energy_budget_j(60.0, False, True)[0] == 0.0


def test_hysteresis_requires_recharge_to_resume():
    c = replace(BatteryConfig(), capacity_ah=0.01)
    battery = BatteryPack(c, 1, 0.19)
    assert not battery.available(True)[0]
    battery.advance(4.0, True, True)
    assert c.cutoff_soc < battery.soc[0] < c.resume_soc
    assert not battery.available(True)[0]
    battery.advance(4.0, True, True)
    assert battery.soc[0] >= c.resume_soc
    assert battery.available(True)[0]


def test_recovery_requires_voltage_as_well_as_soc():
    c = BatteryConfig(capacity_ah=0.01, resume_voltage_v=8.4)
    battery = BatteryPack(c, 1, 0.19)
    battery.advance(8.0, True, True)
    assert battery.soc[0] >= c.resume_soc
    assert battery.open_circuit_voltage_v()[0] < c.resume_voltage_v
    assert not battery.available(True)[0]
    battery.advance(100.0, True, True)
    assert battery.open_circuit_voltage_v()[0] == c.resume_voltage_v
    assert battery.available(True)[0]
    # The recovery threshold does not suspend a healthy pack on subsequent discharge.
    battery.advance(4.0, False, True)
    assert battery.open_circuit_voltage_v()[0] < c.resume_voltage_v
    assert battery.available(True)[0]


def test_overbudget_rejected_without_mutation():
    battery = BatteryPack(BatteryConfig(), 1, 0.21)
    budget = battery.task_energy_budget_j(60.0, False, True)
    with pytest.raises(ValueError, match="budget"):
        battery.advance(60.0, False, True, budget + 1.0)
    np.testing.assert_array_equal(battery.soc, [0.21])
    np.testing.assert_array_equal(battery.current_a, [0.0])
    assert not battery.locked_out[0]


def test_task_work_is_forbidden_when_passive():
    battery = BatteryPack(BatteryConfig(), 1)
    with pytest.raises(ValueError, match="budget"):
        battery.advance(10.0, True, False, 0.00000000001)


def test_current_limit_and_capacity_saturation():
    c = BatteryConfig()
    battery = BatteryPack(c, 2, [0.99, 0.01])
    step = battery.advance(100000.0, [True, False], False)
    np.testing.assert_array_equal(step.soc, [1.0, 0.0])
    assert step.current_a[0] == pytest.approx(
        -0.01 * c.capacity_coulomb / (100000 * c.charge_efficiency)
    )
    assert step.current_a[1] == pytest.approx(0.01 * c.capacity_coulomb / 100000)
    battery = BatteryPack(c, 1)
    budget = battery.task_energy_budget_j(10.0, False, True)
    step = battery.advance(10.0, False, True, budget)
    assert step.current_a[0] <= c.max_discharge_current_a + 1e-12


def test_snapshot_is_independent_of_subsequent_steps():
    battery = BatteryPack(BatteryConfig(), 1)
    step = battery.advance(60.0, False, True)
    soc_before = step.soc.copy()
    battery.advance(60.0, False, True)
    np.testing.assert_array_equal(step.soc, soc_before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity_ah": 0},
        {"solar_current_a": -1},
        {"active_current_a": 0.01},
        {"charge_efficiency": 1.1},
        {"cutoff_soc": 0.5, "resume_soc": 0.4},
        {"voltage_empty_v": 9},
        {"internal_resistance_ohm": 100},
        {"capacity_ah": float("nan")},
        {"resume_voltage_v": 0},
        {"resume_voltage_v": -1},
        {"resume_voltage_v": 8.5},
        {"resume_voltage_v": float("nan")},
        {"resume_voltage_v": float("inf")},
        {"resume_voltage_v": True},
        {"resume_voltage_v": "5.5"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BatteryConfig(**kwargs)


def test_invalid_step_arguments():
    battery = BatteryPack(BatteryConfig(), 2)
    for dt in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            battery.advance(dt, True, True)
    with pytest.raises(ValueError):
        battery.advance(1.0, [True], True)
    with pytest.raises(ValueError):
        battery.advance(1.0, True, [0, 2])
    with pytest.raises(ValueError):
        battery.advance(1.0, True, True, [-1.0, 0.0])
