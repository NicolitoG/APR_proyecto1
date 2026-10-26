"""Loaded-voltage cutoffs and delivered electrical work for queued transfers."""

from dataclasses import replace

import numpy as np
import pytest

from meteogym.battery import BatteryConfig, BatteryPack


@pytest.fixture
def config():
    return BatteryConfig(
        capacity_ah=100 / 3600,
        voltage_empty_v=6.0,
        voltage_full_v=8.0,
        internal_resistance_ohm=0.1,
        passive_current_a=0.1,
        active_current_a=0.5,
        solar_current_a=0.8,
        max_discharge_current_a=2.0,
        minimum_voltage_v=7.0,
    )


def test_minimum_voltage_defaults_to_configured_empty_voltage():
    c = BatteryConfig(voltage_empty_v=6.7)
    assert c.minimum_voltage_v is None
    assert c.effective_minimum_voltage_v == 6.7
    assert replace(c, minimum_voltage_v=7.1).effective_minimum_voltage_v == 7.1


@pytest.mark.parametrize("value", [0, -1, 9.0, float("nan"), float("inf"), True, "7.0"])
def test_invalid_minimum_voltage(value):
    with pytest.raises(ValueError, match="minimum_voltage_v"):
        BatteryConfig(minimum_voltage_v=value)


def test_immediate_load_sag_fails_without_consuming_charge(config):
    battery = BatteryPack(config, 1, 0.52)
    result = battery.advance_transfer(10, False, True, 0)
    assert result.active and result.failed
    assert result.duration_s == 0
    assert result.task_energy_j == 0
    assert battery.soc[0] == 0.52
    assert battery.voltage_v[0] == pytest.approx(6.99)
    assert battery.current_a[0] == 0.5


def test_voltage_crossing_stops_inside_interval(config):
    battery = BatteryPack(config, 1, 0.8)
    result = battery.advance_transfer(100, False, True, 0)
    assert result.failed
    assert result.duration_s == pytest.approx(55)
    assert battery.soc[0] == pytest.approx(0.525)
    assert battery.voltage_v[0] == pytest.approx(config.minimum_voltage_v)


def test_partial_transfer_work_accounts_for_baseline_and_resistance(config):
    battery = BatteryPack(config, 1, 0.8)
    result = battery.advance_transfer(60, False, True, 4)
    assert result.failed
    assert 0 < result.duration_s < 60
    assert 0 < result.task_energy_j < 4 * 60
    assert result.task_energy_j == pytest.approx(4 * result.duration_s, abs=1e-10)
    current = battery.current_a[0]
    mean_voltage = (
        config.voltage_empty_v
        + (config.voltage_full_v - config.voltage_empty_v) * (0.8 + battery.soc[0]) / 2
        - config.internal_resistance_ohm * current
    )
    task_current = current - config.active_current_a
    assert result.task_energy_j == pytest.approx(task_current * mean_voltage * result.duration_s)
    slope = config.voltage_full_v - config.voltage_empty_v
    chemical_work = config.capacity_coulomb * (
        config.voltage_empty_v * (0.8 - battery.soc[0])
        + slope * (0.8**2 - battery.soc[0] ** 2) / 2
    )
    baseline_work = config.active_current_a * mean_voltage * result.duration_s
    resistive_loss = config.internal_resistance_ohm * current**2 * result.duration_s
    assert chemical_work == pytest.approx(result.task_energy_j + baseline_work + resistive_loss)


def test_full_interval_delivers_requested_work(config):
    battery = BatteryPack(config, 1, 0.9)
    result = battery.advance_transfer(5, False, True, 2)
    assert result.active and not result.failed
    assert result.duration_s == 5
    assert result.task_energy_j == pytest.approx(10, abs=1e-10)


def test_hardware_current_limit_throttles_work_without_failure(config):
    battery = BatteryPack(config, 1, 0.9)
    result = battery.advance_transfer(1, False, True, 10000)
    assert not result.failed
    assert result.duration_s == 1
    assert battery.current_a[0] == pytest.approx(config.max_discharge_current_a)
    assert 0 < result.task_energy_j < 10000


def test_legacy_soc_lockout_does_not_deny_queued_processing(config):
    config = replace(config, minimum_voltage_v=6.0)
    battery = BatteryPack(config, 1, 0.15)
    assert battery.locked_out[0]
    result = battery.advance_transfer(1, False, True, 0.1)
    assert result.active and not result.failed
    assert result.task_energy_j == pytest.approx(0.1)


def test_passive_interval_ignores_voltage_failure_and_can_recharge(config):
    battery = BatteryPack(config, 1, 0.1)
    result = battery.advance_transfer(10, True, False, 0)
    assert not result.active and not result.failed
    assert result.duration_s == 10
    assert result.task_energy_j == 0
    assert battery.charging[0]
    assert battery.soc[0] == pytest.approx(0.1 + 0.7 * config.charge_efficiency * 10 / 100)
    assert battery.voltage_v[0] < config.minimum_voltage_v


def test_solar_can_support_task_work_while_charging(config):
    battery = BatteryPack(config, 1, 0.8)
    result = battery.advance_transfer(10, True, True, 1)
    assert not result.failed
    assert result.task_energy_j == pytest.approx(10)
    assert battery.charging[0]
    assert battery.soc[0] > 0.8


def test_empty_pack_never_supplies_work_below_custom_voltage_cutoff(config):
    config = replace(config, minimum_voltage_v=1.0)
    battery = BatteryPack(config, 1, 0.01)
    result = battery.advance_transfer(100, False, True, 0)
    assert result.failed
    assert result.duration_s == pytest.approx(2)
    assert battery.soc[0] == pytest.approx(0)
    result = battery.advance_transfer(100, False, True, 1)
    assert result.failed
    assert result.duration_s == 0
    assert result.task_energy_j == 0


def test_terminal_voltage_exactly_at_cutoff_fails(config):
    battery = BatteryPack(config, 1, 0.525)
    result = battery.advance_transfer(1, False, True, 0)
    assert result.failed
    assert result.duration_s == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("power", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_power_does_not_mutate_battery(config, power):
    battery = BatteryPack(config, 1, 0.8)
    with pytest.raises(ValueError, match="task_power_w"):
        battery.advance_transfer(1, False, True, power)
    np.testing.assert_array_equal(battery.soc, [0.8])
    np.testing.assert_array_equal(battery.current_a, [0.0])


def test_rejects_task_work_when_passive(config):
    with pytest.raises(ValueError, match="passive"):
        BatteryPack(config, 1).advance_transfer(1, False, False, 1)


def test_requires_scalar_pack(config):
    with pytest.raises(ValueError, match="single"):
        BatteryPack(config, 2).advance_transfer(1, False, True, 1)
