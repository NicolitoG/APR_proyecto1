"""Replay timing, historical compatibility, and malformed output contracts."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from meteogym.battery import BatteryConfig
from meteogym.config import SimulationConfig
from meteogym.orbits import OrbitModel
from meteogym.plots.data import load_dataset


def dataset_values():
    config = SimulationConfig(
        satellite_ids=(541, 1), num_users=2, initial_soc=0.75,
        battery=BatteryConfig(voltage_empty_v=6.5, voltage_full_v=8.4),
    )
    summary = {
        "output_schema_version": 2,
        "config": config.to_dict(),
        "start_time_utc": "2016-10-29T19:52:11.055936+00:00",
        "satellite_ids": [541, 1],
        "user_locations_deg": [[-33, -70], [20, 175]],
    }
    rows = [
        {
            "step": 1,
            "start_s": 0.0,
            "end_s": 10.0,
            "time_utc": "2016-10-29T19:52:21.055936+00:00",
            "satellite_geographic_deg_km": [[10, 170, 550], [-20, -100, 550]],
            "user_online": [True, True],
            "contact_count": [2, 1],
            "actions": [1, 1],
            "effective_active": [1, 0],
            "battery_voltage_start_v": [7.9, 7.8],
            "battery_voltage_v": [7.6, 7.5],
            "reward": 1.0,
        },
        {
            "step": 2,
            "start_s": 10.0,
            "end_s": 15.0,
            "time_utc": "2016-10-29T19:52:26.055936+00:00",
            "satellite_geographic_deg_km": [[11, 171, 550], [-21, -99, 550]],
            "user_online": [False, True],
            "contact_count": [0, 1],
            "effective_active": [0, 1],
            "battery_voltage_start_v": [7.6, 7.5],
            "battery_voltage_v": [7.3, 7.2],
        },
    ]
    return summary, rows


def write_dataset(path, summary, rows):
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def legacy_values():
    summary, rows = dataset_values()
    del summary["output_schema_version"]
    for row in rows:
        for name in ("satellite_geographic_deg_km", "user_online", "battery_voltage_start_v"):
            del row[name]
    return summary, rows


def test_new_dataset_uses_exact_start_states_and_effective_mode_without_tle(tmp_path, monkeypatch):
    summary, rows = dataset_values()
    summary["config"]["tle_path"] = str(tmp_path / "missing-original.tle")
    # Presence must come from saved data even when departures are configured.
    summary["config"]["user_departure_rate_per_minute"] = 0.5
    write_dataset(tmp_path, summary, rows)

    def forbidden_propagation(*args, **kwargs):
        raise AssertionError("saved coordinates must not instantiate OrbitModel")

    monkeypatch.setattr("meteogym.orbits.OrbitModel", forbidden_propagation)
    dataset = load_dataset(tmp_path)
    assert len(dataset) == 2
    assert dataset.satellite_ids == (541, 1)
    assert dataset.voltage_limits == (6.5, 8.4)
    np.testing.assert_array_equal(dataset.user_locations_deg, summary["user_locations_deg"])
    first, second = dataset.frame(0), dataset.frame(1)
    assert first.time_utc == datetime.fromisoformat(summary["start_time_utc"])
    assert second.time_utc == first.time_utc + timedelta(seconds=10)
    assert first.step == 1 and second.step == 2
    assert first.elapsed_s == first.start_s == 0
    assert second.start_s == 10
    assert first.active.tolist() == [True, False]
    assert second.user_online.tolist() == [False, True]
    assert first.contact_count.tolist() == [2, 1]
    np.testing.assert_array_equal(first.satellite_locations_deg, [[10, 170], [-20, -100]])
    np.testing.assert_array_equal(first.voltage_v, [7.9, 7.8])
    np.testing.assert_array_equal(second.voltage_v, [7.6, 7.5])
    assert first.reward == 1 and second.reward is None
    assert dataset.frame(0) is first
    assert dataset.frame(-1) is second
    assert not first.voltage_v.flags.writeable


def test_legacy_reconstruction_matches_interval_start_orbits_and_voltage(tmp_path):
    summary, rows = legacy_values()
    write_dataset(tmp_path, summary, rows)
    with pytest.warns(UserWarning, match="Legacy dataset.*reconstructed") as caught:
        dataset = load_dataset(tmp_path)
        first, second = dataset.frame(0), dataset.frame(1)
    assert len(caught) == 1
    model = OrbitModel(satellite_ids=(541, 1))
    start = datetime.fromisoformat(summary["start_time_utc"])
    np.testing.assert_allclose(
        first.satellite_locations_deg, model.snapshot(start).geographic[:, :2]
    )
    np.testing.assert_allclose(
        second.satellite_locations_deg,
        model.snapshot(start + timedelta(seconds=10)).geographic[:, :2],
    )
    np.testing.assert_allclose(first.voltage_v, 6.5 + 0.75 * (8.4 - 6.5))
    np.testing.assert_allclose(second.voltage_v, rows[0]["battery_voltage_v"])
    assert first.user_online.tolist() == second.user_online.tolist() == [True, True]
    assert dataset.frame(1) is second


def test_outcomes_accumulate_once_and_charging_describes_each_interval(tmp_path):
    summary, rows = dataset_values()
    rows[0].update(generated_tasks=4, completed=1, failed=2, reward=-1, charging=[0, 1])
    rows[1].update(generated_tasks=6, completed=2, failed=0, reward=2, charging=[1, 0])
    write_dataset(tmp_path, summary, rows)
    dataset = load_dataset(tmp_path)
    final = dataset.frame(-1)
    assert (final.generated_tasks, final.completed_tasks, final.failed_tasks) == (6, 3, 2)
    assert final.reward == 2 and final.total_reward == 1
    assert final.charging.tolist() == [True, False]
    assert not final.charging.flags.writeable
    first = dataset.frame(0)
    assert (first.generated_tasks, first.completed_tasks, first.failed_tasks) == (4, 1, 2)
    assert first.reward == first.total_reward == -1
    assert first.charging.tolist() == [False, True]
    assert dataset.frame(-1).total_reward == 1


def test_missing_historical_outcomes_remain_unknown(tmp_path):
    summary, rows = dataset_values()
    rows[1].update(generated_tasks=5, completed=2, failed=0, reward=None)
    write_dataset(tmp_path, summary, rows)
    first, second = (load_dataset(tmp_path).frame(index) for index in (0, 1))
    assert first.generated_tasks is first.completed_tasks is first.failed_tasks is None
    assert first.charging is None
    assert first.total_reward == 1
    assert second.generated_tasks == 5
    assert second.completed_tasks is second.failed_tasks is second.total_reward is None


def test_generated_task_total_cannot_decrease(tmp_path):
    summary, rows = dataset_values()
    rows[0]["generated_tasks"] = 5
    rows[1]["generated_tasks"] = 4
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="generated_tasks cannot decrease"):
        load_dataset(tmp_path)


def test_legacy_presence_with_departures_requires_new_run(tmp_path):
    summary, rows = legacy_values()
    summary["config"]["user_departure_rate_per_minute"] = 0.2
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="Historical presence cannot be recovered.*new run"):
        load_dataset(tmp_path)


def test_legacy_missing_tle_has_actionable_error(tmp_path):
    summary, rows = legacy_values()
    summary["config"]["tle_path"] = str(tmp_path / "missing.tle")
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="Restore the original TLE.*new run"):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("satellite_geographic_deg_km", [[0, 0, 500]], "shape"),
        ("satellite_geographic_deg_km", [[91, 0, 500], [0, 0, 500]], "latitude"),
        ("satellite_geographic_deg_km", [[0, 181, 500], [0, 0, 500]], "longitude"),
        ("satellite_geographic_deg_km", [[0, 0, float("nan")], [0, 0, 500]], "finite"),
        ("contact_count", [1.5, 0], "contact_count.*integers"),
        ("contact_count", [3, 0], "contact_count.*online users"),
        ("user_online", [0, 2], "binary"),
        ("effective_active", [1], "shape"),
        ("battery_voltage_start_v", [7, float("inf")], "finite"),
        ("battery_voltage_v", [7, -1], "positive"),
        ("start_s", -1, "times must satisfy"),
        ("step", True, "step must be"),
        ("generated_tasks", -1, "generated_tasks.*nonnegative integer"),
        ("completed", 1.5, "completed.*nonnegative integer"),
        ("failed", True, "failed.*nonnegative integer"),
        ("reward", True, "reward.*finite number"),
        ("charging", [1], "charging.*shape"),
        ("charging", [1, 2], "charging.*binary"),
    ],
)
def test_malformed_frame_rejected(tmp_path, field, value, match):
    summary, rows = dataset_values()
    rows[0][field] = value
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match=match):
        load_dataset(tmp_path)


@pytest.mark.parametrize("satellite_ids", [[1, 1], [True, 1], [541.0, 1], [1, 541]])
def test_invalid_ids_or_mismatched_selection_rejected(tmp_path, satellite_ids):
    summary, rows = dataset_values()
    summary["satellite_ids"] = satellite_ids
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="satellite_ids"):
        load_dataset(tmp_path)


def test_summary_coordinates_must_match_user_count(tmp_path):
    summary, rows = dataset_values()
    summary["user_locations_deg"] = [[0, 0]]
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="user_locations_deg.*shape"):
        load_dataset(tmp_path)


def test_nonmonotonic_steps_and_times_rejected(tmp_path):
    summary, rows = dataset_values()
    rows[1]["step"] = 1
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_dataset(tmp_path)
    rows[1]["step"] = 2
    rows[1]["start_s"] = 5
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="contiguous"):
        load_dataset(tmp_path)


def test_schema_two_requires_saved_start_state(tmp_path):
    summary, rows = dataset_values()
    del rows[0]["satellite_geographic_deg_km"]
    write_dataset(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="missing satellite_geographic_deg_km"):
        load_dataset(tmp_path)


def test_missing_empty_and_invalid_files_produce_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="Expected summary.json and steps.jsonl"):
        load_dataset(tmp_path)
    summary, _ = dataset_values()
    write_dataset(tmp_path, summary, [])
    with pytest.raises(ValueError, match="contains no frames"):
        load_dataset(tmp_path)
    (tmp_path / "steps.jsonl").write_text("{broken json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="steps.jsonl line 1: invalid JSON"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("index", [2, -3])
def test_frame_index_must_be_in_range(tmp_path, index):
    write_dataset(tmp_path, *dataset_values())
    with pytest.raises(IndexError, match="out of range"):
        load_dataset(tmp_path).frame(index)
