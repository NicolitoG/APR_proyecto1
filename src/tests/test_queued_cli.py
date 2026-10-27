"""Configuration and recorded outcomes for queued-mode CLI runs."""

import json
from datetime import datetime

import pytest

from meteogym.cli import main
from meteogym.orbits import OrbitModel


def test_queued_cli_records_failed_batch_and_voltage_suspension(tmp_path):
    model = OrbitModel(satellite_ids=(1,))
    location = model.snapshot(datetime.fromisoformat(model.default_start_time)).geographic[0, :2]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "satellite_ids": [1], "num_users": 1, "user_locations_deg": [location.tolist()],
        "duration_s": 2, "step_s": 1, "initial_tasks": True, "initial_soc": 0.5,
        "task_rate_per_minute": [0, 0], "task_size_mb": [1, 1], "uplink_mbps": 8,
        "battery": {"capacity_ah": 0.1, "active_current_a": 0,
                    "voltage_empty_v": 6.5, "voltage_full_v": 8.4,
                    "passive_current_a": 0, "solar_current_a": 0, "resume_voltage_v": 7.8},
    }))
    output = tmp_path / "run"
    assert main([
        "run", "--config", str(config), "--task-arrival-mode", "queued",
        "--minimum-voltage", "7.44", "--output-dir", str(output),
    ]) == 0
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert summary["config"]["task_arrival_mode"] == "queued"
    assert summary["config"]["battery"]["minimum_voltage_v"] == 7.44
    assert summary["config"]["battery"]["resume_voltage_v"] == 7.8
    assert summary["config"]["battery"]["capacity_ah"] == 0.1
    assert summary["results"]["failed_tasks"] == 1
    assert summary["results"]["pending_tasks"] == 0
    assert rows[0]["task_events"][0]["status"] == "battery_cutoff"
    assert rows[0]["battery_failure_events"][0]["satellite_id"] == 1
    assert rows[0]["battery_failure_events"][0]["voltage_v"] <= 7.44
    assert rows[0]["battery_suspended"] == rows[1]["battery_suspended"] == [1]
    assert rows[1]["task_events"] == []
    assert rows[1]["active_duration_s"] == [0]
    assert rows[1]["pending_tasks_per_user"] == [0]


def test_queued_cli_counts_new_arrivals_in_final_interval(tmp_path):
    assert main([
        "run", "--satellites", "1", "--users", "2", "--duration", "1", "--step", "1",
        "--policy", "passive", "--task-rate", "600", "600", "--task-arrival-mode", "queued",
        "--output-dir", str(tmp_path),
    ]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    row = json.loads((tmp_path / "steps.jsonl").read_text())
    assert summary["results"]["generated_tasks"] > 2
    assert row["pending_tasks"] == sum(row["pending_tasks_per_user"])
    assert summary["results"]["pending_tasks"] == summary["results"]["generated_tasks"]


@pytest.mark.parametrize("args", [
    ["--task-arrival-mode", "invalid"], ["--minimum-voltage", "-1"],
    ["--minimum-voltage", "9"], ["--minimum-voltage", "nan"],
])
def test_invalid_queue_or_voltage_arguments_are_rejected(args):
    with pytest.raises(SystemExit) as error:
        main(["run", *args])
    assert error.value.code == 2
