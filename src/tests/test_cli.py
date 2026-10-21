"""CLI integration checks for reproducibility, data exports and input errors."""

import csv
import json
from datetime import datetime

import numpy as np
import pytest

from meteogym.cli import main
from meteogym.orbits import OrbitModel


def test_seeded_run_reproduces_summary_and_steps_and_handles_partial_step(tmp_path):
    output_a, output_b = tmp_path / "a", tmp_path / "b"
    arguments = [
        "run",
        "--satellites",
        "1",
        "541",
        "--users",
        "3",
        "--duration",
        "25",
        "--step",
        "10",
        "--seed",
        "31",
        "--policy",
        "random",
        "--initial-tasks",
    ]
    assert main(arguments + ["--output-dir", str(output_a)]) == 0
    assert main(arguments + ["--output-dir", str(output_b)]) == 0
    summary = json.loads((output_a / "summary.json").read_text())
    assert summary == json.loads((output_b / "summary.json").read_text())
    assert (output_a / "steps.jsonl").read_text() == (output_b / "steps.jsonl").read_text()
    assert summary["satellite_ids"] == [1, 541]
    assert summary["constellation_size"] == 1080
    assert summary["config"]["initial_tasks"] is True
    assert summary["seed"] == 31
    assert summary["results"]["steps"] == 3
    assert summary["results"]["elapsed_s"] == 25
    assert summary["results"]["truncated"] is True
    assert summary["results"]["generated_tasks"] == 3
    assert len(summary["final_soc"]) == 2
    assert "astropy" in summary["versions"]
    steps = [json.loads(line) for line in (output_a / "steps.jsonl").read_text().splitlines()]
    assert [row["step_s"] for row in steps] == [10, 10, 5]
    assert all(len(row["sunlit"]) == 2 and len(row["task_energy_j"]) == 2 for row in steps)


def test_json_config_overrides_and_single_explicit_satellite_complete_a_task(tmp_path):
    model = OrbitModel(satellite_ids=(541,))
    location = model.snapshot(datetime.fromisoformat(model.default_start_time)).geographic[0, :2]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "satellite_ids": [1, 2],
                "num_users": 10,
                "duration_s": 20,
                "step_s": 10,
                "initial_soc": 0.7,
                "task_size_mb": [0.1, 0.1],
                "initial_tasks": True,
            }
        )
    )
    output = tmp_path / "run"
    main(
        [
            "run",
            "--config",
            str(config),
            "--satellites",
            "541",
            "--duration",
            "10",
            "--users",
            "99",
            "--user-location",
            *map(str, location),
            "--output-dir",
            str(output),
            "--policy",
            "visible",
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["satellite_ids"] == [541]
    assert summary["config"]["num_users"] == 1
    assert summary["config"]["initial_soc"] == 0.7
    assert summary["config"]["duration_s"] == 10
    assert summary["results"]["completed_tasks"] == 1
    assert summary["results"]["total_task_energy_j"] == pytest.approx(0.2)
    np.testing.assert_allclose(summary["user_locations_deg"], [location])


def test_contacts_csv_and_metadata_cover_requested_samples_and_user_locations(tmp_path):
    model = OrbitModel(satellite_ids=(1,))
    location = model.snapshot(datetime.fromisoformat(model.default_start_time)).geographic[0, :2]
    output = tmp_path / "subdir" / "contacts.csv"
    main(
        [
            "contacts",
            "--satellites",
            "1",
            "--user-location",
            *map(str, location),
            "--duration",
            "25",
            "--step",
            "10",
            "--output",
            str(output),
        ]
    )
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert [float(row["elapsed_s"]) for row in rows] == [0, 10, 20]
    assert {row["satellite_id"] for row in rows} == {"1"}
    assert {row["user_id"] for row in rows} == {"0"}
    assert metadata["sampled_end_s"] == 20
    assert metadata["visible_rows"] == 3
    assert metadata["seed"] == 10
    np.testing.assert_allclose(metadata["user_locations_deg"], [location])


def test_all_satellites_selection_and_random_user_region(tmp_path):
    output = tmp_path / "contacts.csv"
    main(
        [
            "contacts",
            "--all-satellites",
            "--users",
            "2",
            "--duration",
            "1",
            "--step",
            "10",
            "--user-region",
            "-20",
            "-10",
            "30",
            "40",
            "--output",
            str(output),
        ]
    )
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert len(metadata["satellite_ids"]) == 1080
    assert metadata["config"]["satellite_ids"] is None
    for latitude, longitude in metadata["user_locations_deg"]:
        assert -20 <= latitude <= -10
        assert 30 <= longitude <= 40


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--duration", "0"], "duration_s"),
        (["--task-rate", "3", "1"], "task_rate_per_minute"),
        (["--satellites", "1081"], "Unknown satellite IDs"),
        (["--seed", "-1"], "seed must be nonnegative"),
        (["--start-time", "2016-10-29T00:00:00"], "UTC offset"),
        (["--users", "0"], "num_users"),
    ],
)
def test_invalid_arguments_produce_clean_cli_errors(arguments, message, capsys, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["run", "--output-dir", str(tmp_path), *arguments])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("contents", ["[1, 2]", "{broken", '{"battery": {"capacity_ah": 0}}'])
def test_invalid_json_configuration_produces_clean_error(contents, tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(contents)
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(config), "--output-dir", str(tmp_path / "out")])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
