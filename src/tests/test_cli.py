"""CLI integration checks for reproducibility, data exports and input errors."""

import builtins
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from meteogym.cli import main
from meteogym.config import SimulationConfig
from meteogym.env import MeteoGymEnv
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
    assert summary["output_schema_version"] == 2
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


def test_logged_snapshots_match_interval_start_and_preserve_end_battery_state(tmp_path):
    config = SimulationConfig(
        satellite_ids=(541, 1),
        num_users=2,
        duration_s=25,
        step_s=10,
        initial_tasks=True,
        user_departure_rate_per_minute=600,
        user_return_rate_per_minute=600,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "run"
    main(
        [
            "run",
            "--config",
            str(config_path),
            "--seed",
            "31",
            "--policy",
            "passive",
            "--output-dir",
            str(output),
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert summary["output_schema_version"] == 2
    assert "start_s" in summary["snapshot_timing"] and "end_s" in summary["snapshot_timing"]
    assert [row["user_online"] for row in rows] == [[1, 1], [0, 0], [1, 1]]
    env = MeteoGymEnv(config)
    try:
        observation, _ = env.reset(seed=31)
        for row in rows:
            assert row["start_s"] == observation["time_s"][0]
            np.testing.assert_array_equal(
                row["satellite_geographic_deg_km"], observation["geographic_deg_km"]
            )
            np.testing.assert_array_equal(row["user_online"], observation["user_online"])
            np.testing.assert_array_equal(
                row["battery_voltage_start_v"], observation["battery_voltage_v"]
            )
            np.testing.assert_array_equal(row["sunlit"], observation["sunlit"])
            np.testing.assert_array_equal(row["contact_count"], observation["contact_count"])
            observation, _, _, _, info = env.step(np.array(row["actions"], dtype=np.int8))
            assert row["end_s"] == observation["time_s"][0]
            for field in ("battery_soc", "battery_voltage_v", "battery_current_a", "charging"):
                np.testing.assert_array_equal(row[field], observation[field])
            np.testing.assert_array_equal(row["task_energy_j"], info["task_energy_j"])
    finally:
        env.close()


@pytest.mark.parametrize(
    "plot_args, mode, interval, save_name, show",
    [
        (["--plot"], "animate", 200, None, True),
        (["--plot", "step", "--plot-interval", "75"], "step", 75, None, True),
        (
            ["--plot", "animate", "--plot-save", "result.gif", "--plot-no-show"],
            "animate",
            200,
            "result.gif",
            False,
        ),
    ],
)
def test_post_run_plot_routing_occurs_after_environment_close(
    plot_args, mode, interval, save_name, show, monkeypatch, tmp_path
):
    calls = []
    output = tmp_path / "run"
    original_close = MeteoGymEnv.close

    def close(env):
        calls.append("close")
        return original_close(env)

    def plot_dataset(path, **kwargs):
        assert calls == ["close"]
        assert (path / "summary.json").exists()
        assert (path / "steps.jsonl").exists()
        calls.append((path, kwargs))

    module = ModuleType("meteogym.plots.plot_results")
    module.plot_dataset = plot_dataset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(MeteoGymEnv, "close", close)
    main(["run", "--users", "1", "--duration", "1", "--output-dir", str(output), *plot_args])
    assert calls[1] == (
        output,
        {
            "mode": mode,
            "interval_ms": interval,
            "save": None if save_name is None else Path(save_name),
            "show": show,
        },
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--plot-save", "result.png"], "require --plot"),
        (["--plot-no-show"], "require --plot"),
        (["--plot", "--plot-no-show"], "requires --plot-save"),
        (["--plot", "--plot-save", "result.txt"], ".png, .pdf, .svg or .gif"),
        (["--plot", "--plot-interval", "0"], "finite and positive"),
        (["--plot", "--plot-interval", "-1"], "finite and positive"),
        (["--plot", "--plot-interval", "nan"], "finite and positive"),
        (["--plot", "--plot-interval", "inf"], "finite and positive"),
        (["--plot", "invalid"], "invalid choice"),
    ],
)
def test_invalid_plot_flags_fail_before_running(arguments, message, monkeypatch, tmp_path, capsys):
    def unexpected_run(*_args):
        pytest.fail("Invalid plot flags must fail before the simulation starts")

    monkeypatch.setattr("meteogym.cli._run", unexpected_run)
    with pytest.raises(SystemExit) as error:
        main(["run", "--output-dir", str(tmp_path / "run"), *arguments])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_run_without_plot_never_imports_plotting_dependencies(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("matplotlib", "meteogym.plots", "plots")):
            pytest.fail(f"Unexpected plotting import without --plot: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert main(["run", "--users", "1", "--duration", "1", "--output-dir", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    "exception, expected",
    [
        (ImportError("No module named 'matplotlib'"), "plot dependencies"),
        (RuntimeError("Interactive backend is unavailable"), "Interactive backend is unavailable"),
    ],
)
def test_optional_plot_errors_are_clean_and_preserve_run_outputs(
    exception, expected, monkeypatch, tmp_path, capsys
):
    def plot_dataset(*_args, **_kwargs):
        raise exception

    module = ModuleType("meteogym.plots.plot_results")
    module.plot_dataset = plot_dataset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(SystemExit) as error:
        main(["run", "--users", "1", "--duration", "1", "--plot", "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert expected in stderr
    assert "Traceback" not in stderr
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "steps.jsonl").exists()
