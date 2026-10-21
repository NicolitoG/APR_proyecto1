"""Map artists, keyboard navigation and portable static/animated exports."""

from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("PIL")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from PIL import Image  # noqa: E402

from meteogym.cli import main as run_cli  # noqa: E402
from meteogym.plots.data import load_dataset  # noqa: E402
from meteogym.plots.plot_results import (  # noqa: E402
    ACTIVE_COLOR,
    INACTIVE_COLOR,
    MapPlayback,
    main,
    plot_dataset,
)


@pytest.fixture
def dataset_dir(tmp_path):
    directory = tmp_path / "dataset"
    run_cli(
        [
            "run",
            "--satellites",
            "1",
            "541",
            "--users",
            "3",
            "--duration",
            "30",
            "--step",
            "10",
            "--policy",
            "random",
            "--seed",
            "10",
            "--output-dir",
            str(directory),
        ]
    )
    return directory


def test_map_marker_positions_voltage_and_labels_follow_frame(dataset_dir):
    data = load_dataset(dataset_dir)
    viewer = MapPlayback(data)
    try:
        for index in (0, 2):
            frame = data.frame(index)
            viewer.update(index)
            np.testing.assert_allclose(
                viewer.satellites.get_offsets(), frame.satellite_locations_deg[:, ::-1]
            )
            np.testing.assert_allclose(viewer.satellites.get_array(), frame.voltage_v)
            np.testing.assert_allclose(
                viewer.users.get_offsets(), data.user_locations_deg[frame.user_online, ::-1]
            )
            assert viewer.satellites.get_sizes()[0] > viewer.users.get_sizes()[0]
            for sat, label in enumerate(viewer.labels):
                assert str(frame.contact_count[sat]) in label.get_text()
                assert f"S{data.satellite_ids[sat]}" in label.get_text()
                expected = ACTIVE_COLOR if frame.active[sat] else INACTIVE_COLOR
                assert to_rgba(label.get_color()) == to_rgba(expected)
        low = viewer.satellites.cmap(0.0)
        high = viewer.satellites.cmap(1.0)
        assert low[0] > low[2] and high[2] > high[0]
        assert viewer.satellites.norm.vmin == data.voltage_limits[0]
        assert viewer.satellites.norm.vmax == data.voltage_limits[1]
        assert len(viewer.ax.patches) > 100  # Actual offline land polygons, not an empty grid.
    finally:
        plt.close(viewer.figure)


def test_enter_previous_and_end_navigation_stop_at_bounds(dataset_dir):
    viewer = MapPlayback(load_dataset(dataset_dir))
    try:
        for key, expected in (
            ("enter", 1),
            ("right", 2),
            ("enter", 2),
            ("left", 1),
            ("home", 0),
            ("backspace", 0),
            ("end", 2),
        ):
            viewer._on_key(SimpleNamespace(key=key))
            assert viewer.index == expected
    finally:
        plt.close(viewer.figure)


def test_static_and_gif_exports_without_gui(dataset_dir, tmp_path):
    png = tmp_path / "map.png"
    viewer = plot_dataset(dataset_dir, mode="step", frame=1, save=png, show=False)
    assert viewer.index == 1
    with Image.open(png) as image:
        assert image.width > 1000 and image.height > 500
    gif = tmp_path / "map.gif"
    assert main([str(dataset_dir), "--save", str(gif), "--no-show", "--interval", "250"]) == 0
    with Image.open(gif) as image:
        assert image.n_frames == 3
        assert image.info["duration"] == 250
    assert not plt.get_fignums()


def test_export_stride_keeps_final_frame(dataset_dir, tmp_path):
    destination = tmp_path / "sparse.gif"
    plot_dataset(dataset_dir, save=destination, show=False, stride=5)
    with Image.open(destination) as image:
        assert image.n_frames == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--interval", "0"],
        ["--frame", "-1"],
        ["--frame", "999"],
        ["--no-show"],
        ["--save", "bad.mp4", "--no-show"],
    ],
)
def test_plot_cli_rejects_invalid_options(dataset_dir, arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(dataset_dir), *arguments])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_noninteractive_backend_explains_how_to_export(dataset_dir):
    with pytest.raises(RuntimeError, match="cannot open an interactive window"):
        plot_dataset(dataset_dir, show=True)
    assert not plt.get_fignums()
