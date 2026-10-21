"""Physical invariants and input contracts for orbit/contact calculations."""

import csv
from datetime import datetime, timedelta
from importlib.resources import files

import numpy as np
import pytest

from meteogym.orbits import OrbitModel, cylindrical_sunlit, export_contact_table


def epoch(model):
    return datetime.fromisoformat(model.default_start_time)


def test_bundled_constellation_and_preserved_selection_order():
    model = OrbitModel(satellite_ids=(1080, 1))
    assert model.constellation_size == 1080
    assert model.satellite_ids == (1080, 1)
    assert model.default_start_time == "2016-10-29T19:52:11.055936Z"
    all_satellites = OrbitModel()
    assert all_satellites.satellite_ids == tuple(range(1, 1081))
    snapshot = model.snapshot(epoch(model))
    assert snapshot.positions_km.shape == (2, 3)
    assert np.all((snapshot.geographic[:, 2] > 520) & (snapshot.geographic[:, 2] < 580))
    assert np.all(
        (np.linalg.norm(snapshot.velocities_km_s, axis=1) > 7.4)
        & (np.linalg.norm(snapshot.velocities_km_s, axis=1) < 7.8)
    )
    assert not snapshot.positions_km.flags.writeable


@pytest.mark.parametrize("ids", [(), (1, 1), (0,), (1081,), (1.0,), (True,)])
def test_invalid_satellite_selection(ids):
    with pytest.raises(ValueError):
        OrbitModel(satellite_ids=ids)


def test_velocity_matches_finite_difference_in_same_frame():
    model = OrbitModel(satellite_ids=(1, 100))
    time = epoch(model)
    velocity = model.snapshot(time).velocities_km_s
    before = model.snapshot(time - timedelta(seconds=0.5)).positions_km
    after = model.snapshot(time + timedelta(seconds=0.5)).positions_km
    np.testing.assert_allclose(after - before, velocity, atol=2e-5, rtol=0)


def test_geodetic_subpoint_is_zenith_and_antipode_invisible():
    model = OrbitModel(satellite_ids=(1,))
    time = epoch(model)
    latitude, longitude, height = model.snapshot(time).geographic[0]
    antipode_lon = (longitude + 360) % 360 - 180
    users = np.array([[latitude, longitude], [-latitude, antipode_lon]])
    contacts = model.contacts(time, users, min_elevation_deg=10)
    assert contacts.visible.tolist() == [[True, False]]
    assert contacts.elevation_deg[0, 0] == pytest.approx(90, abs=1e-5)
    assert contacts.elevation_deg[0, 1] < -89
    assert contacts.distances_km[0, 0] == pytest.approx(height, abs=1e-6)


def test_geographic_contacts_follow_satellite_motion():
    model = OrbitModel(satellite_ids=(1,))
    time = epoch(model)
    user = model.snapshot(time).geographic[:, :2]
    assert model.contacts(time, user).visible[0, 0]
    assert not model.contacts(time + timedelta(minutes=45), user).visible[0, 0]
    assert model.contacts(time, np.empty((0, 2))).visible.shape == (1, 0)


@pytest.mark.parametrize(
    "users,mask",
    [([[91, 0]], 0), ([[0, 181]], 0), ([[np.nan, 0]], 0), ([[0, 0]], -1), ([[0, 0]], 91)],
)
def test_invalid_contact_inputs(users, mask):
    model = OrbitModel(satellite_ids=(1,))
    with pytest.raises(ValueError):
        model.contacts(epoch(model), np.array(users), mask)


def test_cylindrical_shadow_day_night_and_limb():
    positions = np.array([[7000, 0, 0], [-7000, 0, 0], [-7000, 7000, 0], [0, 7000, 0]])
    assert cylindrical_sunlit(positions, np.array([1e8, 0, 0])).tolist() == [
        True,
        False,
        True,
        True,
    ]
    assert cylindrical_sunlit(positions, np.array([-1e8, 0, 0])).tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_actual_orbit_has_both_eclipse_and_sunlight():
    model = OrbitModel(satellite_ids=(1,))
    time = epoch(model)
    assert not model.snapshot(time).sunlit[0]
    assert model.snapshot(time + timedelta(minutes=30)).sunlit[0]


def test_original_directory_ids_are_preserved(tmp_path):
    lines = files("meteogym").joinpath("data/starlink_1080.tle").read_text().splitlines()
    (tmp_path / "TLE_Satellite_42.txt").write_text("\n".join(lines[1:3]) + "\n")
    model = OrbitModel(tmp_path, satellite_ids=(42,))
    assert model.satellite_ids == (42,)
    assert model.constellation_size == 1
    assert np.isfinite(model.snapshot(epoch(model)).positions_km).all()


def test_sparse_contact_export_preserves_ids_and_includes_endpoint(tmp_path):
    model = OrbitModel(satellite_ids=(100,))
    time = epoch(model)
    user = model.snapshot(time).geographic[:, :2]
    path = tmp_path / "contacts.csv"
    assert export_contact_table(model, user, time, 30, 1, 10, path) == 2
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["elapsed_s"]) for row in rows] == [0, 30]
    assert {row["satellite_id"] for row in rows} == {"100"}
    assert {row["user_id"] for row in rows} == {"0"}
    assert all(float(row["elevation_deg"]) >= 10 for row in rows)
