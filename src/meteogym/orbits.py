"""SGP4 orbits, ground contacts, and a cylindrical Earth-shadow model.

SGP4 returns TEME coordinates. Astropy converts them to GCRS for the Sun and
orbital observations, and to ITRS for contacts with fixed WGS84 ground sites.
No Earth-orientation data or solar ephemerides are downloaded at runtime.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    GCRS,
    ITRS,
    TEME,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    get_sun,
)
from astropy.time import Time
from astropy.utils import iers
from numpy.typing import NDArray
from sgp4.api import SGP4_ERRORS, Satrec, SatrecArray

# The installed astropy-iers-data package supplies historical Earth orientation.
iers.conf.auto_download = False
iers.conf.auto_max_age = None

EARTH_SHADOW_RADIUS_KM = 6378.137


@dataclass(frozen=True, slots=True)
class OrbitSnapshot:
    """Satellite rows follow ``OrbitModel.satellite_ids`` in caller order.

    Position and velocity use geocentric GCRS axes, in km and km/s.
    ``geographic`` columns are WGS84 latitude [deg], longitude [deg], height
    above the ellipsoid [km]. All returned arrays are read-only.
    """

    positions_km: NDArray[np.float64]
    velocities_km_s: NDArray[np.float64]
    geographic: NDArray[np.float64]
    sunlit: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ContactSnapshot:
    """Matrices have shape (selected satellites, ground users).

    Distances and elevations include invisible pairs. ``visible`` applies the
    elevation mask; consumers must use it before assigning a direct link.
    """

    visible: NDArray[np.bool_]
    distances_km: NDArray[np.float64]
    elevation_deg: NDArray[np.float64]


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


def _utc(time: datetime) -> datetime:
    """Interpret naive datetime values as UTC, never as the host timezone."""
    if not isinstance(time, datetime):
        raise TypeError("time must be a datetime")
    return (
        time.replace(tzinfo=timezone.utc) if time.tzinfo is None else time.astimezone(timezone.utc)
    )


def cylindrical_sunlit(
    positions_km: NDArray[np.float64],
    sun_position_km: NDArray[np.float64],
    earth_radius_km: float = EARTH_SHADOW_RADIUS_KM,
) -> NDArray[np.bool_]:
    """Test a parallel-ray cylindrical shadow using vectors in the same frame.

    A satellite is eclipsed when it is behind Earth relative to the Sun and
    its perpendicular distance to the Earth-Sun axis is below Earth's radius.
    This binary approximation omits penumbra, refraction, and panel attitude.
    """
    positions = np.asarray(positions_km, dtype=np.float64)
    sun = np.asarray(sun_position_km, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions_km must be a finite array with shape (satellites, 3)")
    if sun.shape != (3,) or not np.isfinite(sun).all() or np.linalg.norm(sun) == 0:
        raise ValueError("sun_position_km must be a finite nonzero vector with shape (3,)")
    if not np.isfinite(earth_radius_km) or earth_radius_km <= 0:
        raise ValueError("earth_radius_km must be finite and positive")
    direction = sun / np.linalg.norm(sun)
    axial = positions @ direction
    perpendicular = positions - axial[:, None] * direction
    shadowed = (axial < 0) & (np.linalg.norm(perpendicular, axis=1) < earth_radius_km)
    return ~shadowed


def _parse_tles(text: str, source: str) -> list[tuple[int, Satrec]]:
    """Accept ordinary two-line or named three-line TLE records."""
    lines = [
        line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    records: list[tuple[int, Satrec]] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("1 "):
            index += 1  # Optional satellite name.
        if (
            index + 1 >= len(lines)
            or not lines[index].startswith("1 ")
            or not lines[index + 1].startswith("2 ")
        ):
            raise ValueError(f"Malformed TLE record in {source}")
        sat = Satrec.twoline2rv(lines[index], lines[index + 1])
        records.append((sat.satnum, sat))
        index += 2
    if not records:
        raise ValueError(f"No TLE records in {source}")
    return records


class OrbitModel:
    """Propagate selected source IDs from the bundled or a custom constellation.

    The bundled synthetic constellation preserves IDs 1..1080. ``None`` selects
    every available satellite. A supplied directory supports the original
    ``TLE_Satellite_<id>.txt`` layout; a file accepts two- or three-line records,
    identified by their TLE catalog numbers. Selection order is preserved.

    The default start is the first selected satellite's TLE epoch, not today's
    date: the bundled synthetic elements describe an epoch in October 2016.
    """

    def __init__(
        self,
        tle_path: Path | None = None,
        satellite_ids: tuple[int, ...] | None = None,
    ) -> None:
        if tle_path is None:
            resource = files("meteogym").joinpath("data/starlink_1080.tle")
            records = _parse_tles(resource.read_text(encoding="utf-8"), str(resource))
        else:
            path = Path(tle_path)
            if path.is_dir():
                records = []
                paths = sorted(path.glob("TLE_*_*.txt"))
                if not paths:
                    raise ValueError(f"No TLE_*_<id>.txt files in {path}")
                for item in paths:
                    match = re.search(r"_(\d+)\.txt$", item.name)
                    if match is None:
                        continue
                    parsed = _parse_tles(item.read_text(encoding="utf-8"), str(item))
                    if len(parsed) != 1:
                        raise ValueError(f"Expected exactly one satellite in {item}")
                    records.append((int(match.group(1)), parsed[0][1]))
            else:
                records = _parse_tles(path.read_text(encoding="utf-8"), str(path))
        available = dict(records)
        if not available:
            raise ValueError("The constellation is empty")
        if len(available) != len(records):
            raise ValueError("Duplicate satellite IDs in the constellation")
        self.constellation_size = len(available)
        selected = tuple(sorted(available)) if satellite_ids is None else tuple(satellite_ids)
        if not selected:
            raise ValueError("Select at least one satellite")
        if any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
            for item in selected
        ):
            raise ValueError("Satellite IDs must be integers")
        if len(set(selected)) != len(selected):
            raise ValueError("Satellite IDs must be unique")
        missing = [item for item in selected if item not in available]
        if missing:
            raise ValueError(f"Unknown satellite IDs: {missing}")
        self.satellite_ids = tuple(int(item) for item in selected)
        self._satellites = [available[item] for item in self.satellite_ids]
        self._propagator = SatrecArray(self._satellites)
        first = self._satellites[0]
        epoch = Time(first.jdsatepoch, first.jdsatepochF, format="jd", scale="utc")
        self.default_start_time = (
            epoch.to_datetime(timezone=timezone.utc).isoformat().replace("+00:00", "Z")
        )
        self._cached_time: datetime | None = None
        self._cached_snapshot: OrbitSnapshot | None = None
        self._cached_itrs_positions: NDArray[np.float64] | None = None

    def snapshot(self, time: datetime) -> OrbitSnapshot:
        """Compute one instant, retaining only the most recent propagation."""
        instant = _utc(time)
        if instant == self._cached_time and self._cached_snapshot is not None:
            return self._cached_snapshot
        obstime = Time(instant, scale="utc")
        errors, position, velocity = self._propagator.sgp4(
            np.array([obstime.jd1]), np.array([obstime.jd2])
        )
        if np.any(errors):
            failed = [
                f"{self.satellite_ids[index]}: {SGP4_ERRORS.get(int(code), str(code))}"
                for index, code in enumerate(errors[:, 0])
                if code
            ]
            raise ValueError("SGP4 propagation failed for satellite " + "; ".join(failed))
        representation = CartesianRepresentation(position[:, 0, :].T * u.km)
        differential = CartesianDifferential(velocity[:, 0, :].T * u.km / u.s)
        teme = TEME(representation.with_differentials(differential), obstime=obstime)
        gcrs = teme.transform_to(GCRS(obstime=obstime))
        # Contacts require position only, so omit the velocity transformation.
        itrs = TEME(representation, obstime=obstime).transform_to(ITRS(obstime=obstime))
        positions = np.asarray(gcrs.cartesian.xyz.to_value(u.km).T)
        velocities = np.asarray(gcrs.velocity.d_xyz.to_value(u.km / u.s).T)
        terrestrial = np.asarray(itrs.cartesian.xyz.to_value(u.km).T)
        locations = EarthLocation.from_geocentric(*terrestrial.T, unit=u.km)
        geographic = np.column_stack(
            (locations.lat.deg, locations.lon.deg, locations.height.to_value(u.km))
        )
        sun = get_sun(obstime).cartesian.xyz.to_value(u.km)
        snapshot = OrbitSnapshot(
            _readonly(positions),
            _readonly(velocities),
            _readonly(geographic),
            _readonly(cylindrical_sunlit(positions, sun)),
        )
        self._cached_time = instant
        self._cached_snapshot = snapshot
        self._cached_itrs_positions = _readonly(terrestrial)
        return snapshot

    def contacts(
        self,
        time: datetime,
        user_lat_lon_deg: NDArray[np.float64],
        min_elevation_deg: float = 0.0,
    ) -> ContactSnapshot:
        """Direct geometric links to stationary users at WGS84 sea level.

        Visibility uses the dot product between the line of sight and local
        geodetic up, extending the original horizon test to an elevation mask.
        Atmosphere, terrain, and antenna/link-budget constraints are omitted.
        """
        users = np.asarray(user_lat_lon_deg, dtype=np.float64)
        if users.ndim != 2 or users.shape[1] != 2 or not np.isfinite(users).all():
            raise ValueError("user_lat_lon_deg must be a finite array with shape (users, 2)")
        if np.any(np.abs(users[:, 0]) > 90) or np.any(np.abs(users[:, 1]) > 180):
            raise ValueError("User latitude must be in [-90, 90] and longitude in [-180, 180]")
        if not np.isfinite(min_elevation_deg) or not 0 <= min_elevation_deg <= 90:
            raise ValueError("min_elevation_deg must be between 0 and 90")
        self.snapshot(time)
        locations = EarthLocation.from_geodetic(
            lon=users[:, 1] * u.deg, lat=users[:, 0] * u.deg, height=0 * u.km
        )
        ground = np.column_stack(
            [component.to_value(u.km) for component in locations.to_geocentric()]
        )
        latitude, longitude = np.deg2rad(users).T
        up = np.column_stack(
            (
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            )
        )
        relative = self._cached_itrs_positions[:, None, :] - ground[None, :, :]
        distances = np.linalg.norm(relative, axis=2)
        sine_elevation = np.einsum("suj,uj->su", relative, up) / distances
        elevations = np.rad2deg(np.arcsin(np.clip(sine_elevation, -1.0, 1.0)))
        return ContactSnapshot(
            _readonly(elevations >= min_elevation_deg),
            _readonly(distances),
            _readonly(elevations),
        )


def export_contact_table(
    model: OrbitModel,
    user_lat_lon_deg: NDArray[np.float64],
    start: datetime,
    dt: float,
    steps: int,
    min_elevation_deg: float,
    path: Path,
) -> int:
    """Write visible sampled edges at t=0..steps*dt (both endpoints included).

    Return the number of data rows. User IDs are zero-based row indices into
    ``user_lat_lon_deg``. This is a sampled contact table, not interpolated
    continuous access windows; a pass shorter than ``dt`` can be missed.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    start = _utc(start)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_utc", "elapsed_s", "satellite_id", "user_id", "distance_km", "elevation_deg"]
        )
        for step in range(steps + 1):
            instant = start + timedelta(seconds=step * dt)
            contacts = model.contacts(instant, user_lat_lon_deg, min_elevation_deg)
            for sat_index, user_index in zip(*np.nonzero(contacts.visible)):
                writer.writerow(
                    [
                        instant.isoformat().replace("+00:00", "Z"),
                        step * dt,
                        model.satellite_ids[sat_index],
                        user_index,
                        contacts.distances_km[sat_index, user_index],
                        contacts.elevation_deg[sat_index, user_index],
                    ]
                )
                rows += 1
    return rows
