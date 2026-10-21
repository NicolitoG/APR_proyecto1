# Bundled orbital elements

`starlink_1080.tle` contains the synthetic constellation used by the source `sat-edge-computation/satellite_constellation` project, from `tles/starlink_1080` on branch `internet-computing-2026`, commit `744c65a`.
It has 72 orbital planes with 15 satellites each, 53-degree inclination, and a nominal altitude of 550 km.
These are synthetic experimental elements, not contemporary operational Starlink TLEs.

Satellite IDs remain **1 through 1080**.
The original two lines of every `TLE_Satellite_<id>.txt` file are concatenated in numeric ID order with an added `SATELLITE <id>` name line.
The extra simulation fields following the standard second TLE line are retained; SGP4 ignores those trailing fields.
`starlink_1080.provenance.json` records the bundled file checksum and the SHA-256 checksum of each original file.

All bundled elements have epoch **2016-10-29T19:52:11.055936Z**.
Simulations default to this epoch.
Propagating these elements years away from the epoch does not produce a physically reliable contemporary constellation.

The contact calculation retains SGP4, Euclidean slant range, and the original horizon line-of-sight criterion, with a configurable elevation mask.
Ground users are now fixed WGS84 sites, using Astropy's TEME-to-ITRS transformation instead of the source's ground rotation initialized without the epoch's sidereal angle.
This correction intentionally changes geographic contact times relative to the source simulator.

Astropy also converts satellite states to GCRS and computes the Sun in that same reference frame.
A cylindrical Earth shadow with radius 6378.137 km defines orbital night.
The model assumes parallel solar rays and omits penumbra, atmosphere, and solar-panel attitude.
Runtime uses the installed Earth-orientation tables without downloads.

References: [Astropy satellite coordinates](https://docs.astropy.org/en/stable/coordinates/satellites.html), [Astropy get_sun](https://docs.astropy.org/en/stable/api/astropy.coordinates.get_sun.html), [Astropy IERS data](https://docs.astropy.org/en/stable/utils/iers.html).
