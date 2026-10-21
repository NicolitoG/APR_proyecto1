# MeteoGym

A standalone Gymnasium environment for deciding which LEO satellites accept tasks from ground users.
Binary actions enable or disable each selected satellite; orbital contacts, task deadlines, and battery availability determine outcomes.
Reward is completed tasks minus failed tasks.

Runtime dependencies are **NumPy, Gymnasium, SGP4, and Astropy**.
The environment uses no Docker, MongoDB, Mininet, ML framework, gateway, inter-satellite link, or MEC placement strategy.
Simulations run offline after installation.

## Installation

From the repository root, use the existing Conda environment:

```bash
conda activate meteogym
python -m pip install -e './src[dev]'
python -m pytest src/tests
ruff check src
```

For a new environment, run **from `src/`**, so the editable path resolves:

```bash
conda env create -f environment.yml
conda activate meteogym
```

Python 3.11 or later is supported.
`requirements-lock.txt` records the exact Python-package versions used for validation; it is not a cross-platform Conda lock.
On the development host, Conda is in `/opt/miniconda3` and the environment's Python is `/home/kaminari/.conda/envs/meteogym/bin/python`.
The Conda entrypoint may need `CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1` for its OpenSSL initialization.

## Run selected satellites

After installation, from the repository root:

```bash
meteogym run --satellites 1 541 --users 100 \
  --duration 5400 --step 10 --seed 10 --policy active \
  --output-dir src/outputs/two-satellites
```

The bundled source constellation has **1,080 satellites**.
This command propagates and controls only source IDs 1 and 541; other satellites cannot receive tasks.
`--satellites 541` selects exactly one satellite.
`--all-satellites` includes the whole constellation.
Default selection is IDs 1 and 2.

```bash
meteogym run --all-satellites --users 100 --duration 600 --step 10
meteogym run --satellites 1 541 --users 50 \
  --user-region -56 -17 -76 -66 --task-rate 1 3 \
  --task-size 0.1 2 --deadline 60 180 --uplink-mbps 1
```

Users are sampled uniformly in surface area within latitude/longitude bounds, or across the Earth by default.
Two selected satellites can have few contacts with a small random population.
Repeat `--user-location LAT LON` for fixed users; this sets their count automatically.
`--initial-tasks` starts one task per user.

Policies `active`, `passive`, `random`, and `visible` are simple action generators for comparisons, with no learning framework.
`visible` requests activation when a satellite sees at least one present user.
Battery limits override every policy.
`python -m meteogym` is equivalent to `meteogym`; see `meteogym run --help`.

Each run writes `summary.json` with full configuration, resolved UTC epoch, source IDs, seed, policy, versions, actual user coordinates/rates, totals and final SoC.
`steps.jsonl` records timestamps, requested/effective modes, starting contact and sunlight state, ending battery diagnostics, task energy, and reward.
Output schema version 2 also saves `satellite_geographic_deg_km`, `user_online`, and `battery_voltage_start_v` at each interval's start, so visualization can replay the recorded state without the original TLE file.
Existing output files are replaced; choose separate directories for separate experiments.

## Plot saved results

Install the optional Matplotlib/Pillow dependencies in `meteogym` from the repository root:

```bash
python -m pip install -e './src[plot]'
```

The Conda `environment.yml` includes these extras when creating a new development environment.
Plotting remains optional for simulation-only installations.
The standalone entry point is `plots/plot_results.py` at the repository root; `meteogym-plot` and `python -m meteogym.plots.plot_results` provide the same interface after installation.

```bash
# Animate an existing dataset.
python plots/plot_results.py src/outputs/two-satellites --mode animate --interval 200

# Advance with Enter while the plot window has focus.
python plots/plot_results.py src/outputs/two-satellites --mode step
```

The longitude/latitude map uses bundled Natural Earth land outlines and requires no map downloads.
Large circles show satellites and small dark circles show present ground users; offline users appear as pale crosses.
Satellite marker color represents battery voltage on a fixed scale from the saved battery's empty voltage (red) to full voltage (blue).
Each adjacent label gives the source satellite ID and number of users in contact.
Label text is green when processing is effectively active and red when inactive, including battery-imposed suspension.
The colorbar uses the configured pack voltage limits, and values outside those limits are clipped to the endpoint colors.

Press Enter or Right to advance, Left or Backspace to go back, Home/End to select the first/last frame, and Q or Escape to close.
In animation mode, Space pauses/resumes playback; manual navigation pauses the animation.
Frames show interval-start positions, contacts, presence, and battery voltage, paired with the effective processing mode used during that interval.
There is one frame per recorded simulation step, including a shortened final interval when present.
The existing end-of-interval voltage fields remain available in the dataset and are not used as start-state values.

Save a static frame or an animation without opening a window:

```bash
python plots/plot_results.py src/outputs/two-satellites \
  --frame 15 --save src/outputs/two-satellites/map.png --no-show
python plots/plot_results.py src/outputs/two-satellites \
  --save src/outputs/two-satellites/map.gif --interval 200 --no-show
```

`--frame` is a zero-based initial frame index; PNG, PDF and SVG save that frame, while GIF exports the episode.
`--stride N` saves every Nth GIF frame and always includes the final frame, which reduces file size and memory use for long runs.
Interactive display requires a GUI-capable Matplotlib backend, such as TkAgg; headless sessions should use `--save ... --no-show`.
GIF export uses Pillow and does not require FFmpeg.

To plot immediately after a new simulation:

```bash
meteogym run --satellites 1 541 --users 100 --duration 600 \
  --output-dir src/outputs/visual-run --plot step
meteogym run --satellites 1 541 --users 100 --duration 600 \
  --output-dir src/outputs/animated-run --plot --plot-interval 150
meteogym run --satellites 1 541 --users 100 --duration 60 \
  --output-dir src/outputs/export-run --plot \
  --plot-save src/outputs/export-run/map.gif --plot-no-show
```

Older datasets without saved satellite coordinates reconstruct positions from their saved epoch, selected IDs, and configured TLE using the currently installed astronomy libraries.
The viewer reports this reconstruction because changed TLE files or astronomy versions can change the result.
Legacy start voltage comes from the previous step's end voltage, with the initial open-circuit voltage used for the first step.
Legacy datasets with user departures cannot recover exact historical user presence and must be regenerated; current outputs save it in every run.
User coordinates always come from `summary.json`, rather than being randomly generated again.

## Gymnasium API

```python
import gymnasium as gym
import meteogym
from meteogym import SimulationConfig

config = SimulationConfig(satellite_ids=(1, 541), num_users=100, duration_s=5400, step_s=10)
env = gym.make("MeteoGym-v0", config=config)
observation, info = env.reset(seed=10)
env.action_space.seed(10)
while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
env.close()
```

Actions use `MultiBinary(S)`: `1` requests active mode and `0` passive mode, in selected-ID order.
Episode duration returns `truncated=True`, `terminated=False`.
Pending tasks remain censored at the horizon, not artificial failures.
The final step is shortened when necessary.
Call `reset()` before another episode.
`render_mode="ansi"` provides a short text status.

`info` includes cumulative generated/completed/failed/pending task counts, task events, assignments, requested/effective modes and per-satellite task joules.
Assignment arrays use satellite row indices (`-1` means none); event records use source satellite IDs.
User IDs are zero-based.
Total reward equals completed minus failed tasks.

### Observation fields

Shapes stay fixed within an episode: `S` selected satellites, `U` user slots.
Floating fields use float64, masks int8, IDs/counts int64.
Physical values are not normalized for training.

| Key | Shape | Meaning / units |
| --- | --- | --- |
| `time_s` | `(1,)` | Elapsed seconds |
| `satellite_ids` | `(S,)` | Source IDs in action order |
| `position_km`, `velocity_km_s` | `(S,3)` | GCRS Cartesian state; km, km/s |
| `geographic_deg_km` | `(S,3)` | WGS84 latitude, longitude, altitude; deg, deg, km |
| `sunlit` | `(S,)` | Sunlight at the observation instant |
| `battery_soc` | `(S,)` | Stored charge fraction in [0,1] |
| `battery_voltage_v`, `battery_current_a` | `(S,)` | Terminal diagnostic V and A; positive current discharges |
| `charging`, `active` | `(S,)` | Charge increase and effective mode in the previous interval |
| `available` | `(S,)` | Battery lockout cleared; interval admission may still reject activation |
| `user_online` | `(U,)` | User presence |
| `user_positions_deg` | `(U,2)` | Ground latitude/longitude |
| `contact_mask`, `contact_count` | `(S,U)`, `(S,)` | Present users visible and their count |
| `user_ids` | `(S,U)` | Visible user's column ID; otherwise -1 |
| `distance_km`, `elevation_deg` | `(S,U)` | Visible slant range/elevation; otherwise zero |
| `task_present` | `(S,U)` | Visible pending task |
| `task_size_mb`, `task_remaining_mb` | `(S,U)` | Original size and remaining upload to this satellite; MB |
| `task_time_left_s` | `(S,U)` | Seconds until expiry |

Task fields are zero outside visibility or without a pending task.
Remaining upload is satellite-specific: a different destination requires the full task size because handover restarts the upload.
Observation arrays are independent snapshots.
Prior-interval mode/current fields are zero at reset.

## Workload and time discretization

- Each user's rate is drawn uniformly from `task_rate_per_minute` at reset (default 1–3 tasks/minute).
  Waiting times are exponential.
  A user has at most one pending task; the next waiting period starts after success or failure.
- Arrivals are admitted at the first decision boundary at or after their sampled time.
  Sizes and relative deadlines are sampled independently, uniformly from their configured ranges.
  Equal bounds give constant sizes/rates/deadlines.
- A user uploads to the **nearest active, battery-admitted satellite in view**.
  Per-satellite bandwidth is shared equally among assigned users for the whole slot.
  Unused capacity is redistributed only at the next slot.
  MB is decimal; `uplink_mbps / 8` gives total MB/s.
- Uploads progress across slots.
  A different destination discards the partial upload; loss of contact pauses it, allowing resumption with the same satellite.
- Full upload is success, including completion exactly at the deadline.
  Transfer is capped by time before expiry; an incomplete task then fails.
  There is no CPU queue or result downlink.
  Communication/processing cost is represented by `task_energy_j_per_mb` (default 2 J/MB), charged for actual uploaded bytes, including partial and failed uploads.
- Optional `user_departure_rate_per_minute` and `user_return_rate_per_minute` enable user presence changes.
  Defaults are zero.
  Boundary transition probability is `1-exp(-rate*dt/60)`.
  Departure fails a pending task; returning users retain their IDs/locations and start another wait.

Geometry, sunlight, bandwidth shares, and mode are held constant during a step.
Battery task current is averaged over that interval.
Following completion/failure, the next waiting period starts at its ending boundary.
Reduce `step_s` and check convergence for short passes, short deadlines, or cutoff-sensitive experiments.

## Orbits and contact tables

The synthetic dataset preserves the original `starlink_1080` TLE lines and IDs: 72 planes × 15 satellites, nominal 550 km altitude and 53° inclination.
Its epoch is **2016-10-29T19:52:11.055936Z**.
This is an experimental constellation, not a live Starlink catalog.
Simulations default to that epoch.
`--start-time` accepts a timezone-aware ISO timestamp; use suitably dated `--tle-path` data for different epochs.
Custom data may be a combined TLE file or the original directory of `TLE_Satellite_<id>.txt` files.

SGP4, geometric visibility and Euclidean slant ranges retain the source contact logic.
Astropy transforms TEME into GCRS for state observations and ITRS for WGS84 users.
This corrects the source's epoch-independent ground rotation, so geographic contact times can differ.
The default elevation mask is 0°; `--min-elevation` raises it.
See [dataset provenance](meteogym/data/README.md) and [Astropy satellite coordinates](https://docs.astropy.org/en/stable/coordinates/satellites.html).

Astropy's [`get_sun`](https://docs.astropy.org/en/stable/api/astropy.coordinates.get_sun.html) computes the Sun in GCRS without an ephemeris download.
A satellite is eclipsed behind Earth when its perpendicular distance to the Earth–Sun axis is below 6378.137 km.
This cylindrical shadow omits penumbra, atmosphere and panel attitude.
Installed Earth-orientation tables are used with automatic downloads disabled.

```bash
meteogym contacts --satellites 1 541 --users 100 --duration 5400 \
  --step 10 --seed 10 --output src/outputs/contacts.csv
```

The CSV contains `time_utc,elapsed_s,satellite_id,user_id,distance_km,elevation_deg` for visible pairs only.
Samples include zero through `floor(duration/step)*step`.
Missing pairs are invisible at that sample.
`.metadata.json` records users, configuration and seed.
Contacts describe geometry independently of actions, workload, battery and user presence changes; they are sampled edges, not interpolated acquisition/loss event times.

## Energy and configuration

The illustrative 3U battery is a 2S1P pack, 2.6 Ah (about 18.7 Wh nominal), 5.5–8.4 V open-circuit limits and 0.14 Ω internal resistance.
Published cell scales support the estimate; these are not measured SUCHAI-3 specifications.
The [energy model](docs/energy-model.md) documents sources, equations and assumptions.

```text
V_oc = V_empty + (V_full - V_empty) * SoC
V_terminal = V_oc - I * R
I = mode_current + task_current - solar_current * sunlit
SoC_next = SoC - I * dt / (3600 * capacity_ah)
```

Charging uses a 0.95 storage efficiency.
Baseline loads are 0.12 A passive and 0.45 A active; sunlight supplies 0.65 A.
Constant mode currents produce linear piecewise depletion.
Processing stops at 20% SoC and resumes after reaching 30%.
Baseline and task charge are reserved before admitting work.
A sunlit satellite can discharge if its loads exceed generation.

Every setting can be supplied through Python or JSON.
For example, save this as `experiment.json` and run `meteogym run --config experiment.json --seed 10`:

```json
{
  "satellite_ids": [1, 541],
  "num_users": 100,
  "duration_s": 5400,
  "step_s": 10,
  "task_rate_per_minute": [1, 3],
  "task_energy_j_per_mb": 2.0,
  "battery": {"capacity_ah": 2.6, "cutoff_soc": 0.2, "resume_soc": 0.3}
}
```

Unspecified settings retain defaults; CLI arguments override JSON.
To replay a summary, extract its `config` object and use its seed/policy.
The summary itself is not a configuration file.

## Validation and structure

Tests cover the [Gymnasium environment contract](https://gymnasium.farama.org/api/env/), seeded replay, tasks/deadlines/handovers, shared capacity, user presence, battery energy conservation and cutoff, shadow geometry, coordinate transforms, dataset integrity and CLI exports.
Gymnasium may advise against infinite physical-state Box bounds; these deliberately permit custom orbital datasets.

```text
meteogym/
  config.py       Validated experiment parameters
  orbits.py       SGP4, Sun, direct contacts, CSV export
  battery.py      Linear voltage, resistance, solar/load integration
  workload.py     Users and gated randomized tasks
  env.py          Gymnasium actions, observations, reward
  cli.py          Runs and contact exports
  plots/          Dataset replay, Matplotlib viewer and offline map data
  data/           Synthetic TLEs and provenance
docs/             Physical assumptions and sources
tests/            Behavior and scientific boundary checks
```
