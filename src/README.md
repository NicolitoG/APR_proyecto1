# MeteoGym

A standalone Gymnasium environment for deciding which LEO satellites accept tasks from ground users.
Binary actions enable or disable each selected satellite computation payload; orbital contacts, task deadlines, and battery availability determine outcomes.
Reward is completed tasks minus failed tasks.

Runtime dependencies are **NumPy, Gymnasium, SGP4, and Astropy**.

## Installation

Python 3.11 or later is supported.

### System prerequisites

Install Python, virtual-environment support, pip and the optional Tk GUI toolkit through your distribution's package manager before creating the virtual environment.

For Ubuntu 24.04 or later:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk
```

Arch Linux:

```bash
sudo pacman -Syu --needed python python-pip tk
```

| Component | Ubuntu package | Arch Linux package | Purpose |
| --- | --- | --- | --- |
| Python 3 | `python3` | `python` | Runs MeteoGym; version 3.11 or later is required |
| Virtual environments | `python3-venv` | Included in `python` | Creates an isolated environment for pip dependencies |
| pip | `python3-pip` | `python-pip` | Installs MeteoGym and its Python dependencies |
| Tk/Tkinter | `python3-tk` | `tk` | Enables Matplotlib's TkAgg interactive windows |

Tk is optional for simulations and a PNG/PDF/SVG/GIF export can be made with `--no-show` option.

### Python installation

From the repository root, install with standard Python and pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./src
meteogym --help
```

Use `python -m pip install './src[plot]'` to include plotting, or `python -m pip install -e './src[dev,plot]'` for an editable development installation with tests and plotting.
The package installs compatible runtime dependencies automatically using the ranges declared in `pyproject.toml`.

For workflows that install dependencies separately, `requirements.txt` contains the same runtime dependency ranges and is compatible with pip:

```bash
python -m pip install -r src/requirements.txt
python -m pip install ./src
```

Run the development checks after installing the development and plotting extras:

```bash
python -m pip install -e './src[dev,plot]'
python -m pytest src/tests
python -m ruff check src
```

### Check interactive plotting

After activating the virtual environment, run this check from a graphical desktop session:

```bash
python -m tkinter
```

A small window confirms that the current interpreter can load Tk and reach the display.
If the module is missing, install the matching Tk package from the prerequisites above.
If it reports that no display is available, run the viewer from a graphical session or export with `--save map.gif --no-show`.

```bash
python -m meteogym.plots.plot_results ./outputs/run/ --mode animate
```

Replace `./outputs/run/` with the directory containing your saved dataset.
See [plot saved results](#plot-saved-results) for animation, step-by-step playback and exports without a window.

## Run selected satellites

After installation, from the repository root:

```bash
meteogym run --satellites 1 541 --users 100 \
  --duration 5400 --step 10 --seed 10 --policy active \
  --output-dir src/outputs/two-satellites
```

The bundled source constellation has **1,080 satellites** conformed of 72 orbital planes with 15 satellites each.
They are ordered first by ID and second by plane, so the satellite 16 is the first satellite of the second plane. 
This command propagates and controls only IDs 1 and 541;
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
Use `--user-location LAT LON` for fixed users; this sets their count automatically.
`--initial-tasks` starts one task per user.
The default `--task-arrival-mode single` permits one pending task per user.
Use `--task-arrival-mode queued` to keep generating tasks during a backlog and attempt the accumulated batch when a satellite becomes available:

```bash
meteogym run --satellites 1 541 --users 100 --task-arrival-mode queued \
  --task-rate 1 3 --minimum-voltage 5.5 --duration 5400 --step 10 \
  --output-dir src/outputs/queued-tasks
```

An assigned batch has one simulation step to finish; its unfinished tasks fail and are discarded.
Users without an assigned satellite retain their queues.
The [workload rules](#workload-and-time-discretization) describe deadlines, bandwidth and battery failures.

Policies `active`, `passive`, `random`, and `visible` are simple action generators for comparisons, with no learning framework.
`visible` requests activation when a satellite sees at least one user.
Battery limits override every action given to satellites.
`python -m meteogym` is equivalent to `meteogym`; see `meteogym run --help`.

Each run writes `summary.json` with full configuration, resolved UTC epoch, source IDs, seed, policy, versions, actual user coordinates/rates, totals and final SoC.
`steps.jsonl` records timestamps, requested/effective modes, starting contact and sunlight state, ending battery diagnostics, task energy, and reward.
It also records `pending_tasks_per_user`, `task_events`, `battery_failure_events`, `battery_suspended`, and each satellite's `active_duration_s`, `satellite_geographic_deg_km`, `user_online`, and `battery_voltage_start_v` at each interval's start.
Existing output files are replaced; choose separate directories for separate experiments.

## Plot saved results

Install the optional Matplotlib/Pillow dependencies from the repository root:

```bash
python -m pip install './src[plot]'
```

Plotting remains optional for simulation-only installations.
The standalone file is `plots/plot_results.py` at the repository root; `meteogym-plot` and `python -m meteogym.plots.plot_results` provide the same interface after installation.

```bash
# Animate an existing dataset.
python plots/plot_results.py src/outputs/two-satellites --mode animate --interval 200

# Advance with Enter while the plot window has focus.
python plots/plot_results.py src/outputs/two-satellites --mode step
```

Large circles show satellites and small dark circles show present ground users; offline users appear as pale crosses.
Satellite marker color represents battery voltage on a fixed scale from the saved battery's empty voltage (red) to full voltage (blue).
Satellites suspended after battery discharge appear gray until they recover.
Each adjacent label gives the source satellite ID, number of users in contact, and whether stored charge increased during the step.
Label text is green when processing is effectively active and red when inactive, including battery-imposed suspension.
The title shows cumulative generated, failed and completed tasks and cumulative reward through the end of the displayed step.
The colorbar uses the configured voltage limits.

Save a static frame or an animation without opening a window:

```bash
python plots/plot_results.py src/outputs/two-satellites \
  --frame 15 --save src/outputs/two-satellites/map.png --no-show
python plots/plot_results.py src/outputs/two-satellites \
  --save src/outputs/two-satellites/map.gif --interval 200 --no-show
```

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


## Gymnasium API

Importing `meteogym` registers `MeteoGym-v0` with Gymnasium.
Your model receives the observation dictionary and must return one binary action per selected satellite.
The above example uses a small policy; replace `AvailabilityModel.predict()` with your model's inference call.

```python
import gymnasium as gym
import meteogym
from meteogym import SimulationConfig


class AvailabilityModel:
    def predict(self, observation):
        has_users = observation["contact_count"] > 0
        enough_charge = observation["battery_soc"] >= 0.4
        return (has_users & enough_charge).astype("int8")


config = SimulationConfig(satellite_ids=(1, 541), num_users=100, duration_s=5400, step_s=10)
env = gym.make("MeteoGym-v0", config=config)
model = AvailabilityModel()
try:
    observation, info = env.reset(seed=10)
    while True:
        action = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

Use `env.observation_space` and `env.action_space` when constructing your model; model training and any ML dependencies belong to your application.
Configure `SimulationConfig` class accordingly to the experiment you want to reproduce.
Actions use `MultiBinary(S)`: `1` requests active mode and `0` passive mode.
Tasks still pending after normal deadline and batch processing remain censored at the horizon.
The final step is shortened when necessary.
`render_mode="ansi"` provides a short text status.

`info` includes cumulative generated/completed/failed/pending task counts, task events, assignments, requested/effective modes and per-satellite task joules.
`pending_tasks_per_user` contains one count per user, while `active_duration_s` contains the effective processing duration for each satellite in the last step.
`battery_failure_events` records the source satellite ID, elapsed failure time and terminal voltage for each queued-mode cutoff.
Assignment arrays use satellite row indices (`-1` means none); event records use source satellite IDs.
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
| `battery_suspended` | `(S,)` | Processing suspended by the mode's battery cutoff |
| `user_online` | `(U,)` | User presence |
| `user_positions_deg` | `(U,2)` | Ground latitude/longitude |
| `contact_mask`, `contact_count` | `(S,U)`, `(S,)` | Present users visible and their count |
| `user_ids` | `(S,U)` | Visible user's column ID; otherwise -1 |
| `distance_km`, `elevation_deg` | `(S,U)` | Visible slant range/elevation; otherwise zero |
| `task_present` | `(S,U)` | Visible user has a pending task |
| `task_size_mb`, `task_remaining_mb` | `(S,U)` | First task's original size and remaining upload to this satellite; MB |
| `task_time_left_s` | `(S,U)` | First task's seconds until expiry |
| `task_queue_length` | `(S,U)` | Visible user's number of pending tasks |
| `task_queue_size_mb`, `task_queue_remaining_mb` | `(S,U)` | Total queued original size and remaining upload to this satellite; MB |

Task fields are zero outside visibility or without a pending task.
Observation arrays are independent snapshots.

## Workload and time discretization

- Each user's rate is drawn uniformly from `task_rate_per_minute` at reset (default 1–3 tasks/minute).
  Waiting times are exponential.
  Configure the range with `--task-rate MIN MAX`, the JSON field `task_rate_per_minute`, or the same Python configuration field.
- In default `single` mode, a user has at most one pending task and the next waiting period starts at the step boundary after success or failure.
  Its creation timestamp and relative deadline start at boundary admission.
- In `queued` mode, arrivals continue independently of completion and accumulate in a per-user FIFO queue.
  Every sampled arrival through the current boundary is admitted, including arrivals during the final episode step.
  Creation timestamps and deadlines retain the exact sampled arrival time, so a task can expire before its first transmission opportunity.
  New arrivals become eligible for transmission only at the next decision boundary.
  Per-satellite bandwidth is shared equally among assigned users for the whole slot.
  Unused capacity is redistributed only at the next slot.
  MB is decimal; `uplink_mbps / 8` gives total MB/s.
- In `single` mode, uploads progress across slots.
  A different destination discards the partial upload; loss of contact pauses it, allowing resumption with the same satellite.
- In `queued` mode, assignment commits the user's current queue as a batch for that step.
  Each user uploads its tasks in FIFO order and can complete several during the same slot.
  Tasks still unfinished at the slot's end fail with `transfer_incomplete` and are discarded, including tasks the available bandwidth could not start.
- Full upload is success, including completion exactly at the deadline.
  Transfer is capped by time before expiry; an incomplete task then fails with `deadline`.
  Deadlines also apply to queued tasks waiting behind another task or without contact.
  There is no CPU queue or result downlink.
  Communication/processing cost is represented by `task_energy_j_per_mb` (default 2 J/MB), charged for actual uploaded bytes, including partial and failed uploads.
- Optional `user_departure_rate_per_minute` and `user_return_rate_per_minute` enable user presence changes.
  Defaults are zero.
  Boundary transition probability is `1-exp(-rate*dt/60)`.

Geometry, sunlight, bandwidth shares and requested actions are held constant during a step.
The maximum discharge current can throttle transfer throughput.
Reduce `step_s` and check convergence for short passes, short deadlines, or cutoff-sensitive experiments.
In queued mode, `step_s` also defines the allowed transfer window for each assigned batch, so changing it changes that workload policy.

## Orbits and contact tables

The synthetic dataset preserves the original `starlink_1080` TLE lines and IDs: 72 planes × 15 satellites, nominal 550 km altitude and 53° inclination.
Its epoch is **2016-10-29T19:52:11.055936Z** which creates an experimental constellation.
Simulations default to that epoch.
`--start-time` accepts a timezone-aware ISO timestamp; use suitably dated `--tle-path` data for different epochs.
Custom data may be a combined TLE file.

SGP4, geometric visibility and Euclidean slant ranges use contact logic.
Astropy transforms TEME into GCRS for state observations and ITRS for WGS84 users.
The default elevation mask is 0°; `--min-elevation` raises it.
See [dataset provenance](meteogym/data/README.md) and [Astropy satellite coordinates](https://docs.astropy.org/en/stable/coordinates/satellites.html).

Astropy's computes the Sun in GCRS.
A satellite is eclipsed behind Earth when its perpendicular distance to the Earth–Sun axis is below 6378.137 km.

```bash
meteogym contacts --satellites 1 541 --users 100 --duration 5400 \
  --step 10 --seed 10 --output src/outputs/contacts.csv
```

The CSV contains `time_utc,elapsed_s,satellite_id,user_id,distance_km,elevation_deg` for visible pairs only.
Samples include zero through `floor(duration/step)*step`.
Missing pairs are invisible at that sample.
`.metadata.json` records users, configuration and seed.
Contacts describe geometry independently other changes.

## Energy and configuration

The battery defaults use an illustrative 2S pack with 2.6 Ah capacity, 5.5–8.4 V open-circuit limits and 0.14 Ω internal resistance.
These settings are configurable and can change in the simulation.
The original assumptions are based on published cell scales specifications.
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
In `single` mode, processing stops at 20% SoC and resumes after reaching 30%; baseline and task charge are reserved before admitting work.
In `queued` mode, processing continues until the loaded terminal voltage reaches `battery.minimum_voltage_v`, which defaults to `voltage_empty_v` when omitted or set to `None`.
Set this threshold with `--minimum-voltage V`, the JSON battery field `minimum_voltage_v`, or `BatteryConfig(minimum_voltage_v=V)` in Python.
Recovery requires actual recharge after the failure, SoC at least `resume_soc`, and open-circuit voltage above the threshold; processing can resume at the next decision boundary.
Both modes also require open-circuit voltage at least `battery.resume_voltage_v` before processing resumes after discharge.
This additional recovery threshold defaults to 5.5 V, matching the default `voltage_empty_v`, and can be set independently through JSON or `BatteryConfig(resume_voltage_v=7.5)` in Python.
A sunlit satellite can discharge if its loads exceed generation.

The [fast discharge battery configuration](configs/fast-discharge.json) gives an example for a battery with reduced-capacity.
From the repository root it can be run:

```bash
python -m meteogym run --config src/configs/modified-battery.json \
  --satellites 1 541 --output-dir outputs/modified-battery
```

# Experiment Configuration

All simulation settings can be supplied through Python or JSON.
The complete [default configuration](configs/default.json) lists every supported JSON field with its default value, including all battery parameters.
Run it from the repository root with `meteogym run --config src/configs/default.json` or copy it to create an experiment.
`null` selects the bundled constellation for `tle_path`, the TLE epoch for `start_time`, randomized coordinates for `user_locations_deg`, and `voltage_empty_v` for `battery.minimum_voltage_v`.
Policy, seed, output paths and plotting options are configured through CLI arguments.
For example, save this as `experiment.json` and run `meteogym run --config experiment.json --seed 10`:

```json
{
  "satellite_ids": [1, 541],
  "num_users": 100,
  "duration_s": 5400,
  "step_s": 10,
  "task_arrival_mode": "queued",
  "task_rate_per_minute": [1, 3],
  "task_energy_j_per_mb": 2.0,
  "battery": {"minimum_voltage_v": 5.5, "resume_soc": 0.3, "resume_voltage_v": 5.5}
}
```

Unspecified settings retain defaults; CLI arguments override JSON.
To replay a summary, extract its `config` object and use its seed/policy.

## Validation and structure

Tests cover the [Gymnasium environment contract](https://gymnasium.farama.org/api/env/), seeded replay, tasks/deadlines/handovers, shared capacity, user presence, battery energy conservation and cutoff, shadow geometry, coordinate transforms, dataset integrity and CLI exports.

```text
meteogym/
  config.py       Validated experiment parameters
  orbits.py       SGP4, Sun, direct contacts, CSV export
  battery.py      Linear voltage, resistance, solar/load integration
  workload.py     Users, gated arrivals and accumulating task queues
  env.py          Gymnasium actions, observations, reward
  cli.py          Runs and contact exports
  plots/          Dataset replay, Matplotlib viewer and offline map data
  data/           Synthetic TLEs and provenance
docs/             Physical assumptions and sources
tests/            Behavior and scientific boundary checks
```
