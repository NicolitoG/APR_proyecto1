# Energy model

MeteoGym uses a configurable internal-resistance battery model with linear open-circuit voltage and Coulomb counting.
It provides distinct passive, active, and task loads, orbital solar charging, and a processing cutoff with recharge hysteresis.
The model is intended for reproducible scheduling experiments.

## Evidence and assumptions

The [SUCHAI mission team's Small Satellite Conference paper](https://digitalcommons.usu.edu/smallsat/2023/all2023/56/) identifies SUCHAI-2, SUCHAI-3, and Plantsat as 3U spacecraft.
We did not establish a measured SUCHAI-3 battery specification; the defaults below are an illustrative 3U power scenario.

[Knap, Beczkowski, and Stroe (2020), *Development of a Model-Based Approach to Capture Battery Parameter Degradation in Satellites*](https://vbn.aau.dk/ws/files/627198939/vkn_abaf2020_v4.doc.pdf) reports GomSpace cells with 2.6 Ah nominal capacity, 3.6 V nominal voltage, and 2.75–4.2 V limits.
Table I gives a fitted cell ohmic resistance of 0.0697 ohm at 25 °C.
Their GOMX-3 spacecraft used four cells in series.
MeteoGym **assumes two such cells in series**, preserving capacity and doubling cell voltage and resistance.
This gives approximately 18.7 Wh at nominal 7.2 V; integrating our linear voltage curve gives 18.07 Wh over 0–100% SoC.
These differ because the linear approximation does not reproduce the real cell's discharge plateau.
The paper supports the component scale; it does not validate our simplified pack or load model.

A commercial example of a two-cell-series option for 1U–3U platforms is the [ISISPACE ICEPS2 manufacturer datasheet](https://isispace.nl/wp-content/uploads/2025/01/ISIS-ICEPS2-DSH-0001-ICEPS2_Datasheet-01_02.pdf), which lists a 7.2 V, 28.8 Wh option.
That product has a different capacity from our assumed pack.

| `BatteryConfig` field | Default | Basis |
| --- | ---: | --- |
| `capacity_ah` | 2.6 Ah | Published nominal cell capacity; assumed 2S1P pack |
| `voltage_empty_v` | 5.5 V | Twice the published 2.75 V cell lower limit |
| `voltage_full_v` | 8.4 V | Twice the published 4.2 V cell upper limit |
| `internal_resistance_ohm` | 0.14 ohm | Rounded twice the fitted 25 °C cell ohmic resistance |
| `passive_current_a` | 0.12 A | Assumed housekeeping load, roughly 0.86 W at 7.2 V |
| `active_current_a` | 0.45 A | Assumed total active baseline, roughly 3.24 W at 7.2 V |
| `solar_current_a` | 0.65 A | Assumed effective bus-side generation, roughly 4.68 W at 7.2 V |
| `max_discharge_current_a` | 2.5 A | Assumed simulation operating limit |
| `charge_efficiency` | 0.95 | Assumed charge-storage efficiency |
| `cutoff_soc` / `resume_soc` | 0.20 / 0.30 | Assumed processing suspension/resumption thresholds |

Active current already includes housekeeping.
Solar current represents generation after panel, attitude, and regulator effects have been summarized into one constant parameter.
The simulation does not require a battery or astronomy service over the network.

## Step equations

Let `s` be SoC in `[0, 1]`, `Q = 3600 * capacity_ah` in coulombs, `R` internal resistance, and `dt` the step duration in seconds.
For each satellite:

```text
V_oc(s) = V_empty + (V_full - V_empty) * s
I_base  = active_current if active else passive_current
I_solar = solar_current if sunlit else 0
I       = I_base + I_task - I_solar       # positive means discharge
V_term  = V_oc(s) - R * I
ds/dt   = -I / Q                         # discharging
ds/dt   = -charge_efficiency * I / Q      # charging
```

SoC and open-circuit voltage change linearly within a constant-mode step until a capacity bound is reached.
Solar generation supplies the load first: a sunlit satellite can still discharge when active/task consumption exceeds generation.
At full charge, surplus solar generation is curtailed.
At empty charge, passive demand beyond available energy is unserved.
SoC never leaves `[0, 1]`.

The internal-resistance term causes an immediate load-dependent voltage drop.
Removing the extra task current removes that drop, while the charge consumed by the task remains depleted.
Temperature, aging, polarization transients, nonlinear end-of-discharge behavior, regulator dynamics, and panel attitude are omitted.
The presentation's plotting script draws an illustrative single-cell-shaped curve; its arbitrary time/voltage parameters are not treated as battery data.

## Task energy and admission

Task energy is additional electrical work at the pack terminals, in joules.
The environment charges its configured joules per transferred MB as task upload progresses, representing the associated communication/processing cost.
A task succeeds when its complete upload is accepted; there is no separate CPU queue.

For a constant extra task current during one step:

```text
E_task = I_task * integral(V_term(t), t=0..dt)
       = I_task * dt * mean_terminal_voltage
```

The mean voltage is calculated from the exact piecewise-linear SoC trajectory, including capacity saturation.
Bounded bisection solves for `I_task`, so energy accounting uses the same voltage model as charge depletion and includes the extra resistive loss in the battery.
No fixed nominal-voltage conversion or second independent subtraction of task energy is used.
Without solar or baseline load, the chemical energy removed equals task work plus `I_task**2 * R * dt`.

Before accepting work, `plan_active` reserves the entire interval's active load.
If that would cross the cutoff, the satellite uses passive mode for the whole interval.
`task_energy_budget_j` then limits work by both remaining charge above the cutoff and the maximum battery discharge current.
A conservative monotonicity bound also keeps task-current inversion well-defined for unusually long steps.
Budgets for passive, locked-out, or unaffordable satellites are zero.

`advance` rejects task work beyond this budget without modifying the battery.
The environment must limit transferred bytes before marking tasks complete.
Reaching the cutoff locks out processing; it resumes only after SoC reaches `resume_soc`, starting on the following step.
Passive charging continues while processing is locked out.

```python
from meteogym.battery import BatteryConfig, BatteryPack

battery = BatteryPack(BatteryConfig(), count=2, initial_soc=[0.8, 0.5])
sunlit = [True, False]
requested_active = [True, True]
active = battery.plan_active(10.0, sunlit, requested_active)
budget_j = battery.task_energy_budget_j(10.0, sunlit, requested_active)
# Allocate task work no greater than budget_j for each satellite.
step = battery.advance(10.0, sunlit, requested_active, task_energy_j=[1.0, 2.0])
```

Every result has one element per selected satellite.
`BatteryStep.active` describes the interval just executed.
`charging` means stored charge actually increased.
`current_a` is the interval-average battery current, including solar curtailment or exhaustion; `voltage_v` is ending open-circuit voltage minus that reported current times resistance.
At a capacity boundary this reported voltage is a step diagnostic, not the instantaneous voltage after the controller switches.
