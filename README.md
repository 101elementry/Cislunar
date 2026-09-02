# Cislunar

Numerical foundation for cislunar spacecraft dynamics: the Earth-Moon
circular restricted three-body problem, a halo-orbit differential
corrector, and a small mission-analysis tool built on top of them.

## Layout

Three layers with a one-way dependency: `app` uses `model`, `model`
uses `engine`, `engine` imports nothing above it.

| Layer | Contents | Rules |
|---|---|---|
| `engine/` | dynamics (`crtbp`), corrector, frames, propagation, geometry, photometry, constraints, access | numpy arrays and plain values in and out; no plotting, no file IO, no Dash, no model objects |
| `model/` | `Scenario` and its objects, JSON save/load, the halo-family file, `runner.run_scenario` | knows nothing about display |
| `app/` | the Dash interface and Plotly figure builders | callbacks only read the model and call the engine through the runner |

Top-level `validate.py` and `plots.py` build and check the halo family
and make the thesis figures with matplotlib; `scripts/` holds worked
examples that use the model and engine without any interface.

## Running

```
pip install numpy scipy matplotlib plotly dash
python validate.py                                  # builds output/halo_family.npz, prints checks
python plots.py                                     # thesis figures in output/
python -m app.main                                  # interface at http://127.0.0.1:8050
python scripts/sweep_min_elevation.py               # GUI-free example writing a CSV
```

## Setting up your own simulation

A scenario is a time span plus objects.  Spacecraft are propagated over
the span; every ground station (or sensor on a station) is then paired
with every spacecraft and the access constraints are evaluated on the
time grid.

### In the interface

1. Set the epoch (UTC), duration and time step in the top bar.  The
   epoch fixes where the Sun and the Earth's rotation are at time zero.
2. Add spacecraft.  A spacecraft is defined either by a **halo family
   member** (index into `output/halo_family.npz`; index 0 is the largest
   halo, the last index has the lowest perilune) or by an **initial
   state** in the rotating frame.  Family members can be propagated as
   "periodic" (repeat the converged orbit exactly, like a station-kept
   vehicle) or "integrate" (integrate the initial state, so an unstable
   halo eventually departs).  "Add several halo family members" in the
   tree panel adds a whole range at once.
3. Add a ground station (latitude, longitude, altitude, elevation
   cutoff, darkness threshold) and optionally an optical sensor on it
   (limiting magnitude, lunar exclusion angle).
4. Click an object in the tree, edit its fields, press **Apply**.
5. Press **Run analysis**.  The 3D view shows every trajectory; the
   windows panel shows one observer-spacecraft pair at a time (choose it
   in the dropdown); the time series shows that pair's elevation,
   magnitude and lunar separation with the windows shaded.
6. **Save JSON** to keep the scenario.  Use the **Panels** buttons in the
   top bar to hide the tree, the windows table or the time series when
   you want the orbit view to have the room.

### From a script

The interface does nothing a script cannot.  The pattern is

```python
from model.scenario import Scenario, Spacecraft, GroundStation, OpticalSensor
from model.family import load_family
from model import runner

scenario = Scenario.load("output/example_scenario.json")   # or build one
scenario.add(Spacecraft(name="Halo #30", source="family", family_index=30, propagation="periodic"))
results = runner.run_scenario(scenario, load_family())

for (observer, spacecraft), windows in results["windows"].items():
    print(observer, spacecraft, results["duty_cycle"][(observer, spacecraft)], windows)
```

`results["trajectories"][name]` is the (n, 6) state history of each
spacecraft, `results["observations"][(observer, spacecraft)]["geometry"]`
holds the elevation, range, magnitude and separation arrays, and
`["constraint_masks"]` is the per-constraint pass table.  Two worked
examples in `scripts/`:

- `sweep_min_elevation.py` sweeps the elevation cutoff and writes duty
  cycle and per-constraint pass fractions to CSV.
- `compare_family_members.py` runs the same observer against every
  N-th family member and tabulates period, perilune, stability index
  and observability along the family.

### Reading the windows panel and time series

An **access window** is a run of consecutive time steps in which every
constraint passes at once for the chosen observer-spacecraft pair.  The
constraints are: spacecraft above the station's elevation cutoff,
station in darkness (Sun below the threshold), spacecraft sunlit (not in
the Earth's or Moon's shadow), spacecraft brighter than the sensor's
limiting magnitude, and line of sight farther from the Moon than the
exclusion angle.  The chips under the summary give the fraction of the
span each constraint alone would allow, which tells you which one is
doing the cutting.  **Duty cycle** is the fraction of the whole span
that lies inside windows.

The **time series** plots the three continuous quantities the
constraints test against, with their thresholds as dashed lines and the
windows shaded green.  Darkness and shadow are boolean and are not
drawn.  The magnitude axis is reversed so that brighter is up.  The
black vertical line is the slider time, which also positions the
markers in the 3D view.

## Access constraints

A constraint is any function `constraint(step) -> bool` where `step`
is an `engine.geometry.StepGeometry` (elevation, Sun elevation, range,
lunar separation, phase angle, shadow flag, apparent magnitude at one
instant).  `engine/constraints.py` provides factories for elevation
cutoff, station darkness, target illumination, limiting magnitude and
lunar exclusion; `engine/access.py` evaluates any list of them without
knowing what they test.  Add a constraint by writing another factory
and appending its result to the list (`runner.run_scenario` accepts
`extra_constraints`).
