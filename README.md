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
