# Cislunar

Numerical foundation for cislunar spacecraft dynamics: the Earth-Moon
circular restricted three-body problem, periodic orbit families found
by differential correction and continuation, their stability and
manifolds, and a mission-analysis tool built on top of them.

## Layout

Three layers with a one-way dependency: `app` uses `model`, `model`
uses `engine`, `engine` imports nothing above it.

| Layer | Contents | Rules |
|---|---|---|
| `engine/` | dynamics (`crtbp`), corrector, families, manifolds, station keeping, estimation, rendezvous, kepler, frames, propagation, geometry, photometry, constraints, access | numpy arrays and plain values in and out; no plotting, no file IO, no Dash, no model objects |
| `model/` | `Scenario` and its objects, JSON save/load, family files, `orbits` (any spacecraft to a state, correction to periodic), `runner.run_scenario`, `sweep` | knows nothing about display |
| `app/` | the Dash interface and Plotly figure builders | callbacks only read the model and call the engine through the runner |

Top-level `validate.py` and `plots.py` build and check the L2 southern
halo family and make the thesis figures with matplotlib;
`build_families.py` builds the other families; `scripts/` holds worked
examples that use the model and engine without any interface.

## Running

```
pip install numpy scipy matplotlib plotly dash
python validate.py                          # checks + output/halo_family.npz (~20 s)
python build_families.py                    # L1 halos, northern halos, Lyapunov, DRO (~10 s)
python plots.py                             # thesis figures in output/
python -m app.main                          # interface at http://127.0.0.1:8050
```

Worked examples, all GUI-free, each writing a CSV to `output/`:

```
python scripts/sweep_min_elevation.py       # duty cycle against elevation cutoff
python scripts/compare_family_members.py    # observability along a family
python scripts/manifold_transfers.py        # manifold branches ranked by closest approach
python scripts/station_keeping_sweep.py     # delta-v per year against stability index
python scripts/orbit_determination.py       # rung 2: batch, EKF, UKF and their consistency
python scripts/geo_rendezvous.py            # GEO parking-orbit drift and two-burn rendezvous
python scripts/elfo_drift.py                # how frozen a lunar frozen orbit stays
python scripts/export_gmat.py               # GMAT script for a high-fidelity cross-check
```

## What the interface does

Set an epoch, duration and step.  Add spacecraft defined by an orbit
family member (L1/L2 halos north and south, Lyapunov, DRO; pick by
index or by perilune radius and period), by a rotating-frame initial
state (which can be corrected to a periodic orbit in place), or by
two-body elements about the Moon or the Earth (presets for a lunar
frozen relay orbit, a low lunar polar orbit, GEO, GTO and LEO).  Add
ground stations and optical sensors with their constraints.  Run, then
read access windows, per-constraint pass fractions, multi-station
coverage, time series, and the 3D scene in rotating or inertial frames
with optional stable and unstable manifolds.  A sweep panel runs the
scenario across a range of one setting and tabulates the results.

## Access constraints

A constraint is any function `constraint(step) -> bool` where `step`
is an `engine.geometry.StepGeometry` (elevation, Sun elevation, range,
lunar separation, phase angle, shadow flag, apparent magnitude and
line-of-sight rate at one instant).  `engine/constraints.py` provides
factories for elevation cutoff, station darkness, target illumination,
limiting magnitude, lunar exclusion, maximum range and maximum slew
rate; `engine/access.py` evaluates any list of them without knowing
what they test and combines masks across observers for coverage.  Add
a constraint by writing another factory and appending its result to
the list (`runner.run_scenario` accepts `extra_constraints`).

## The thesis ladder (THESIS_BRIEF.md)

Angles-only orbit determination and manoeuvre detection of the 9:2
NRHO from a Sydney telescope, with JPL DE440 for the sky
(`data/de440_ephemeris.npz`, rebuilt by `scripts/fetch_ephemeris.py`).

| Rung | Script | Output |
|---|---|---|
| 1 Simulated observations | `scripts/simulate_observations.py` | `output/observations.csv`, `fig7_observation_schedule.png` |
| 2 Orbit determination | `scripts/orbit_determination.py`, `scripts/observability_sweep.py` | `od_consistency.csv`, `fig9`; `observability.csv`, `fig8` |
| 3 Manoeuvre detection | `scripts/manoeuvre_detection.py` | `manoeuvre_detection.csv`, `fig10` |
| 4 Manoeuvre estimation | `scripts/manoeuvre_estimation.py` | `manoeuvre_estimation.csv`, `fig11` |

The estimation code is `engine/estimation.py` (measurement models with
analytic Jacobians, batch least squares, EKF, UKF, NEES and NIS),
`engine/observability.py` (Fisher information, Cramer-Rao bound) and
`engine/detection.py` (burn injection, per-night innovation test,
Monte Carlo minimum detectable burn, burn estimation).

## Relationship to GMAT

GMAT propagates with real ephemerides, force models and manoeuvre
modelling and is the right tool to verify a chosen trajectory.  It has
no notion of periodic orbit families, continuation, monodromy or
manifolds, which are what this code provides.  Design here, verify
there: `scripts/export_gmat.py` hands a scenario to GMAT with the
state in an Earth-Moon rotating barycentric coordinate system.
