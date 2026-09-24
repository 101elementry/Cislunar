# Cislunar

Numerical foundation for cislunar spacecraft dynamics: the Earth-Moon
circular restricted three-body problem, periodic orbit families found
by differential correction and continuation, their stability and
manifolds, and a mission-analysis tool built on top of them.

## Model assumptions

Every trajectory this repository produces, without exception, comes from
the circular restricted three-body problem:

- Earth and Moon as **point masses** on a common circular orbit, mass
  ratio mu = 0.01215058560962404, in a frame rotating with them at a
  constant rate.  The spacecraft has no mass.
- DOP853 at a relative and absolute tolerance of 1e-12, with the Jacobi
  constant checked afterwards (drift about 1e-11 over 10 TU).

Not modelled, in any result below (the one exception is the fidelity
comparison at the end of this section):

- No lunar gravity field beyond the point mass: no J2, no spherical
  harmonics, no mascons.
- No solar gravity, no solar radiation pressure, no Earth oblateness,
  no lunar librations.
- No eccentricity (the real value is 0.055) or inclination in the
  Moon's orbit.
- No ephemeris dynamics.  JPL DE440 is used only to place the Sun,
  Moon, Earth and the observing site for observation geometry, never in
  the equations of motion.

Three consequences worth stating before any result is quoted.  A CRTBP
periodic orbit repeats exactly and a real NRHO does not, which is why a
real one needs a station keeping burn every revolution.  The synodic
resonance that names these orbits (`engine.families.synodic_resonance`)
is a resonance with the Sun, so here it is a label computed from the
period rather than anything the dynamics enforce.  And in the estimation
work the simulated truth and the filter share these equations, so the
filter meets no dynamic mismodelling: the errors and minimum detectable
manoeuvres are a floor, and repeating the ladder on an ephemeris model
is what would test it.

The first step of that test exists.  `engine/ephemeris_dynamics.py` flies
a spacecraft under the Earth, Moon and Sun as point masses at their
DE440 positions (Moon-centred ICRF, km and seconds), and carries a CRTBP
state into that model with the day's Earth-Moon distance and Kepler's
time unit for it.  `python scripts/ephemeris_divergence.py` flies member
49 that way, uncorrected, from 2026-01-01: it leaves the CRTBP orbit by
100 km within a day and by 1,000 km within about six days, its perilune
falls lap by lap (3,048, 2,869, 2,268 km) and it strikes the Moon on
day 18.  From eight epochs across the month it passes 1,000 km in two
to eight days.  Leaving out the Sun changes these very little, so the
circular Earth-Moon orbit is the assumption that costs most.  A usable
NRHO in this model needs the CRTBP orbit corrected into it (multiple
shooting), which is not built yet.

Orbits are named the way missions name them, by the libration point and
branch plus the synodic resonance: the Gateway orbit is the 9:2 southern
L2 NRHO, member 49 of `output/halo_family.npz`.  `python validate.py`
prints the resonance of every member and which member stands closest to
each of 11:3, 4:1, 9:2 and 19:4.

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
python -m app.showcase                      # static pages in site/ (~5 s)
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
python scripts/mars_transfers.py            # Mars launch windows, porkchop plot, NRHO staging
python scripts/artemis_profile.py           # lander and crew legs to the NRHO (about two minutes)
python scripts/nrho_departure.py            # NRHO to a low Earth perigee for a Mars departure
python scripts/mars_short_stay.py           # 30-day Mars stay with and without a Venus flyby
python scripts/ephemeris_divergence.py      # the 9:2 NRHO flown uncorrected with the real Earth, Moon and Sun
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

## Static showcase

`python -m app.showcase` runs five fixed scenarios and writes them to
`site/` as plain web pages: the halo family tipping into the NRHO, the
manifolds of a halo orbit, a DRO in rotating and inertial frames, a
month of access windows from Sydney, and a lunar south pole relay
orbit.  The pages hold the same Plotly figures as the interface
(`app/scene.py`) and reuse its playback and zoom scripts, so each scene
rotates, zooms toward the cursor and plays like a video, but nothing
can be edited because no Python runs behind them.  Open
`site/index.html`, or host the folder on any static host (on Vercel set
the project's root directory to `site` with no build command).  The
folder is committed; rebuild it after changing a scene or a figure.

## Proximity operations

A spacecraft can be placed as an offset from another one in that
spacecraft's LVLH frame (source "relative": radial, along-track and
cross-track position in km and velocity in m/s; see
`engine.rendezvous.state_from_lvlh_offset`), which is how a chaser is
put near a target.  An optical sensor can be carried by a spacecraft
instead of a ground station; it then watches every other spacecraft
through `engine.geometry.space_observation_geometry`, with Sun and
Earth exclusion angles in place of a horizon and a dark sky.  The
scene's frame menu gains "Relative to X (LVLH)" for each spacecraft,
the view rendezvous is flown in, with a keep-out sphere around the
target.  The Examples menu loads a chaser 50 km behind a Gateway-like
target with a camera.

## Crewed mission legs

`engine/transfers.py` finds two-burn transfers from a circular parking
orbit about the Moon or the Earth to a moving target: a Lambert seed
about the parking body, with the departure point chosen so the burn is
tangential, corrected in the full three-body equations.  A spacecraft
can carry a list of impulsive `burns` (day, rotating-frame delta-v in
m/s), flown by `engine.propagation.propagate_with_burns`.
`python scripts/artemis_profile.py` scans the lander leg (100 km polar
lunar orbit to the NRHO, about 700 m/s in half a day, arriving just
after perilune) and the crew leg (200 km Earth orbit to the NRHO,
direct), and writes `scenarios/lander_to_nrho.json` and
`scenarios/crew_to_nrho.json`, which the Examples menu loads: the
station, the vehicle with its arrival burn, and the Sydney telescope
watching both.  The lander arrives at a hold point 30 km behind the
station and closes through 12 km to 500 m
(`engine.rendezvous.approach_sequence`, about 14 m/s).  The crew vehicle
flies a powered lunar flyby (`transfers.flyby_from_earth`): injection
3,134 m/s, 295 m/s at 150 km above the Moon, 171 m/s insertion, against
921 m/s insertion for the direct transfer.  The cheapest flyby cases
need a parking orbit steeply inclined to the Earth-Moon plane; the scan
prints the inclination of each.

## Earth-Mars transfers

A separate, Sun-centred model for interplanetary legs, which the
Earth-Moon three-body model cannot reach.  `engine/lambert.py` solves
Lambert's problem (universal variables, checked against Curtis example
5.2); `engine/interplanetary.py` does patched conics with planet
positions from DE440 (`Ephemeris.heliocentric_state`; Mars is in the
extract): launch energy, arrival speed, porkchop grids, and the
departure burn from a low circular orbit against a burn at a low
perigee reached from the Moon's distance, which is what staging in the
NRHO buys.  `python scripts/mars_transfers.py` finds the windows from
2030 to 2042, draws `fig12_mars_porkchop.png` and compares a long stay
with a 30-sol stay.  `python scripts/nrho_departure.py` prices the leg
from the NRHO down to that perigee in the three-body model: no single
burn of up to 200 m/s along the direction of travel gets below about
100,000 km, but the crew arrival flown backwards in the mirror (the
equations are symmetric under y -> -y, t -> -t) is an exact departure,
about 0.48 km/s, so leaving the NRHO for Mars in 2035 costs about
1.0 km/s against 3.6 from low Earth orbit.
`python scripts/mars_short_stay.py` adds Venus (in the extract) and
`interplanetary.gravity_assist`: a 30-day stay with a Venus flyby on
the way out comes to about 640 days and 5.4 km/s with a 12.7 km/s
entry.

## Access constraints

A constraint is any function `constraint(step) -> bool` where `step`
is an `engine.geometry.StepGeometry` (elevation, Sun elevation, range,
lunar separation, phase angle, shadow flag, apparent magnitude and
line-of-sight rate at one instant).  `engine/constraints.py` provides
factories for elevation cutoff, station darkness, target illumination,
limiting magnitude, lunar exclusion, solar and Earth exclusion (for a
camera in space), maximum range and maximum slew rate; `engine/access.py` evaluates any list of them without knowing
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
