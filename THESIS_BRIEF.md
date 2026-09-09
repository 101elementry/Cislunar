# Thesis brief: manoeuvre detection for cislunar objects from sparse ground-based optical tracks

## Who and what

Undergraduate thesis, third year Aeronautical Engineering (Space), University of
Sydney. Supervisor works in autonomy and estimation. The student knows classical
control, state space and pole placement, and is learning three-body dynamics
through this project. Every part of the thesis must be defensible line by line in
an oral examination, so clarity and verification beat sophistication.

## Working title

Orbit determination and manoeuvre detection for spacecraft in Earth-Moon L2
near-rectilinear halo orbits using sparse angles-only observations from a
southern-hemisphere ground telescope.

## Research question

Given realistic observation windows from a single optical telescope near Sydney,
how accurately can the orbit of an object on an L2 southern near-rectilinear halo
orbit (NRHO) be determined from angle measurements alone, and what is the smallest
impulsive manoeuvre that can be detected between observing nights?

## Why it is worth doing

Cislunar space domain awareness is a live problem: Gateway, CAPSTONE and follow-on
missions use NRHOs, and ground-based custody of objects there is immature.
Published work assumes northern-hemisphere sites. Southern halos have apolune over
the lunar south pole, and Australia hosts a Deep Space Network site at Canberra, so
the southern-hemisphere geometry is a real and unexamined case. A first result
from the existing tool already shows that an L2 NRHO never lies more than about
10 degrees from the Moon as seen from Earth, so lunar glare and the sensor's
exclusion angle, not the orbit geometry, set the observation schedule.

## Scope (fixed, stated in the introduction, defended)

- Dynamics: circular restricted three-body problem (CRTBP), Earth-Moon,
  mu = 0.01215058560962404, throughout. No ephemeris-model dynamics.
- Objects: one spacecraft at a time, on members of the existing L2 southern halo
  family (69 members, index 49 is the 9:2 Gateway-type NRHO).
- Sensor: one ground optical telescope near Sydney; angles only (right ascension
  and declination); Gaussian measurement noise fixed early and not revisited.
- Observations: simulated, inside the access windows computed by the existing
  constraint model (elevation, darkness, illumination, limiting magnitude, lunar
  exclusion).
- Frames: the current one-degree Sun and Moon model in engine/frames.py must be
  replaced by a real ephemeris before any observability result is reported.

## The ladder (each rung is a complete, defensible thesis on its own)

1. Simulated observations. Generate noisy angle measurements inside the access
   windows for one object, using a real ephemeris for the Sun and Moon and
   analytic measurement Jacobians. The azimuth/elevation model in
   engine/estimation.py is the starting point. Deliverable: a plot of observation
   times over a lunar month and a check of the Jacobians against finite
   differences.
2. Orbit determination. Batch least squares using the state transition matrix the
   engine already integrates, then the existing extended Kalman filter made
   honest: Monte Carlo consistency (NEES), process-noise tuning, and if needed a
   UKF. Deliverable: estimation error and covariance against truth; observability
   (Fisher information) along the orbit and across the family, plotted against
   perilune radius next to the existing stability-index plot.
3. Manoeuvre detection. Inject an impulsive burn between two nights; detect it from
   the filter innovations with a chi-squared test. Sweep burn size, burn location
   on the orbit (perilune vs apolune), and gap between nights. Deliverable: the
   smallest detectable burn as a function of those three variables.
4. Manoeuvre estimation (stretch). Add the burn as an unknown in a batch solve and
   recover its size and epoch.

Rungs 1 and 2 are the safe thesis. Rung 3 is where it becomes novel. Rung 4 is
only if time allows.

## Known hazards

- Perilune passage: dynamics are strongly nonlinear there and an EKF that is fine
  at apolune can diverge across a pass with no measurements. Expect to spend time
  on this; it is also where the science is, because a burn at perilune is cheap
  for the operator and hard to see.
- Sparse data: one window per night of a few hours, nothing near new Moon. The
  filter must survive multi-day gaps.
- The frames model. See scope.

## What already exists (repository, branch claude/cislunar-crtbp-halo-orbits-aelm0l)

Read HANDOFF.md for the full list. The parts that matter for this thesis:

- engine/crtbp.py: CRTBP equations of motion, analytic Jacobian, 42-element
  state + STM propagation (DOP853, 1e-12), Jacobi constant, unit conversions.
  Verified: Jacobi drift 3.6e-12 over 10 TU, STM vs finite difference 2.3e-8.
- engine/corrector.py, engine/families.py: halo, Lyapunov and DRO correctors and
  families. L2 southern halo member 49 matches the Gateway 9:2 NRHO (6.5608 d,
  C = 3.04652, perilune 3,245 km). L1 and northern families also exist.
- engine/geometry.py, constraints.py, access.py, photometry.py: ground-station
  geometry, shadow, diffuse-sphere magnitude, pluggable constraints, access
  windows and duty cycle. Sensors have limiting magnitude, lunar exclusion, max
  range and max slew rate.
- engine/estimation.py: azimuth/elevation and range/range-rate measurement
  models, central-difference measurement Jacobians, an EKF with STM covariance
  propagation in Joseph form. Angles-only from one station brings a 100 km
  initial error to about 8 km after 14 days, but the formal sigma (1.2 km) is
  optimistic because range along the line of sight is weakly observed. This is
  rung 2 in draft form; it needs a batch solve alongside it, honest consistency
  checks (Monte Carlo, NEES), and process-noise tuning or a UKF.
- engine/stationkeeping.py, engine/manifolds.py: impulsive STM targeting and
  invariant manifolds. Useful for choosing realistic burn sizes to inject in
  rung 3 (NRHO station keeping costs 3 to 5 m/s per year in this model).
- scripts/orbit_determination.py: the existing GUI-free OD example to start from.
- model/, app/: scenario data model, runner, sweep panel, Dash interface with
  GMAT export for cross-checking against an independent tool.
- CLAUDE.md: layering and style rules. HANDOFF.md: session state and gaps.

Still open and relevant: ephemeris-quality Sun and Moon positions in
engine/frames.py (about a degree today), and measurement Jacobians by central
difference rather than analytically. Both should be fixed inside rung 1.

## Deliverables expected of the thesis

- A verified simulation chain: dynamics, periodic orbit, observation windows,
  simulated measurements, estimator, detector.
- Figures: observability and estimation error along the orbit and across the
  family; minimum detectable burn against size, location and gap.
- A comparison of at least the periodic-orbit numbers against the JPL three-body
  periodic orbit catalogue.
- An honest chapter on what did not work and why.

## Rough timeline for a two-semester thesis

- Weeks 1-3: literature (AMOS proceedings, JGCD, Purdue theses on NRHO
  navigation), ephemeris frames, measurement model.
- Weeks 4-11: batch OD, then EKF; observability sweep across the family.
- Weeks 12-17: manoeuvre injection and detection sweep.
- Weeks 18-20: write-up; stretch rung only if rung 3 is finished.

## Rules for any Claude session working on this

Read CLAUDE.md and HANDOFF.md in the repository first. Keep the engine/model/app
layering. Plain Python, physics comments, docstrings with units, numpy/scipy only in
the engine. Every new numerical result needs a check the student can defend.
Explain the estimation theory as it is built; the student must understand it, not
just run it.
