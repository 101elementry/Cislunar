# Handoff (updated 10 Sep 2026, local session)

Branch: `claude/cislunar-crtbp-halo-orbits-aelm0l`. Everything described
here is committed. Read `CLAUDE.md` for the rules and `THESIS_BRIEF.md`
for the goal; the section below records where each rung stands.

## Thesis ladder: status and the numbers to defend

All four rungs run end to end.  Sky model: JPL DE440 through
`engine/ephemeris.py` (checked against jplephem to 3e-8 km); Earth
orientation IAU 2006 sidereal time and precession, nutation omitted
(under 20 arcsec, same model simulates and estimates).  Measurements:
topocentric right ascension and declination, 2 arcsec Gaussian, every
10 min inside the access windows the constraint model produces, member
49 from the Sydney station of the example scenario, epoch 2026-01-01.

**Rung 1, scripts/simulate_observations.py.**  One synodic month: 21
windows, 11.1 % duty cycle, 476 measurements.  The old mean-longitude
model gave 11.8 %; it differs from the ephemeris by up to 5 degrees
because it took the Moon's orbit plane as the ecliptic.  Analytic
RA/Dec Jacobian against central differences: worst relative error
3.4e-9 over all measurements; both rows perpendicular to the line of
sight.  Figure: output/fig7_observation_schedule.png.

**Rung 2, scripts/orbit_determination.py and observability_sweep.py.**
- Batch least squares (Gauss-Newton with backtracking; the arc grown in
  four stages so a 100 km, 1 m/s guess converges across two perilune
  passes): normalised residual rms 1.00, the noise-model check.
- Cold start of the filters from the 100 km guess: EKF NEES 870 and
  4.6 km, UKF NEES 71 and 1.7 km.  Warm start from a three-day batch
  solution: both filters NEES 5.67 (band [4.2, 8.1]) and NIS 1.99
  (band [1.0, 3.3]) for process noise 1e-9 to 1e-7 LU/TU^2, final error
  1.07 km against a formal 1.17 km, and the EKF equals the UKF.
  Conclusion to state: the perilune problem is an initialisation
  problem; after initial orbit determination the EKF is adequate.
  Process noise above 1e-6 makes the filters pessimistic.
- Cramer-Rao position bound after one month, 1,000 km / 10 m/s prior:
  0.9 to 3.1 km across the family (best near 32,000 km perilune, worst
  at both ends); along the NRHO 2.6 km with the epoch at perilune, 0.7
  to 1.1 km elsewhere.  Velocity bound rises from 0.01 to 2 m/s as the
  perilune tightens.  Figure: output/fig8_observability.png.

**Rung 3, scripts/manoeuvre_detection.py** (EKF, q = 1e-8, 8 trials
per case, per-night chi-squared test at 1e-3 false alarm, along-track
burns, 21 days).  Measured false alarm rate with no burn: 0.000.
Smallest burn detected with probability 0.9:

| burn at | resume after 1 night | 2 nights | 4 nights |
|---|---|---|---|
| perilune (day 6.56) | 0.87 m/s | 0.026 m/s | 0.087 m/s |
| apolune (day 3.28) | 0.87 m/s | 0.85 m/s | 0.83 m/s |

Read with care: the size grid is 0.01, 0.03, 0.1, 0.3, 1, 3 m/s and 8
trials, so each entry is a bracket, not a measurement to two figures.
The physics is right though: a perilune burn changes the period and
its effect grows through the following passage, so waiting a night
makes it far easier to see; an apolune burn moves the trajectory
slowly and needs about 1 m/s whatever the gap.  The perilune
non-monotonicity (2 nights better than 4) needs more trials and a
finer grid before it is quoted; that is the first thing to do next.
Figure: output/fig10_minimum_detectable_burn.png.

**Rung 4, scripts/manoeuvre_estimation.py.**  A 0.2 m/s along-track
burn at apolune: batch without a burn leaves rms 1.90 (the burn shows
in the residuals); with the burn as three unknowns at a known epoch,
estimate 0.205 m/s with 0.003 to 0.007 m/s one-sigma per axis, rms
0.93; profiled over 11 candidate epochs across the gap, the epoch is
recovered to 0.05 days.  Figure: output/fig11_burn_estimation.png.

**Next steps, in order.**  (1) Rung 3 with 30+ trials and a finer size
grid around each threshold; add radial and normal burn directions.
(2) Repeat rung 3 with the UKF as a check that the EKF's linearisation
is not what limits detection.  (3) A second station (Canberra DSN
site) to show what range or a second angle baseline buys.  (4) Write
up the initialisation finding of rung 2 as its own section.

## What exists and is verified

**Dynamics and corrector (engine/crtbp.py, engine/corrector.py)**
- CRTBP equations of motion from the pseudo-potential, analytic
  Jacobian, 42-element state + STM propagation, DOP853 at 1e-12.
- Checks (from `validate.py`): Jacobi drift over 10 TU 3.6e-12 (large
  halo) and 2.4e-11 (NRHO); STM column vs central difference 2.3e-8
  relative; monodromy eigenvalue product 1 to 3e-10; C(L1) = 3.188341,
  C(L2) = 3.172160 (now computed, matching the docstring values).
- Richardson third-order seed, xz-plane symmetric single-shooting
  corrector, natural-parameter continuation with jump rejection, walks
  69 members from a 50,000 km perilune halo to 1,780 km perilune.
- Member 49 matches the Gateway 9:2 NRHO: period 6.5608 d, C = 3.04652,
  perilune 3,245 km, apolune 71,213 km, stability index 1.32.
- New correctors: planar (Lyapunov, DRO: one condition, one unknown),
  general minimum-norm (any state plus period guess, no symmetry), and
  `correct_any` which picks one.  The general corrector converges onto
  a neighbouring member of the family rather than the exact input
  orbit, as expected with a free direction along the family.

**Families (engine/families.py, build_families.py, output/families/)**
- L1 southern halo (28 members to 1,725 km perilune; the Richardson L1
  seed only converges from the Earth-side crossing, so the seed is
  reflected about L1), L1/L2 northern halos (exact mirrors), L1 and L2
  Lyapunov (35 and 22 members, seeded from the linear in-plane mode,
  periods match 2 pi / lambda), DRO (18 members, 11,500 to 115,000 km,
  all linearly stable).  `model.family.load_families()` returns all.

**Manifolds, station keeping, estimation, rendezvous, elements**
- `engine/manifolds.py`: hyperbolic directions from the monodromy
  matrix, transported with the STM, integrated with Moon/Earth impact
  events.  Large-halo branches reach the Moon's surface and the Earth's
  vicinity in 30 days; NRHO branches barely leave in that time (small
  unstable eigenvalue), which is physics, not a bug.
- `engine/stationkeeping.py`: impulsive STM targeting to the reference
  position at the next node, nodes spaced to avoid perilune.  With 1 km
  / 1 cm/s navigation error and 1 % execution error: NRHO members 3 to
  5 m/s per year with one node per revolution; the largest halo needs
  four nodes per revolution (about 8 m/s per year) and diverges with
  one.
- `engine/estimation.py`: RA/Dec, azimuth/elevation and range/range-
  rate models with analytic Jacobians (central differences kept as the
  check), batch least squares with backtracking and a growing arc, EKF
  (Joseph form) and UKF, NEES and NIS with chi-squared bounds.  See the
  thesis ladder above for the numbers; the earlier "optimistic EKF"
  finding was a cold-start effect and is resolved by the batch warm
  start.  `engine/observability.py` (Fisher information, Cramer-Rao)
  and `engine/detection.py` (burn injection, per-night innovation
  test, minimum detectable burn, burn estimation) sit beside it.
- `engine/rendezvous.py`: LVLH relative motion with the frame rotation
  removed (Clohessy-Wiltshire sense), two-impulse rendezvous by STM
  targeting, transfer-time sweep.  GEO example: 50 km lower orbit
  drifts 472 km/day; 18 h transfer costs 19 m/s, 6 h costs 139 m/s.
- `engine/kepler.py`: two-body elements about the Moon or Earth to and
  from the rotating frame, with an optional reference-frame rotation
  (Earth equator via `frames.equatorial_to_rotating_matrix`).  ELFO
  preset stays frozen in the CRTBP over 60 days (argument of perilune
  85 to 90 degrees, perilune 2,270 to 2,470 km).
- `engine/frames.py`: rotating to inertial states and body positions
  for the display frames.

**Mission tool (model/, app/, scripts/)**
- Spacecraft sources: family member (any family, pick by index or
  nearest perilune/period), typed state (correct to periodic from the
  form; period stored), two-body elements with presets (ELFO, low lunar
  polar, GEO, GTO, LEO).  Manifold settings per spacecraft.
- Sensors gain max range and max slew rate; line-of-sight rate is in
  StepGeometry; runner reports multi-observer coverage.
- 3D scene: frames (rotating barycentric/Moon-centred, inertial
  Moon/Earth/barycentre), manifolds, moving bodies in inertial views,
  turntable drag, Focus menu that recentres the camera, and scroll
  zoom toward the point under the cursor (assets/zoom_to_cursor.js;
  Plotly's camera box spans plus or minus half the aspect ratio, which
  was measured, not guessed).
- Sweep panel (model/sweep.py) and CSV download.
- Dark instrument-console theme; colours carry meaning (see the
  header comment in app/assets/style.css).
- Scripts: sweep_min_elevation, compare_family_members,
  manifold_transfers, station_keeping_sweep, orbit_determination,
  geo_rendezvous, elfo_drift, export_gmat.

## Known gaps and limits

1. Ephemeris-quality frames are done (DE440, see above); the fixed
   length unit against the real Earth-Moon distance (356,800 to
   406,700 km) is the remaining inconsistency of the CRTBP frame and
   must be stated.  UT1 = UTC and no nutation are the other stated
   approximations.
2. Lunar oblateness and the 6.7 degree tilt of the Moon's equator are
   not modelled, so low lunar orbits look better behaved than reality.
3. The EKF is honest after a batch warm start (rung 2); cold-started
   from 100 km it is not, and that is documented, not hidden.
4. Manifold and station-keeping results depend on the displacement,
   node count and error settings; the defaults are documented in the
   docstrings and are tuned, not derived.
5. The interface's accessibility tree cannot be read by some browser
   automation because pattern-matching ids are JSON strings; this does
   not affect use.

## Numerical fragility to remember

- Richardson coefficients are a long hand transcription, only verified
  by convergence from the seed; the L1 seed needs reflection.
- Continuation jump-rejection thresholds (0.02 LU, 0.2 LU/TU for halos;
  0.05 LU/TU and 25 % period for planar families) are tuned.
- Stability index drops the two eigenvalues nearest 1 as the trivial
  pair; fragile if the unit pair drifts.
- Crossing event relies on direction = -sign(vy0) to skip t = 0.
- Perilune passes dominate integration error; below ~1,800 km perilune
  the family is unphysical anyway.
- Access constraints are evaluated per step in a Python loop (about
  100k calls for the example, well under a second).

## Results to check against the JPL three-body periodic orbit catalogue

- Family member 49: period, Jacobi constant, perilune and apolune radii
  (values above). JPL uses the same mu but LU = 389,703 km and
  TU = 382,981 s, so compare non-dimensional numbers, not km or days.
- Family member 0: period 3.404403 TU, C = 3.146266, x0 = 1.115378,
  z0 = 0.026023, vy0 = 0.190478.
- Perilune radii where the stability index crosses 1.
- New: L1 halo family, Lyapunov and DRO members (build_families.py
  prints the tables) against the same catalogue.

## The code walkthrough for the exam

Section 2 (Jacobi constant) was delivered in the chat on 9 Sep 2026.
Sections owed, in this order, stopping after each for questions:

3. State transition matrix: what it represents, why propagated with the
   state rather than computed afterwards, how the analytic Jacobian was
   derived. Go slowest here.
4. Differential corrector: why periodic orbits must be solved for, how
   xz-plane symmetry reduces the problem to two conditions, how Newton
   uses the STM; show the actual update step in `engine/corrector.py`.
   Now also the planar and general correctors.
5. Continuation: why stepping a parameter and re-converging walks the
   family; what makes the NRHO region different from larger halos.
6. Monodromy matrix and stability index: eigenvalue meaning, why
   reciprocal pairs, link to station-keeping frequency (now with real
   numbers from scripts/station_keeping_sweep.py) and to manifolds.

Format: explain why before what, LaTeX maths with one equation per
display block on a single line, reference real functions and line
numbers rather than fresh example code, no bare algebra and no
derivation essays. Finish with: least-confident parts, numerical
fragility and what failure looks like, what to change first in a
proper rewrite, and which results to check against the JPL catalogue.
