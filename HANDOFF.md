# Handoff (updated 9 Sep 2026, local session)

Branch: `claude/cislunar-crtbp-halo-orbits-aelm0l`. Everything described
here is committed. Read `CLAUDE.md` first for the rules.

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
- `engine/estimation.py`: azimuth/elevation and range/range-rate
  models, central-difference measurement Jacobians, EKF with STM
  covariance propagation (Joseph form).  Angles-only from one station:
  100 km initial error to about 8 km after 14 days, but the formal
  sigma (1.2 km) is optimistic because range along the line of sight is
  weakly observed; adding range brings the error to 0.5 km (formal 0.1
  km, still optimistic by a few).  A UKF or better process-noise tuning
  is the natural next step; say this in the thesis.
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

1. Ephemeris-quality frames are still not done: the Sun and Moon use
   the mean-longitude model (about a degree).  Replacing
   `engine/frames.py` with a real ephemeris while keeping the function
   signatures is the remaining roadmap item; the GMAT export exists so
   the difference can be measured instead.
2. Lunar oblateness and the 6.7 degree tilt of the Moon's equator are
   not modelled, so low lunar orbits look better behaved than reality.
3. The EKF is optimistic (see above).
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
