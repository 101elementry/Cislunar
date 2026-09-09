# Handoff from the Claude Code web session (9 Sep 2026)

Branch: `claude/cislunar-crtbp-halo-orbits-aelm0l`. Everything described
here is committed and pushed. Read `CLAUDE.md` first for the rules.

## What exists and is verified

**Dynamics and corrector (engine/crtbp.py, engine/corrector.py)**
- CRTBP equations of motion from the pseudo-potential, analytic
  Jacobian, 42-element state + STM propagation, DOP853 at 1e-12.
- Checks (from `validate.py`): Jacobi drift over 10 TU 3.6e-12 (large
  halo) and 2.4e-11 (NRHO); STM column vs central difference 2.3e-8
  relative; monodromy eigenvalue product 1 to 3e-10.
- Richardson third-order seed, xz-plane symmetric single-shooting
  corrector, natural-parameter continuation with jump rejection, walks
  69 members from a 50,000 km perilune halo to 1,780 km perilune.
- Member 49 matches the Gateway 9:2 NRHO: period 6.5608 d, C = 3.04652,
  perilune 3,245 km, apolune 71,213 km, stability index 1.32. The
  stability index crosses 1 near perilune 17,100 km, 13,500 km and
  1,830 km, matching the published NRHO bounds.

**Mission tool (engine/, model/, app/, scripts/)**
- Scenario = epoch + duration + step + spacecraft + ground stations +
  optical sensors, saved as JSON.
- Frames: mean lunar longitude defines the rotating frame, low-precision
  solar longitude, GMST, spherical tilted Earth. Good to ~1 degree.
- Geometry per pair: elevation, Sun elevation, range, lunar separation,
  phase angle, cylindrical Earth/Moon shadow, diffuse-sphere magnitude.
- Constraints: elevation cutoff, station darkness, target illumination,
  limiting magnitude, lunar exclusion, all `f(StepGeometry) -> bool`.
- Access windows resolved to the grid step, duty cycle, per-constraint
  pass fractions.
- Dash GUI: tree + property forms, 3D rotating-frame view with equal
  aspect, windows table, time series, time slider, panel toggles, bulk
  add of family members, built-in help fold-outs.
- Example result: 9:2 NRHO from Sydney, 14 days from 2026-01-01, 13
  nightly windows, 18.5% duty cycle. From Earth an L2 NRHO never exceeds
  ~10 degrees from the Moon, so lunar exclusion above that gives nothing.

## Known gaps the owner has already hit

1. **No way to correct a typed initial state into a periodic orbit from
   the GUI.** "Initial state" spacecraft are always integrated. Wanted: a
   "correct to periodic" action calling `corrector.correct_halo` and
   storing the period so it can be propagated as periodic.
2. **No pick-by-property for family members.** Wanted: choose a member
   by perilune radius or period instead of scanning the dropdown.
3. Only the L2 southern family exists. L1, northern, and other families
   need the seed sign checked (see `richardson_halo_guess`: the
   southern/northern label was fixed empirically).

## Numerical fragility to remember

- Richardson coefficients are a long hand transcription, only verified
  by convergence from the seed.
- Continuation jump-rejection thresholds (0.02 LU, 0.2 LU/TU) are tuned,
  not derived.
- Stability index drops the two eigenvalues nearest 1 as the trivial
  pair; fragile if the unit pair drifts.
- Crossing event relies on direction = -sign(vy0) to skip t = 0.
- Perilune passes dominate integration error; below ~1,800 km perilune
  the family is unphysical anyway.
- Access constraints are evaluated per step in a Python loop (about
  100k calls for the example, well under a second); vectorise inside
  `engine/access.evaluate_constraints` only if scenarios grow to
  millions of samples.

## Results to check against the JPL three-body periodic orbit catalogue

- Family member 49: period, Jacobi constant, perilune and apolune radii
  (values above). JPL uses the same mu but LU = 389,703 km and
  TU = 382,981 s, so compare non-dimensional numbers, not km or days.
- Family member 0: period 3.404403 TU, C = 3.146266, x0 = 1.115378,
  z0 = 0.026023, vy0 = 0.190478.
- Perilune radii where the stability index crosses 1.

## The owner's stated direction: "a full system suite"

Not yet specified in detail. The pieces that fit the existing layering,
roughly in order of value for a thesis supervised in autonomy and
estimation:

1. Close the two GUI gaps above (corrector from a typed state, pick by
   property).
2. Other orbit families: L1 halos, northern families, Lyapunov, DROs.
   The corrector and continuation already generalise; each family needs
   a seed and a walk.
3. Ephemeris-quality frames: replace `engine/frames.py` with real Sun
   and Moon positions (SPICE-like or a JPL ephemeris reader) while
   keeping the same function signatures.
4. Manifolds and transfers: stable/unstable manifolds from the
   monodromy eigenvectors, which the engine already computes.
5. Station keeping: perturb a periodic orbit, apply a simple
   targeting manoeuvre each revolution, record delta-v. Pairs with the
   stability index story.
6. Estimation hooks: measurement models for the ground-based sensors
   (angles, range if radar), an EKF or batch filter on the CRTBP
   dynamics using the existing STM. This is the supervisor's field.
7. More constraint types via the existing interface: sensor field of
   view, slew rate, weather/cloud fraction, multi-station coverage.
8. Batch/experiment runner: sweep any scenario parameter, write CSV,
   mirrored in the GUI as a results browser.

Ask the owner which of these the "suite" means before building.
