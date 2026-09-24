# Cislunar project rules

Undergraduate thesis code for cislunar spacecraft dynamics (USYD
Aeronautical Engineering, Space). The owner must be able to defend every
line in an examination, so clarity beats cleverness everywhere.

## Layering (hard boundary, do not cross)

- `engine/`  pure analysis. numpy arrays and plain values in and out.
  No file IO, no plotting, no Dash, no imports from `model/` or `app/`.
  Modules: crtbp, corrector, families, manifolds, stationkeeping,
  lambert, interplanetary (Sun-centred, km and seconds, not LU/TU),
  transfers (parking orbit to target, Lambert seed then correction),
  ephemeris, ephemeris_dynamics (Earth, Moon and Sun point masses at
  their DE440 positions, Moon-centred ICRF, km and seconds; only the
  fidelity comparison uses it), estimation, observability, detection,
  rendezvous, kepler,
  frames, propagation, geometry, photometry, constraints, access.
- `model/`   scenario dataclasses, JSON serialisation, family file
  loading, `orbits` (a Spacecraft to a state, correction to periodic),
  `runner.run_scenario` which maps a Scenario onto engine calls, and
  `sweep` for batch runs. Knows nothing about display. May import
  `engine`.
- `app/`     Dash interface and Plotly figure builders. Callbacks only
  read the model and call the runner, `model.orbits`, `model.sweep` or
  engine frame conversions; no physics or geometry inside a callback.
  `app/scene.py` builds the 3D scene for both the interface and
  `app/showcase.py`, which writes the static pages in `site/`
  (stylesheet and script in `app/showcase_assets/`, not `app/assets/`,
  because Dash loads everything in `assets/` into the interface).
- `scripts/` GUI-free worked examples; import only `model` and `engine`.
- `validate.py`, `plots.py`, `build_families.py` at the root are the
  thesis checks, the matplotlib figures, and the extra family builder.

Audit after any change:
`grep -rnE "^\s*(import|from) (model|app|plotly|dash|matplotlib)|open\(|np\.save|np\.load" engine/`
must print nothing.

## Style (owner's explicit requirements)

- Plain readable Python, no clever tricks.
- Write expressions out in full rather than abbreviated intermediates.
- Comment the physics, not the syntax.
- Docstrings state what a function does and its units.
- Dependencies (requirements.txt): numpy, scipy, matplotlib, plotly,
  dash; jplephem only for scripts/fetch_ephemeris.py.  The engine
  stays numpy and scipy.  The owner has said further dependencies are
  fine when they earn their place; keep them out of the engine.
- Constraints are pluggable: any `f(StepGeometry) -> bool` with `.name`
  and `.kind` attributes; `engine/access.py` must stay agnostic.
- Interface: a dark instrument console (see `app/assets/style.css`
  header comment for the colour meanings). Keep new controls in that
  language; the 3D scene is the signature element.

## Units and conventions

- Non-dimensional Earth-Moon rotating frame, mu = 0.01215058560962404,
  Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0). State order
  [x, y, z, vx, vy, vz]. LU = 384400 km, TU = 375190.26 s.
- All unit conversion lives in one block in `engine/crtbp.py`.
- Families: `output/halo_family.npz` is the L2 southern halo family
  (69 members, index 0 largest halo, index 49 the 9:2 Gateway-type
  NRHO, index 68 lowest perilune; rebuild with `python validate.py`).
  `output/families/*.npz` hold L1 southern/northern halos, L2 northern
  halos (mirror), L1/L2 Lyapunov, DRO; rebuild with
  `python build_families.py`. `model.family.load_families()` gives
  them all as {name: list}.
- Orbits are dictionaries: state0, period, jacobi, perilune_radius,
  apolune_radius, stability_index, eigenvalues.
- Spacecraft sources: "family" (family_name, family_index), "state"
  (initial_state, period_tu > 0 once corrected), "elements" (two-body
  elements about centre "moon" or "earth", reference plane "moon orbit"
  or "earth equator"), "relative" (an LVLH offset from the spacecraft
  named in relative_to, about `centre`; always integrated).
- A Spacecraft may carry `burns` ({time_days, delta_v_m_s} in the
  rotating frame); it is then integrated leg by leg.  `scenarios/`
  holds the mission examples written by scripts/artemis_profile.py.
- Observers: a sensor's `station` field names its host, a ground
  station or a spacecraft.  `runner.observers` yields (name, host,
  sensor); check `host.kind`.  A spacecraft host uses
  `space_observation_geometry` and the Sun and Earth exclusion
  constraints; elevation and Sun elevation are NaN for it.
- Display frames: the fixed ones in `app.scene.FRAME_LABELS`, plus
  "lvlh:<spacecraft>" for the relative-motion view in km.
- Inertial frames for display are aligned with the rotating axes at
  t = 0; `engine.frames.rotating_to_inertial_states`.
- Sky model: with `data/de440_ephemeris.npz` present (committed;
  rebuild with `python scripts/fetch_ephemeris.py`) every sky function
  in `engine/frames.py` uses JPL DE440 through the optional `ephemeris`
  argument; without it the mean-longitude model.  Thesis results use
  the ephemeris.  Times: UTC in, TDB = UTC + 69.184 s inside.
- Thesis (THESIS_BRIEF.md): angles-only OD and manoeuvre detection of
  the NRHO from Sydney.  Filters are warm-started from a batch solve
  of the first nights; the honest process noise is <= 1e-7 LU/TU^2.

## Commands

```
pip install numpy scipy matplotlib plotly dash
python validate.py                          # checks + family file (~20 s)
python build_families.py                    # other families (~10 s)
python plots.py                             # thesis figures
python -m app.main                          # GUI at http://127.0.0.1:8050
python -m app.showcase                      # rebuild the static showcase in site/
python scripts/<example>.py                 # see README for the list
python scripts/fetch_ephemeris.py           # rebuild data/de440_ephemeris.npz (downloads 32 MB once)
python scripts/simulate_observations.py     # rung 1
python scripts/orbit_determination.py       # rung 2 (filters, consistency)
python scripts/observability_sweep.py       # rung 2 (Cramer-Rao bound)
python scripts/manoeuvre_detection.py       # rung 3 (about 15 minutes)
python scripts/manoeuvre_estimation.py      # rung 4
```

Read `HANDOFF.md` for the state of the work and the agreed next steps.
