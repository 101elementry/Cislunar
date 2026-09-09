# Cislunar project rules

Undergraduate thesis code for cislunar spacecraft dynamics (USYD
Aeronautical Engineering, Space). The owner must be able to defend every
line in an examination, so clarity beats cleverness everywhere.

## Layering (hard boundary, do not cross)

- `engine/`  pure analysis. numpy arrays and plain values in and out.
  No file IO, no plotting, no Dash, no imports from `model/` or `app/`.
  Modules: crtbp, corrector, families, manifolds, stationkeeping,
  estimation, rendezvous, kepler, frames, propagation, geometry,
  photometry, constraints, access.
- `model/`   scenario dataclasses, JSON serialisation, family file
  loading, `orbits` (a Spacecraft to a state, correction to periodic),
  `runner.run_scenario` which maps a Scenario onto engine calls, and
  `sweep` for batch runs. Knows nothing about display. May import
  `engine`.
- `app/`     Dash interface and Plotly figure builders. Callbacks only
  read the model and call the runner, `model.orbits`, `model.sweep` or
  engine frame conversions; no physics or geometry inside a callback.
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
- Dependencies: numpy, scipy, matplotlib (root scripts), plotly, dash.
  Nothing else without asking.
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
  or "earth equator").
- Inertial frames for display are aligned with the rotating axes at
  t = 0; `engine.frames.rotating_to_inertial_states`.

## Commands

```
pip install numpy scipy matplotlib plotly dash
python validate.py                          # checks + family file (~20 s)
python build_families.py                    # other families (~10 s)
python plots.py                             # thesis figures
python -m app.main                          # GUI at http://127.0.0.1:8050
python scripts/<example>.py                 # see README for the list
```

Read `HANDOFF.md` for the state of the work and the agreed next steps.
