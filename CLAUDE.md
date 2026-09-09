# Cislunar project rules

Undergraduate thesis code for cislunar spacecraft dynamics (USYD
Aeronautical Engineering, Space). The owner must be able to defend every
line in an examination, so clarity beats cleverness everywhere.

## Layering (hard boundary, do not cross)

- `engine/`  pure analysis. numpy arrays and plain values in and out.
  No file IO, no plotting, no Dash, no imports from `model/` or `app/`.
- `model/`   scenario dataclasses, JSON serialisation, family file
  loading, and `runner.run_scenario` which maps a Scenario onto engine
  calls. Knows nothing about display. May import `engine`.
- `app/`     Dash interface and Plotly figure builders. Callbacks only
  read the model and call the runner or engine; no physics or geometry
  inside a callback.
- `scripts/` GUI-free worked examples; import only `model` and `engine`.
- `validate.py`, `plots.py` at the root are the thesis checks and
  matplotlib figures for the halo family.

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

## Units and conventions

- Non-dimensional Earth-Moon rotating frame, mu = 0.01215058560962404,
  Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0). State order
  [x, y, z, vx, vy, vz]. LU = 384400 km, TU = 375190.26 s.
- All unit conversion lives in one block in `engine/crtbp.py`.
- Halo family: `output/halo_family.npz`, 69 L2 southern members, index 0
  largest halo, index 49 is the 9:2 Gateway-type NRHO, index 68 lowest
  perilune. Rebuild with `python validate.py`.
- Orbits are dictionaries: state0, period, jacobi, perilune_radius,
  apolune_radius, stability_index, eigenvalues.

## Commands

```
pip install numpy scipy matplotlib plotly dash
python validate.py                          # checks + family file (~20 s)
python plots.py                             # thesis figures
python -m app.main                          # GUI at http://127.0.0.1:8050
python scripts/sweep_min_elevation.py       # GUI-free CSV example
python scripts/compare_family_members.py    # sweep along the family
```

Read `HANDOFF.md` for the state of the work and the agreed next steps.
