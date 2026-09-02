"""
engine: pure analysis.

Rules for everything in this package
  * functions take numpy arrays and plain Python values and return the
    same; no model objects, no Dash, no Plotly, no matplotlib
  * no file input or output
  * no imports from model/ or app/

Modules
  crtbp        Earth-Moon circular restricted three-body dynamics
  corrector    halo orbit differential correction and continuation
  frames       time, Sun direction, Earth rotation, station positions
  propagation  spacecraft trajectories on a time grid
  photometry   reflected-light apparent magnitude
  geometry     observer-to-target geometry on a time grid
  constraints  pluggable access constraints, one function each
  access       constraint evaluation, windows and duty cycle
"""
