"""
engine: pure analysis.

Rules for everything in this package
  * functions take numpy arrays and plain Python values and return the
    same; no model objects, no Dash, no Plotly, no matplotlib
  * no file input or output
  * no imports from model/ or app/

Modules
  crtbp          Earth-Moon circular restricted three-body dynamics
  corrector      periodic orbit correctors (halo, planar, general) and
                 halo continuation
  families       seeds and continuation for L1 halos, Lyapunov and DRO
                 families, mirroring for northern families
  manifolds      stable and unstable manifold branches of a periodic orbit
  stationkeeping impulsive targeting station keeping and its delta-v
  ephemeris      JPL DE440 Chebyshev evaluation, rotating-frame axes
  estimation     RA/Dec and az/el models with analytic Jacobians, batch
                 least squares, EKF, UKF, NEES and NIS
  observability  Fisher information and the Cramer-Rao bound
  detection      burn injection, per-night innovation test, Monte Carlo
                 minimum detectable burn, burn estimation
  rendezvous     LVLH relative motion and two-impulse rendezvous
  kepler         two-body elements about the Moon or Earth <-> state
  frames         time, Sun direction, Earth rotation, station positions,
                 inertial frames for display
  propagation    spacecraft trajectories on a time grid
  photometry     reflected-light apparent magnitude
  geometry       observer-to-target geometry on a time grid
  constraints    pluggable access constraints, one function each
  access         constraint evaluation, windows, duty cycle, coverage
"""
