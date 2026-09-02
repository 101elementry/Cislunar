"""
Spacecraft trajectories on a time grid.  Thin wrappers around the
integrator in crtbp.py that return states sampled at requested times.
All positions are non-dimensional rotating-frame quantities (LU, LU/TU).
"""

import numpy as np

from engine import crtbp


def propagate_state(initial_state, times_nondim, mu=crtbp.MU):
    """
    Integrate an initial state with the full equations of motion and
    sample it at the given times (TU past zero, increasing, starting at
    zero).  Returns an array of shape (n, 6).
    """
    initial_state = np.asarray(initial_state, dtype=float)
    times_nondim = np.asarray(times_nondim, dtype=float)
    if times_nondim[-1] == 0.0:
        return np.tile(initial_state, (len(times_nondim), 1))
    sol = crtbp.propagate(initial_state, times_nondim[-1], mu, t_eval=times_nondim)
    return sol.y.T


def propagate_periodic(initial_state, period, times_nondim, mu=crtbp.MU):
    """
    Sample a periodic orbit at the given times by integrating one period
    once and evaluating at time modulo the period.  The spacecraft stays
    on the orbit for the whole span, which is what a perfectly
    station-kept vehicle does; contrast propagate_state, where an
    unstable orbit is eventually left.  Returns shape (n, 6).
    """
    initial_state = np.asarray(initial_state, dtype=float)
    times_nondim = np.asarray(times_nondim, dtype=float)
    one_period = crtbp.propagate(initial_state, float(period), mu, dense_output=True)
    return one_period.sol(np.mod(times_nondim, float(period))).T


def fixed_points(mu=crtbp.MU):
    """Positions that do not move in the rotating frame: Earth, Moon, L1, L2."""
    libration = crtbp.collinear_libration_points(mu)
    return {"earth": crtbp.earth_position(mu),
            "moon": crtbp.moon_position(mu),
            "L1": np.array([libration["L1"], 0.0, 0.0]),
            "L2": np.array([libration["L2"], 0.0, 0.0])}
