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


def propagate_with_burns(initial_state, times_nondim, burn_times_nondim, burn_delta_vs, mu=crtbp.MU):
    """
    Integrate a state through a list of impulsive burns and sample it on
    the grid.  An impulsive burn changes the velocity in an instant and
    leaves the position alone, so the trajectory is a chain of coasts:
    integrate to the burn, add its delta-v, carry on.

    burn_times_nondim : (k,) burn times in TU, increasing, inside the grid
    burn_delta_vs     : (k, 3) velocity changes in LU/TU, rotating frame
    Returns (n, 6).  A grid sample that falls exactly on a burn holds
    the state just before it.
    """
    times_nondim = np.asarray(times_nondim, dtype=float)
    states = np.zeros((len(times_nondim), 6))
    state = np.asarray(initial_state, dtype=float).copy()
    leg_start = 0.0
    edges = list(burn_times_nondim) + [times_nondim[-1] + 1.0]
    burns = list(burn_delta_vs) + [np.zeros(3)]
    for leg_end, delta_v in zip(edges, burns):
        if leg_start == 0.0:
            on_leg = times_nondim <= leg_end
        else:
            on_leg = (times_nondim > leg_start) & (times_nondim <= leg_end)
        stop = min(leg_end, times_nondim[-1])
        if stop > leg_start:
            sample_times = np.concatenate([times_nondim[on_leg] - leg_start, [stop - leg_start]])
            solution = crtbp.propagate(state, stop - leg_start, mu, t_eval=np.unique(sample_times))
            lookup = {round(float(t), 12): solution.y[:, k] for k, t in enumerate(solution.t)}
            for index in np.where(on_leg)[0]:
                states[index] = lookup[round(float(times_nondim[index] - leg_start), 12)]
            state = solution.y[:, -1].copy()
        elif np.any(on_leg):
            states[on_leg] = state
        state[3:] = state[3:] + np.asarray(delta_v, dtype=float)
        leg_start = leg_end
        if leg_start >= times_nondim[-1]:
            break
    return states


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
