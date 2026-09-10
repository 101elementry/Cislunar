"""
Observability of a periodic orbit from a set of angle measurements:
the Fisher information and the Cramer-Rao bound it implies.

For measurements y_k = h(x_k) + noise with covariance R_k, and x_k the
state at t_k related to the epoch state x_0 by the STM, the Fisher
information about x_0 is
    J = sum_k Phi_k^T H_k^T R_k^-1 H_k Phi_k
with H_k the measurement Jacobian at the truth.  Its inverse is the
smallest covariance any unbiased estimator can reach (the Cramer-Rao
lower bound), so the square root of the trace of the position block
is the best possible position uncertainty from that observation set,
before any filter is run.  This is what "how observable is the orbit
from here" means quantitatively, and it depends on the geometry
alone: where the station is, when it can see, and how the STM couples
the seen part of the state to the rest.

The information is singular when angles from one station over a short
arc leave the range direction undetermined; a weak prior covariance
regularises it and is stated with the result.
"""

import numpy as np

from engine import crtbp
from engine.crtbp import MU


def fisher_information(state0, measurements, prior_covariance=None, mu=MU):
    """
    Fisher information (6, 6) about the epoch state from a list of
    measurements (as built by estimation.simulate_measurements, only
    time_nondim, function and noise_sigma are used), evaluated along
    the trajectory that starts at state0.
    """
    information = np.zeros((6, 6)) if prior_covariance is None else np.linalg.inv(prior_covariance)
    times = np.array([m["time_nondim"] for m in measurements])
    if len(times) == 0:
        return information
    unique_times = np.unique(times)
    t_final = float(unique_times.max())
    sol = crtbp.propagate_with_stm(state0, t_final, mu, t_eval=unique_times) if t_final > 0.0 else None
    for measurement in measurements:
        t = measurement["time_nondim"]
        if sol is not None and t > 0.0:
            column = int(np.searchsorted(unique_times, t))
            state_t, phi = crtbp.split_state_and_stm(sol.y[:, column])
        else:
            state_t, phi = np.asarray(state0, dtype=float), np.eye(6)
        h = measurement["function"].jacobian(state_t) @ phi
        r_inverse = np.diag(1.0 / np.asarray(measurement["noise_sigma"], dtype=float) ** 2)
        information = information + h.T @ r_inverse @ h
    return information


def cramer_rao_bound(information):
    """
    Covariance lower bound (6, 6) from the information matrix, and the
    position and velocity one-sigma bounds it implies in km and m/s.
    Returns (covariance, position_sigma_km, velocity_sigma_m_s).
    """
    covariance = np.linalg.inv(information)
    position_sigma = crtbp.length_to_km(np.sqrt(np.trace(covariance[:3, :3])))
    velocity_sigma = crtbp.velocity_to_km_s(np.sqrt(np.trace(covariance[3:, 3:]))) * 1000.0
    return covariance, position_sigma, velocity_sigma


def information_along_orbit(orbit, measurements_at_phase, phases, prior_covariance=None, mu=MU):
    """
    Position bound (km) as a function of where on the orbit the epoch
    sits, for a fixed observation schedule relative to the epoch.

    orbit                : dictionary with state0 and period
    measurements_at_phase: callable(state0) -> measurement list, so the
                           caller decides the schedule (for example the
                           same nights, re-evaluated at the new epoch)
    phases               : fractions of the period at which to place the
                           epoch (0 is the family's perilune crossing)
    Returns an array of position bounds, one per phase.
    """
    bounds = np.zeros(len(phases))
    for k, phase in enumerate(phases):
        state0 = orbit["state0"] if phase == 0.0 else crtbp.propagate(orbit["state0"], phase * orbit["period"], mu).y[:, -1]
        information = fisher_information(state0, measurements_at_phase(state0), prior_covariance, mu)
        _, bounds[k], _ = cramer_rao_bound(information)
    return bounds
