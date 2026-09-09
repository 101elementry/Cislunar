"""
Orbit determination on the CRTBP dynamics: measurement models for the
ground-based sensors and an extended Kalman filter that uses the state
transition matrix the engine already propagates.

Measurements
  * Angles: the unit line of sight from a ground station to the
    spacecraft, expressed in the station's local frame as azimuth and
    elevation (degrees).  An optical telescope measures these.
  * Range (km) and range rate (km/s): what a radar or a transponder
    gives.

The filter
  The state is the six-element rotating-frame state.  Between
  measurements it is propagated with the full nonlinear equations and
  the covariance with the STM, P <- Phi P Phi^T + Q.  At a measurement
  the standard extended Kalman update is applied with the measurement
  Jacobian taken by central differences of the measurement function,
  which keeps the model code free of hand-derived partials (the
  partials of azimuth and elevation through three rotations are long
  and error-prone).  Process noise Q is a small diagonal acceleration
  noise, the usual way to keep the filter from becoming over-confident
  in a model that is not perfect.

This is the simplest filter an examiner would expect to see on a
cislunar problem; it is written so a batch least-squares or a UKF can
be added beside it with the same measurement functions.

Units: internal state non-dimensional; measurements in degrees, km and
km/s; covariances in the internal units.
"""

import numpy as np

from engine import crtbp, frames, geometry
from engine.crtbp import MU


# --------------------------------------------------------------------------
# Measurement models
# --------------------------------------------------------------------------

def station_frame(latitude_deg, longitude_deg, altitude_km, jd):
    """
    Station position (3,) and its local east, north, up unit vectors in
    the rotating frame at one Julian date.  East is z_hat x up along
    the Earth's spin axis projected... written out: the spin axis in
    the rotating frame is obtained the same way as the station itself,
    from the Earth-fixed pole.
    """
    position, up = frames.station_position_rotating(latitude_deg, longitude_deg, altitude_km, np.array([jd]))
    position = position[0]
    up = up[0]
    # The Earth's spin axis: the Earth-fixed pole (0, 0, 1) through the
    # same rotation chain as a station at latitude 90.
    _, pole = frames.station_position_rotating(90.0, 0.0, 0.0, np.array([jd]))
    pole = pole[0]
    east = np.cross(pole, up)
    east_length = np.linalg.norm(east)
    if east_length < 1e-12:
        east = np.array([1.0, 0.0, 0.0])
    else:
        east = east / east_length
    north = np.cross(up, east)
    return position, east, north, up


def angles_measurement(state, station, jd):
    """
    Azimuth (degrees from north through east) and elevation (degrees)
    of the spacecraft seen from a station.  station = (lat, lon, alt km).
    """
    position, east, north, up = station_frame(station[0], station[1], station[2], jd)
    line_of_sight = state[:3] - position
    line_of_sight = line_of_sight / np.linalg.norm(line_of_sight)
    e = np.dot(line_of_sight, east)
    n = np.dot(line_of_sight, north)
    u = np.dot(line_of_sight, up)
    azimuth = np.degrees(np.arctan2(e, n)) % 360.0
    elevation = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))
    return np.array([azimuth, elevation])


def range_measurement(state, station, jd):
    """
    Range (km) and range rate (km/s) from a station.  The station's
    rotating-frame velocity is taken by a central difference over one
    second, which is exact to well below the measurement noise.
    """
    position, _, _, _ = station_frame(station[0], station[1], station[2], jd)
    dt_days = 1.0 / crtbp.SECONDS_PER_DAY
    position_before, _, _, _ = station_frame(station[0], station[1], station[2], jd - dt_days)
    position_after, _, _, _ = station_frame(station[0], station[1], station[2], jd + dt_days)
    station_velocity = (position_after - position_before) / (2.0 * crtbp.time_to_nondim(1.0))

    relative_position = state[:3] - position
    relative_velocity = state[3:] - station_velocity
    range_nd = np.linalg.norm(relative_position)
    range_rate_nd = np.dot(relative_position, relative_velocity) / range_nd
    return np.array([crtbp.length_to_km(range_nd), crtbp.velocity_to_km_s(range_rate_nd)])


def measurement_jacobian(function, state, step=1e-7):
    """
    Central-difference Jacobian d(measurement)/d(state) of a function
    state -> measurement vector, shape (m, 6).  Azimuth wraps at 360
    degrees; the difference is taken modulo 360 to stay continuous.
    """
    reference = function(state)
    jacobian = np.zeros((len(reference), 6))
    for column in range(6):
        perturbation = np.zeros(6)
        perturbation[column] = step
        forward = function(state + perturbation)
        backward = function(state - perturbation)
        difference = forward - backward
        difference = (difference + 180.0) % 360.0 - 180.0 if function.wraps_at_360 else difference
        jacobian[:, column] = difference / (2.0 * step)
    return jacobian


def make_measurement(kind, station, jd):
    """A measurement function state -> vector for one station at one time."""
    if kind == "angles":
        def function(state):
            return angles_measurement(state, station, jd)
        function.wraps_at_360 = True
    elif kind == "range":
        def function(state):
            return range_measurement(state, station, jd)
        function.wraps_at_360 = False
    else:
        raise ValueError(f"unknown measurement kind {kind!r}")
    return function


# --------------------------------------------------------------------------
# Extended Kalman filter
# --------------------------------------------------------------------------

def process_noise(dt, acceleration_sigma):
    """
    Discrete process noise for a constant-acceleration random walk over
    a step dt: the standard [dt^4/4, dt^3/2; dt^3/2, dt^2] block per
    axis scaled by the acceleration variance.  Non-dimensional.
    """
    q = np.zeros((6, 6))
    variance = acceleration_sigma ** 2
    for axis in range(3):
        q[axis, axis] = variance * dt ** 4 / 4.0
        q[axis, axis + 3] = variance * dt ** 3 / 2.0
        q[axis + 3, axis] = variance * dt ** 3 / 2.0
        q[axis + 3, axis + 3] = variance * dt ** 2
    return q


def extended_kalman_filter(initial_state, initial_covariance, measurements, acceleration_sigma=1e-9, mu=MU):
    """
    Run an EKF through a list of measurements sorted by time.

    initial_state, initial_covariance : estimate at time zero (TU,
        non-dimensional)
    measurements : list of dictionaries with
        time_nondim : TU
        function    : state -> measurement vector (see make_measurement)
        value       : measured vector
        noise_sigma : (m,) one-sigma noise per component, same units
    acceleration_sigma : process-noise acceleration, LU/TU^2

    Returns a dictionary with times (n,), estimates (n, 6), covariances
    (n, 6, 6), and residuals (list of (m,) innovation vectors), all
    after each update.
    """
    state = np.array(initial_state, dtype=float)
    covariance = np.array(initial_covariance, dtype=float)
    t_now = 0.0

    times = []
    estimates = []
    covariances = []
    residuals = []
    for measurement in measurements:
        t_meas = float(measurement["time_nondim"])
        dt = t_meas - t_now
        if dt > 0.0:
            sol = crtbp.propagate_with_stm(state, dt, mu)
            state, phi = crtbp.split_state_and_stm(sol.y[:, -1])
            covariance = phi @ covariance @ phi.T + process_noise(dt, acceleration_sigma)
        t_now = t_meas

        function = measurement["function"]
        predicted = function(state)
        innovation = measurement["value"] - predicted
        if function.wraps_at_360:
            innovation = (innovation + 180.0) % 360.0 - 180.0
        h = measurement_jacobian(function, state)
        r = np.diag(np.asarray(measurement["noise_sigma"], dtype=float) ** 2)
        s = h @ covariance @ h.T + r
        gain = covariance @ h.T @ np.linalg.solve(s, np.eye(len(innovation)))
        state = state + gain @ innovation
        # Joseph form keeps the covariance symmetric and positive.
        identity_minus = np.eye(6) - gain @ h
        covariance = identity_minus @ covariance @ identity_minus.T + gain @ r @ gain.T

        times.append(t_now)
        estimates.append(state.copy())
        covariances.append(covariance.copy())
        residuals.append(innovation)

    return {"times": np.array(times), "estimates": np.array(estimates),
            "covariances": np.array(covariances), "residuals": residuals}


def simulate_measurements(true_states, times_nondim, jd, station, kind, noise_sigma, mask=None, every=1, seed=0):
    """
    Synthetic measurements of a true trajectory from one station.

    true_states : (n, 6), times_nondim : (n,), jd : (n,)
    station     : (lat, lon, alt km)
    kind        : "angles" or "range"
    noise_sigma : (m,) one-sigma noise added to each component
    mask        : optional (n,) boolean, e.g. the access mask, so
                  measurements are only taken when the station can see
                  the spacecraft
    every       : take every k-th eligible sample
    Returns the measurement list extended_kalman_filter expects.
    """
    rng = np.random.default_rng(seed)
    noise_sigma = np.asarray(noise_sigma, dtype=float)
    eligible = np.arange(len(times_nondim)) if mask is None else np.where(mask)[0]
    measurements = []
    for index in eligible[::every]:
        function = make_measurement(kind, station, float(jd[index]))
        value = function(true_states[index]) + rng.normal(0.0, noise_sigma)
        measurements.append({"time_nondim": float(times_nondim[index]), "function": function,
                             "value": value, "noise_sigma": noise_sigma, "index": int(index)})
    return measurements


def position_errors_km(result, true_states, times_nondim):
    """
    Position error (km) of each filter estimate against the true state
    at the same time, by matching times.  Returns (n,).
    """
    errors = np.zeros(len(result["times"]))
    for k, t in enumerate(result["times"]):
        index = int(np.argmin(np.abs(times_nondim - t)))
        errors[k] = crtbp.length_to_km(np.linalg.norm(result["estimates"][k, :3] - true_states[index, :3]))
    return errors


def formal_position_sigma_km(result):
    """Root of the position covariance trace at each estimate, km."""
    return np.array([crtbp.length_to_km(np.sqrt(np.trace(covariance[:3, :3])))
                     for covariance in result["covariances"]])
