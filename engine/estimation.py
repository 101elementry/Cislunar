"""
Orbit determination on the CRTBP dynamics: measurement models for the
ground-based sensors and an extended Kalman filter that uses the state
transition matrix the engine already propagates.

Measurements
  * Right ascension and declination (degrees): the unit line of sight
    from a ground station to the spacecraft on the ICRF equatorial
    basis.  This is what an astrometric telescope reports after a
    plate solution against the star catalogue, and the thesis's
    measurement.
  * Azimuth and elevation (degrees): the same line of sight on the
    station's local basis; used by the access constraints.
  * Range (km) and range rate (km/s): what a radar or a transponder
    gives.

Both angle pairs have the same analytic Jacobian, derived once in
angle_pair_jacobian: the two rows are perpendicular to the line of
sight (angles carry no range information) and scale as one over the
range.  measurement_jacobian keeps the central-difference version as
the check the analytic one is tested against.

The estimators
  * Batch least squares: all measurements at once, solved for the
    state at the epoch by Gauss-Newton with the STM mapping each
    measurement partial back to the epoch.  The reference solution.
  * Extended Kalman filter: the state is propagated with the full
    nonlinear equations and the covariance with the STM,
    P <- Phi P Phi^T + Q, then updated with the analytic measurement
    Jacobian.  Process noise Q is a small diagonal acceleration noise.
  * Unscented Kalman filter: the same predict and update through sigma
    points integrated with the full dynamics, which keeps the
    nonlinearity of a perilune passage that the EKF linearises away.
  * Consistency: normalised estimation error squared (NEES, needs the
    truth) and normalised innovation squared (NIS, does not), the
    chi-squared statistics that say whether a filter's covariance is
    honest.  NIS is also the manoeuvre detector's statistic.

Units: internal state non-dimensional; measurements in degrees, km and
km/s; covariances in the internal units.

Units: internal state non-dimensional; measurements in degrees, km and
km/s; covariances in the internal units.
"""

import numpy as np

from engine import crtbp, frames, geometry
from engine.crtbp import MU


# --------------------------------------------------------------------------
# Measurement models
# --------------------------------------------------------------------------

def station_frame(latitude_deg, longitude_deg, altitude_km, jd, ephemeris=None):
    """
    Station position (3,) LU and its local east, north, up unit vectors
    in the rotating frame at one Julian date.  The Earth's spin axis in
    the rotating frame is obtained the same way as the station itself,
    from the Earth-fixed pole; east is the pole crossed with up, north
    completes the set.
    """
    position, up = frames.station_position_rotating(latitude_deg, longitude_deg, altitude_km, np.array([jd]), ephemeris)
    position = position[0]
    up = up[0]
    _, pole = frames.station_position_rotating(90.0, 0.0, 0.0, np.array([jd]), ephemeris)
    pole = pole[0]
    east = np.cross(pole, up)
    east_length = np.linalg.norm(east)
    if east_length < 1e-12:
        east = np.array([1.0, 0.0, 0.0])
    else:
        east = east / east_length
    north = np.cross(up, east)
    return position, east, north, up


def angle_pair(rho, basis_1, basis_2, basis_3):
    """
    Two angles of a line-of-sight vector on an orthonormal basis, degrees:
        first  = atan2(rho . b2, rho . b1)   (in [0, 360))
        second = atan2(rho . b3, sqrt((rho . b1)^2 + (rho . b2)^2))
    Right ascension and declination on the ICRF basis; azimuth (from b1
    through b2) and elevation on the north, east, up basis.
    """
    rho_1 = np.dot(rho, basis_1)
    rho_2 = np.dot(rho, basis_2)
    rho_3 = np.dot(rho, basis_3)
    first = np.degrees(np.arctan2(rho_2, rho_1)) % 360.0
    second = np.degrees(np.arctan2(rho_3, np.sqrt(rho_1 ** 2 + rho_2 ** 2)))
    return np.array([first, second])


def angle_pair_jacobian(rho, basis_1, basis_2, basis_3):
    """
    Partials of the two angles (degrees) with respect to the components
    of rho (in rho's own units), shape (2, 3).

    With s^2 = rho_1^2 + rho_2^2 and r^2 = s^2 + rho_3^2,
        d(first)/d(rho)  = (rho_1 b2 - rho_2 b1) / s^2
        d(second)/d(rho) = (s^2 b3 - rho_3 (rho_1 b1 + rho_2 b2)) / (r^2 s)
    Both rows are perpendicular to rho: angles say nothing about range.
    The first row grows as 1 / s, the singularity at the pole of the
    basis (declination 90 degrees, or the zenith for azimuth).
    """
    rho_1 = np.dot(rho, basis_1)
    rho_2 = np.dot(rho, basis_2)
    rho_3 = np.dot(rho, basis_3)
    s_squared = rho_1 ** 2 + rho_2 ** 2
    r_squared = s_squared + rho_3 ** 2
    d_first = (rho_1 * basis_2 - rho_2 * basis_1) / s_squared
    d_second = (s_squared * basis_3 - rho_3 * (rho_1 * basis_1 + rho_2 * basis_2)) / (r_squared * np.sqrt(s_squared))
    return np.degrees(np.vstack([d_first, d_second]))


def angles_measurement(state, station, jd, ephemeris=None):
    """
    Azimuth (degrees from north through east) and elevation (degrees)
    of the spacecraft seen from a station.  station = (lat, lon, alt km).
    """
    position, east, north, up = station_frame(station[0], station[1], station[2], jd, ephemeris)
    return angle_pair(state[:3] - position, north, east, up)


def angles_jacobian(state, station, jd, ephemeris=None):
    """Analytic (2, 6) Jacobian of azimuth and elevation, degrees per LU; velocity columns are zero."""
    position, east, north, up = station_frame(station[0], station[1], station[2], jd, ephemeris)
    jacobian = np.zeros((2, 6))
    jacobian[:, :3] = angle_pair_jacobian(state[:3] - position, north, east, up)
    return jacobian


def radec_measurement(state, station, jd, ephemeris=None):
    """
    Topocentric right ascension and declination (degrees, ICRF) of the
    spacecraft seen from a station.  The spacecraft and the station are
    both expressed in geocentric ICRF kilometres; light time (about
    1.3 s, moving the target by a few metres) is ignored.
    """
    spacecraft = frames.spacecraft_geocentric_icrf(np.asarray(state, dtype=float)[np.newaxis, :], np.array([jd]), ephemeris)[0]
    station_icrf, _ = frames.station_position_icrf(station[0], station[1], station[2], np.array([jd]),
                                                   precession=ephemeris is not None)
    rho = spacecraft - station_icrf[0]
    return angle_pair(rho, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def radec_jacobian(state, station, jd, ephemeris=None):
    """
    Analytic (2, 6) Jacobian of right ascension and declination with
    respect to the rotating-frame state, degrees per LU.  The chain is
    the angle partials with respect to the ICRF line of sight, times
    the rotation from rotating-frame to ICRF components, times the
    length unit.  Velocity columns are zero.
    """
    spacecraft = frames.spacecraft_geocentric_icrf(np.asarray(state, dtype=float)[np.newaxis, :], np.array([jd]), ephemeris)[0]
    station_icrf, _ = frames.station_position_icrf(station[0], station[1], station[2], np.array([jd]),
                                                   precession=ephemeris is not None)
    rho = spacecraft - station_icrf[0]
    d_angles_d_rho = angle_pair_jacobian(rho, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                                         np.array([0.0, 0.0, 1.0]))
    rotation = frames.rotating_to_icrf_matrix(np.array([jd]), ephemeris)[0]
    jacobian = np.zeros((2, 6))
    jacobian[:, :3] = d_angles_d_rho @ rotation * crtbp.LENGTH_UNIT_KM
    return jacobian


def range_measurement(state, station, jd, ephemeris=None):
    """
    Range (km) and range rate (km/s) from a station.  The station's
    rotating-frame velocity is taken by a central difference over one
    second, which is exact to well below the measurement noise.
    """
    position, _, _, _ = station_frame(station[0], station[1], station[2], jd, ephemeris)
    dt_days = 1.0 / crtbp.SECONDS_PER_DAY
    position_before, _, _, _ = station_frame(station[0], station[1], station[2], jd - dt_days, ephemeris)
    position_after, _, _, _ = station_frame(station[0], station[1], station[2], jd + dt_days, ephemeris)
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


def make_measurement(kind, station, jd, ephemeris=None):
    """
    A measurement function state -> vector for one station at one time,
    carrying `wraps_at_360` (first component is an angle in [0, 360))
    and `jacobian(state)`, analytic where one exists and the central
    difference otherwise.
    """
    if kind == "angles":
        def function(state):
            return angles_measurement(state, station, jd, ephemeris)
        function.wraps_at_360 = True
        function.jacobian = lambda state: angles_jacobian(state, station, jd, ephemeris)
    elif kind == "radec":
        def function(state):
            return radec_measurement(state, station, jd, ephemeris)
        function.wraps_at_360 = True
        function.jacobian = lambda state: radec_jacobian(state, station, jd, ephemeris)
    elif kind == "range":
        def function(state):
            return range_measurement(state, station, jd, ephemeris)
        function.wraps_at_360 = False
        function.jacobian = lambda state: measurement_jacobian(function, state)
    else:
        raise ValueError(f"unknown measurement kind {kind!r}")
    function.kind = kind
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
    innovation_covariances = []
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
        h = function.jacobian(state) if hasattr(function, "jacobian") else measurement_jacobian(function, state)
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
        innovation_covariances.append(s)

    return {"times": np.array(times), "estimates": np.array(estimates),
            "covariances": np.array(covariances), "residuals": residuals,
            "innovation_covariances": innovation_covariances}


def simulate_measurements(true_states, times_nondim, jd, station, kind, noise_sigma, mask=None, every=1, seed=0,
                          ephemeris=None):
    """
    Synthetic measurements of a true trajectory from one station.

    true_states : (n, 6), times_nondim : (n,), jd : (n,)
    station     : (lat, lon, alt km)
    kind        : "radec", "angles" or "range"
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
        function = make_measurement(kind, station, float(jd[index]), ephemeris)
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


# --------------------------------------------------------------------------
# Batch least squares
# --------------------------------------------------------------------------

def batch_least_squares(initial_state, measurements, prior_covariance=None, iterations=8, tolerance=1e-10, mu=MU,
                        verbose=False):
    """
    Gauss-Newton solution for the state at time zero from all the
    measurements at once.

    Each iteration integrates the current estimate with its STM to every
    measurement time, forms the residual y_k - h(x_k) and the partial
    H_k Phi(t_k, 0) with respect to the epoch state, and solves the
    normal equations
        (sum H^T R^-1 H + P0^-1) dx = sum H^T R^-1 residual - P0^-1 (x - x_prior)
    with the optional prior covariance P0 about the initial guess.
    The Gauss-Newton step is the linearised optimum; across a perilune
    passage the linearisation is only good near the solution, so the
    step is shortened by halving until the weighted cost actually
    falls (a backtracking line search), which is what keeps the solver
    from running away from a rough first guess.  The information
    matrix at convergence is the inverse of the covariance of the
    estimate; it is also the Fisher information of the measurement
    set, which observability.py reuses.

    Returns a dictionary with state (6,), covariance (6, 6),
    information (6, 6), residuals (list of arrays after the last
    iteration, in measurement units), rms_normalised_residual and the
    number of iterations.
    """
    state = np.array(initial_state, dtype=float)
    prior_state = state.copy()
    prior_information = np.zeros((6, 6)) if prior_covariance is None else np.linalg.inv(prior_covariance)
    times = np.array([m["time_nondim"] for m in measurements])
    unique_times = np.unique(times)
    t_final = float(unique_times.max()) if len(times) else 0.0

    def evaluate(state_epoch):
        """Residuals, normalised residuals, information and gradient at one epoch state."""
        sol = crtbp.propagate_with_stm(state_epoch, t_final, mu, t_eval=unique_times) if t_final > 0.0 else None
        information = prior_information.copy()
        right_hand_side = -prior_information @ (state_epoch - prior_state)
        residuals = []
        normalised = []
        for measurement in measurements:
            t = measurement["time_nondim"]
            if sol is not None and t > 0.0:
                column = int(np.searchsorted(unique_times, t))
                state_t, phi = crtbp.split_state_and_stm(sol.y[:, column])
            else:
                state_t, phi = state_epoch, np.eye(6)
            function = measurement["function"]
            residual = measurement["value"] - function(state_t)
            if function.wraps_at_360:
                residual = (residual + 180.0) % 360.0 - 180.0
            h = function.jacobian(state_t) @ phi
            r_inverse = np.diag(1.0 / np.asarray(measurement["noise_sigma"], dtype=float) ** 2)
            information = information + h.T @ r_inverse @ h
            right_hand_side = right_hand_side + h.T @ r_inverse @ residual
            residuals.append(residual)
            normalised.append(residual / np.asarray(measurement["noise_sigma"], dtype=float))
        prior_term = (state_epoch - prior_state) @ prior_information @ (state_epoch - prior_state)
        cost = float(np.sum(np.concatenate(normalised) ** 2)) + prior_term if normalised else prior_term
        return residuals, normalised, information, right_hand_side, cost

    residuals, normalised, information, right_hand_side, cost = evaluate(state)
    for iteration in range(iterations):
        correction = np.linalg.solve(information, right_hand_side)
        # Backtracking: accept the full Gauss-Newton step only if the
        # cost falls; otherwise halve it, up to twelve times.
        step = 1.0
        for _ in range(12):
            trial = state + step * correction
            trial_evaluation = evaluate(trial)
            if trial_evaluation[4] < cost:
                break
            step = step * 0.5
        state = trial
        residuals, normalised, information, right_hand_side, cost = trial_evaluation
        rms = float(np.sqrt(np.mean(np.concatenate(normalised) ** 2))) if normalised else 0.0
        if verbose:
            print(f"  batch iteration {iteration}: step {step:g}, |dx| = {np.linalg.norm(step * correction):.3e}, "
                  f"rms normalised residual {rms:.3f}")
        if np.linalg.norm(step * correction) < tolerance:
            break

    covariance = np.linalg.inv(information)
    return {"state": state, "covariance": covariance, "information": information, "residuals": residuals,
            "rms_normalised_residual": rms, "iterations": iteration + 1}


# --------------------------------------------------------------------------
# Unscented Kalman filter
# --------------------------------------------------------------------------

def sigma_points(mean, covariance, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    The 2n + 1 sigma points of the unscented transform (Julier and
    Uhlmann; Wan and van der Merwe scaling) and their mean and
    covariance weights.  With n = 6 the points sit at the mean and at
    plus and minus the columns of sqrt((n + lambda) P).
    """
    n = len(mean)
    lam = alpha ** 2 * (n + kappa) - n
    square_root = np.linalg.cholesky((n + lam) * covariance)
    points = np.zeros((2 * n + 1, n))
    points[0] = mean
    for i in range(n):
        points[1 + i] = mean + square_root[:, i]
        points[1 + n + i] = mean - square_root[:, i]
    weights_mean = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    weights_covariance = weights_mean.copy()
    weights_mean[0] = lam / (n + lam)
    weights_covariance[0] = lam / (n + lam) + (1.0 - alpha ** 2 + beta)
    return points, weights_mean, weights_covariance


def unscented_kalman_filter(initial_state, initial_covariance, measurements, acceleration_sigma=1e-9, mu=MU,
                            alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Run a UKF through a list of measurements sorted by time; same
    interface and return value as extended_kalman_filter.

    Predict: every sigma point is integrated with the full equations of
    motion, and the predicted mean and covariance are their weighted
    moments plus the process noise.  Update: the sigma points are pushed
    through the measurement function, giving the predicted measurement,
    its covariance S and the cross covariance, from which the Kalman
    gain follows without any Jacobian.  Angle wrap-around is handled by
    referring every sigma-point measurement to the first one.
    """
    state = np.array(initial_state, dtype=float)
    covariance = np.array(initial_covariance, dtype=float)
    t_now = 0.0

    times, estimates, covariances, residuals, innovation_covariances = [], [], [], [], []
    for measurement in measurements:
        t_meas = float(measurement["time_nondim"])
        dt = t_meas - t_now
        points, w_mean, w_cov = sigma_points(state, covariance, alpha, beta, kappa)
        if dt > 0.0:
            propagated = np.array([crtbp.propagate(point, dt, mu).y[:, -1] for point in points])
            state = w_mean @ propagated
            deviations = propagated - state
            covariance = (deviations.T * w_cov) @ deviations + process_noise(dt, acceleration_sigma)
            points, w_mean, w_cov = sigma_points(state, covariance, alpha, beta, kappa)
        t_now = t_meas

        function = measurement["function"]
        predicted_points = np.array([function(point) for point in points])
        if function.wraps_at_360:
            predicted_points[:, 0] = predicted_points[0, 0] + ((predicted_points[:, 0] - predicted_points[0, 0] + 180.0) % 360.0 - 180.0)
        predicted = w_mean @ predicted_points
        measurement_deviation = predicted_points - predicted
        state_deviation = points - state
        r = np.diag(np.asarray(measurement["noise_sigma"], dtype=float) ** 2)
        s = (measurement_deviation.T * w_cov) @ measurement_deviation + r
        cross = (state_deviation.T * w_cov) @ measurement_deviation
        gain = cross @ np.linalg.inv(s)
        innovation = measurement["value"] - predicted
        if function.wraps_at_360:
            innovation = (innovation + 180.0) % 360.0 - 180.0
        state = state + gain @ innovation
        covariance = covariance - gain @ s @ gain.T
        covariance = 0.5 * (covariance + covariance.T)

        times.append(t_now)
        estimates.append(state.copy())
        covariances.append(covariance.copy())
        residuals.append(innovation)
        innovation_covariances.append(s)

    return {"times": np.array(times), "estimates": np.array(estimates), "covariances": np.array(covariances),
            "residuals": residuals, "innovation_covariances": innovation_covariances}


# --------------------------------------------------------------------------
# Consistency statistics
# --------------------------------------------------------------------------

def normalised_estimation_error_squared(result, true_states, times_nondim):
    """
    NEES at every estimate: e^T P^-1 e with e the error against the
    truth at the same time.  For an honest filter each value is
    chi-squared with six degrees of freedom, mean six; averaged over N
    Monte Carlo runs the mean lies in the chi-squared interval
    [chi2_{6N}(0.025), chi2_{6N}(0.975)] / N with 95 % probability.
    Values far above six mean the covariance is too small (optimistic).
    """
    nees = np.zeros(len(result["times"]))
    for k, t in enumerate(result["times"]):
        index = int(np.argmin(np.abs(times_nondim - t)))
        error = result["estimates"][k] - true_states[index]
        nees[k] = error @ np.linalg.solve(result["covariances"][k], error)
    return nees


def normalised_innovation_squared(result):
    """
    NIS at every update: nu^T S^-1 nu with nu the innovation and S its
    predicted covariance.  Chi-squared with as many degrees of freedom
    as the measurement has components (two for angles) when the filter
    is consistent and the model is right.  Needs no truth, so it is the
    statistic a real operator has, and the one the manoeuvre detector
    tests.  Filters that do not record S (the EKF result) get it
    reconstructed from H P H^T + R.
    """
    if "innovation_covariances" in result:
        return np.array([nu @ np.linalg.solve(s, nu) for nu, s in zip(result["residuals"], result["innovation_covariances"])])
    raise ValueError("this result carries no innovation covariances; run the filter with record_innovations=True")


def chi_squared_bounds(degrees_of_freedom, n_runs=1, probability=0.95):
    """
    Two-sided acceptance interval for the average of n_runs chi-squared
    variables with the given degrees of freedom, using scipy.
    """
    from scipy.stats import chi2
    lower = chi2.ppf((1.0 - probability) / 2.0, degrees_of_freedom * n_runs) / n_runs
    upper = chi2.ppf(1.0 - (1.0 - probability) / 2.0, degrees_of_freedom * n_runs) / n_runs
    return lower, upper
