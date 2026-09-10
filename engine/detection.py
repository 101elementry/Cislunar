"""
Manoeuvre detection and estimation from filter innovations.

Detection (rung 3)
  A filter that has converged on an object's orbit predicts each new
  measurement with a covariance S.  If the object performed an
  impulsive burn since the last observation, the prediction is wrong
  by more than S allows, and the normalised innovation squared
      NIS_k = nu_k^T S_k^-1 nu_k
  is large.  Summed over the m measurements of one observing night
  the statistic is chi-squared with 2m degrees of freedom when there
  was no burn (two angles per measurement).  Declaring a manoeuvre when
  the night's sum exceeds the chi-squared quantile at a chosen false
  alarm probability is a Neyman-Pearson test on the innovations, the
  standard way to detect an unmodelled event without knowing what it
  is.  The test is applied per night because that is the unit of
  observing: the burn is between nights, and one night's residuals is
  what an operator has when deciding.

  The smallest detectable burn is found by Monte Carlo: for a burn
  size, location and gap, the truth is re-flown with the burn, noisy
  measurements are drawn, the filter is run, and the first night after
  the burn is tested.  The detection probability over the trials is
  compared with a target (0.9 here).

Estimation (rung 4)
  Once a burn is known to have happened, its vector is added to the
  batch solve as three more unknowns applied at the burn epoch: for
  measurements after t_b the partial with respect to the burn is
      d(y_k)/d(dv) = H_k Phi(t_k, t_b)[:, 3:6]
  because an impulsive burn is a velocity jump at t_b carried forward
  by the STM from t_b.  With the epoch unknown as well, the solve is
  repeated on a grid of candidate epochs between the two nights and
  the one with the smallest cost is taken: a profile likelihood over
  the epoch, which is not a linear parameter and cannot be solved for
  in the same step.

All internal quantities are non-dimensional; the interface takes and
returns metres per second, kilometres and days where it says so.
"""

import numpy as np

from engine import crtbp, estimation
from engine.crtbp import MU


# --------------------------------------------------------------------------
# Truth with a burn
# --------------------------------------------------------------------------

def trajectory_with_burn(state0, times_nondim, burn_time, delta_v, mu=MU):
    """
    (n, 6) states of an object that starts at state0, coasts to
    burn_time (TU), receives the impulsive velocity change delta_v (3,)
    in LU/TU in the rotating frame, and coasts on.  Samples at
    times_nondim, which must be increasing and start at zero.
    """
    times_nondim = np.asarray(times_nondim, dtype=float)
    before = times_nondim <= burn_time
    states = np.zeros((len(times_nondim), 6))
    if before.any():
        states[before] = crtbp.propagate(state0, float(times_nondim[before][-1]), mu,
                                         t_eval=times_nondim[before]).y.T if times_nondim[before][-1] > 0.0 \
            else np.tile(state0, (before.sum(), 1))
    state_at_burn = crtbp.propagate(state0, burn_time, mu).y[:, -1] if burn_time > 0.0 else np.array(state0, dtype=float)
    state_at_burn = state_at_burn.copy()
    state_at_burn[3:] = state_at_burn[3:] + np.asarray(delta_v, dtype=float)
    after = ~before
    if after.any():
        # Integrate from the burn to the remaining samples in one call by
        # shifting the clock to the burn instant.
        offsets = times_nondim[after] - burn_time
        sol = crtbp.propagate(state_at_burn, float(offsets[-1]), mu, t_eval=offsets)
        states[after] = sol.y.T
    return states


def velocity_direction(state, kind="along_track"):
    """
    Unit vector (3,) in the rotating frame for a burn: "along_track"
    (along the rotating-frame velocity, the cheapest way to change the
    period), "radial" (toward the Moon) or "normal" (along the orbital
    angular momentum about the Moon).
    """
    velocity = np.asarray(state[3:], dtype=float)
    to_moon = crtbp.moon_position() - np.asarray(state[:3], dtype=float)
    if kind == "along_track":
        direction = velocity
    elif kind == "radial":
        direction = to_moon
    elif kind == "normal":
        direction = np.cross(-to_moon, velocity)
    else:
        raise ValueError(f"unknown burn direction {kind!r}")
    return direction / np.linalg.norm(direction)


# --------------------------------------------------------------------------
# Nights and the per-night test
# --------------------------------------------------------------------------

def group_into_nights(measurements, gap_hours=6.0):
    """
    Split a time-ordered measurement list into nights: a new night
    starts wherever the gap to the previous measurement exceeds
    gap_hours.  Returns a list of index lists into `measurements`.
    """
    nights = []
    current = []
    last_time = None
    gap = crtbp.time_to_nondim(gap_hours * 3600.0)
    for index, measurement in enumerate(measurements):
        t = measurement["time_nondim"]
        if last_time is not None and t - last_time > gap and current:
            nights.append(current)
            current = []
        current.append(index)
        last_time = t
    if current:
        nights.append(current)
    return nights


def night_statistics(result, nights):
    """
    Per-night sum of the normalised innovation squared and its degrees
    of freedom, from a filter result that recorded innovation
    covariances.  Returns (sums (k,), degrees_of_freedom (k,)).
    """
    nis = estimation.normalised_innovation_squared(result)
    sums = np.array([nis[indices].sum() for indices in nights])
    degrees = np.array([sum(len(result["residuals"][i]) for i in indices) for indices in nights])
    return sums, degrees


def detection_threshold(degrees_of_freedom, false_alarm_probability=1e-3):
    """Chi-squared quantile above which a night's NIS sum declares a manoeuvre."""
    from scipy.stats import chi2
    return chi2.ppf(1.0 - false_alarm_probability, degrees_of_freedom)


def detect_manoeuvre(result, nights, false_alarm_probability=1e-3):
    """
    Apply the per-night chi-squared test.  Returns a dictionary with
    the per-night sums, thresholds, a boolean flag per night, and the
    index of the first flagged night (or None).
    """
    sums, degrees = night_statistics(result, nights)
    thresholds = detection_threshold(degrees, false_alarm_probability)
    flagged = sums > thresholds
    first = int(np.argmax(flagged)) if flagged.any() else None
    return {"night_sums": sums, "degrees_of_freedom": degrees, "thresholds": thresholds,
            "flagged": flagged, "first_flagged_night": first}


# --------------------------------------------------------------------------
# Monte Carlo minimum detectable burn
# --------------------------------------------------------------------------

def run_filter(filter_name, initial_state, initial_covariance, measurements, acceleration_sigma, mu=MU):
    if filter_name == "ukf":
        return estimation.unscented_kalman_filter(initial_state, initial_covariance, measurements, acceleration_sigma, mu)
    return estimation.extended_kalman_filter(initial_state, initial_covariance, measurements, acceleration_sigma, mu)


def detection_probability(state0, times_nondim, jd, station, access_mask, burn_time, delta_v_m_s, burn_direction,
                          noise_sigma, cadence_every, initial_covariance, acceleration_sigma, n_trials=8,
                          filter_name="ekf", false_alarm_probability=1e-3, seed=0, mu=MU, ephemeris=None,
                          warm_nights=2):
    """
    Fraction of Monte Carlo trials in which the first night after a
    burn is flagged.  Each trial draws its own measurement noise and
    its own initial guess error from initial_covariance.  The first
    warm_nights nights are solved by batch (the initial orbit
    determination) and hand state and covariance to the filter, which
    then runs through every later night, exactly as an operator's
    filter would; the nights between the hand-over and the burn are
    the filter's own warm-up and the baseline of its innovations.

    Returns (probability, detected_flags, night_of_burn) where
    night_of_burn is the index, among the filtered nights, of the first
    night after the burn.
    """
    rng = np.random.default_rng(seed)
    state_at_burn = crtbp.propagate(state0, burn_time, mu).y[:, -1] if burn_time > 0.0 else np.array(state0)
    direction = velocity_direction(state_at_burn, burn_direction)
    delta_v = crtbp.velocity_to_nondim(delta_v_m_s / 1000.0) * direction
    truth = trajectory_with_burn(state0, times_nondim, burn_time, delta_v, mu)

    detected = []
    night_of_burn = None
    for trial in range(n_trials):
        measurements = estimation.simulate_measurements(truth, times_nondim, jd, station, "radec", noise_sigma,
                                                        mask=access_mask, every=cadence_every,
                                                        seed=int(rng.integers(1 << 30)), ephemeris=ephemeris)
        nights = group_into_nights(measurements)
        if len(nights) <= warm_nights:
            return 0.0, [], None
        warm = [measurements[i] for k in range(warm_nights) for i in nights[k]]
        rest = [measurements[i] for k in range(warm_nights, len(nights)) for i in nights[k]]
        if warm[-1]["time_nondim"] >= burn_time:
            raise ValueError("the burn must come after the warm-up nights")
        error = rng.multivariate_normal(np.zeros(6), initial_covariance)
        iod = estimation.batch_least_squares_growing_arc(truth[0] + error, warm, prior_covariance=initial_covariance * 100.0,
                                                         stages=2, mu=mu)
        t_hand = warm[-1]["time_nondim"]
        state_hand, phi = crtbp.split_state_and_stm(crtbp.propagate_with_stm(iod["state"], t_hand, mu).y[:, -1])
        covariance_hand = phi @ iod["covariance"] @ phi.T
        shifted = [dict(m, time_nondim=m["time_nondim"] - t_hand) for m in rest]
        result = run_filter(filter_name, state_hand, covariance_hand, shifted, acceleration_sigma, mu)
        filtered_nights = group_into_nights(shifted)
        after = [k for k, indices in enumerate(filtered_nights) if shifted[indices[0]]["time_nondim"] + t_hand > burn_time]
        if not after:
            return 0.0, [], None
        night_of_burn = after[0]
        test = detect_manoeuvre(result, filtered_nights, false_alarm_probability)
        detected.append(bool(test["flagged"][night_of_burn]))
    return float(np.mean(detected)), detected, night_of_burn


# --------------------------------------------------------------------------
# Manoeuvre estimation (rung 4)
# --------------------------------------------------------------------------

def batch_with_burn(initial_state, measurements, burn_time, prior_covariance=None, iterations=15, mu=MU,
                    verbose=False):
    """
    Batch least squares for the epoch state and an impulsive burn at a
    known epoch burn_time (TU): nine unknowns.  Measurements before
    the burn see only the state; measurements after it see the state
    through Phi(t_k, 0) and the burn through Phi(t_k, t_b)[:, 3:6].
    Backtracking as in batch_least_squares.

    Returns a dictionary with state (6,), delta_v (3,) LU/TU,
    delta_v_m_s (3,), covariance (9, 9), cost and rms normalised
    residual.
    """
    ordered = sorted(measurements, key=lambda m: m["time_nondim"])
    times = np.array([m["time_nondim"] for m in ordered])
    prior_information = np.zeros((9, 9))
    if prior_covariance is not None:
        prior_information[:6, :6] = np.linalg.inv(prior_covariance)
    prior_parameters = np.concatenate([np.array(initial_state, dtype=float), np.zeros(3)])
    parameters = prior_parameters.copy()

    def evaluate(parameters_now):
        state0 = parameters_now[:6]
        delta_v = parameters_now[6:]
        # Trajectory to the burn, then from the burn, both with STMs.
        before = times <= burn_time
        information = prior_information.copy()
        gradient = -prior_information @ (parameters_now - prior_parameters)
        normalised = []
        residuals = []
        # STM to every pre-burn time.
        pre_times = np.unique(times[before])
        pre_sol = crtbp.propagate_with_stm(state0, float(pre_times.max()), mu, t_eval=pre_times) if len(pre_times) and pre_times.max() > 0.0 else None
        # State and STM at the burn, then apply it.
        at_burn = crtbp.propagate_with_stm(state0, burn_time, mu).y[:, -1] if burn_time > 0.0 else np.concatenate([state0, np.eye(6).reshape(36)])
        state_burn, phi_burn = crtbp.split_state_and_stm(at_burn)
        state_burn = state_burn.copy()
        state_burn[3:] = state_burn[3:] + delta_v
        post_times = np.unique(times[~before])
        post_sol = None
        if len(post_times):
            post_sol = crtbp.propagate_with_stm(state_burn, float(post_times.max() - burn_time), mu,
                                                t_eval=post_times - burn_time)
        for measurement in ordered:
            t = measurement["time_nondim"]
            function = measurement["function"]
            if t <= burn_time:
                if pre_sol is not None and t > 0.0:
                    state_t, phi = crtbp.split_state_and_stm(pre_sol.y[:, int(np.searchsorted(pre_times, t))])
                else:
                    state_t, phi = state0, np.eye(6)
                h_state = function.jacobian(state_t) @ phi
                h_burn = np.zeros((h_state.shape[0], 3))
            else:
                state_t, phi_from_burn = crtbp.split_state_and_stm(post_sol.y[:, int(np.searchsorted(post_times, t))])
                h_t = function.jacobian(state_t)
                h_state = h_t @ phi_from_burn @ phi_burn
                h_burn = h_t @ phi_from_burn[:, 3:6]
            residual = measurement["value"] - function(state_t)
            if function.wraps_at_360:
                residual = (residual + 180.0) % 360.0 - 180.0
            h = np.hstack([h_state, h_burn])
            r_inverse = np.diag(1.0 / np.asarray(measurement["noise_sigma"], dtype=float) ** 2)
            information = information + h.T @ r_inverse @ h
            gradient = gradient + h.T @ r_inverse @ residual
            residuals.append(residual)
            normalised.append(residual / np.asarray(measurement["noise_sigma"], dtype=float))
        deviation = parameters_now - prior_parameters
        cost = float(np.sum(np.concatenate(normalised) ** 2)) + deviation @ prior_information @ deviation
        return residuals, normalised, information, gradient, cost

    residuals, normalised, information, gradient, cost = evaluate(parameters)
    for iteration in range(iterations):
        correction = np.linalg.solve(information, gradient)
        step = 1.0
        for _ in range(12):
            trial = parameters + step * correction
            trial_evaluation = evaluate(trial)
            if trial_evaluation[4] < cost:
                break
            step = step * 0.5
        parameters = trial
        residuals, normalised, information, gradient, cost = trial_evaluation
        if verbose:
            print(f"  burn batch iteration {iteration}: step {step:g}, cost {cost:.1f}")
        if np.linalg.norm(step * correction) < 1e-10:
            break
    covariance = np.linalg.inv(information)
    rms = float(np.sqrt(np.mean(np.concatenate(normalised) ** 2)))
    return {"state": parameters[:6], "delta_v": parameters[6:],
            "delta_v_m_s": crtbp.velocity_to_km_s(parameters[6:]) * 1000.0,
            "delta_v_sigma_m_s": crtbp.velocity_to_km_s(np.sqrt(np.diag(covariance)[6:])) * 1000.0,
            "covariance": covariance, "cost": cost, "rms_normalised_residual": rms}


def estimate_burn_epoch(initial_state, measurements, candidate_times, prior_covariance=None, mu=MU, verbose=False):
    """
    Profile the batch cost over candidate burn epochs (TU) and return
    the best solve together with the cost curve: (best_time, best_result,
    costs).  The minimum of the cost curve is the burn epoch estimate;
    its sharpness says how well the epoch is determined.
    """
    costs = np.zeros(len(candidate_times))
    results = []
    for k, t_b in enumerate(candidate_times):
        result = batch_with_burn(initial_state, measurements, float(t_b), prior_covariance, mu=mu)
        costs[k] = result["cost"]
        results.append(result)
        if verbose:
            print(f"  candidate epoch {crtbp.time_to_days(t_b):7.3f} d: cost {result['cost']:.1f}, "
                  f"|dv| = {np.linalg.norm(result['delta_v_m_s']):.3f} m/s")
    best = int(np.argmin(costs))
    return float(candidate_times[best]), results[best], costs
