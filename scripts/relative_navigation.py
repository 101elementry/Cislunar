"""
Study B: can a filter use the range information the three-body motion
puts into a chaser's camera angles?  EKF and UKF, Monte Carlo, two laps
of the 9:2 NRHO.

    python scripts/relative_navigation.py [monte carlo runs]

Truth.  The target is member 49 started at apolune; the chaser starts
50 km behind it along-track in the target's LVLH frame about the Moon
with no velocity in that frame; both are flown for two laps (13.1 days)
in the full CRTBP with no burns.  Started at apolune the pair stays
between about 15 and 300 km apart; started near perilune the chaser
would drift thousands of km away within a lap (see Study A).  The
camera, its constraints and its noise are those of Study A:
20 arcsec per axis every 10 minutes whenever the target is sunlit,
bright enough, and clear of the Sun, Earth and Moon.

Estimation problem.  The chaser knows its own state perfectly (stated
assumption); the filters estimate the target's rotating-frame state
with the engine's EKF and UKF unchanged, fed by
relative_navigation.make_camera_measurement.  Process noise
1e-9 LU/TU^2 (about 3e-9 m/s^2): the truth has no unmodelled
acceleration, so it only keeps the covariance from collapsing; it is
the smallest level the ground pipeline found honest.

Initial knowledge.  One-sigma 5 km along the line of sight, 0.5 km
across it on each axis, 10 cm/s per velocity axis; each run draws its
initial error from that covariance, so an honest filter starts
consistent.  This is the situation Woffinden and Geller describe:
the direction is known far better than the range.  The velocity sigma
is kept loose on purpose.  A relative trajectory scaled up by 10 %
needs its relative velocity scaled by 10 % too, about 1 cm/s at 50 km
near apolune, so a 1 cm/s prior would fix the range at the
few-kilometre level through the velocity alone (tried: the bound then
falls from 5 to 2.8 km in the first hours, before any perilune).  At
10 cm/s the prior says nothing about range beyond its own 5 km.

Starts, as in the ground pipeline (scripts/orbit_determination.py):
  cold : the filter from the initial guess and covariance, all data;
  warm : batch least squares (growing arc, the initial covariance as its
         prior) on the first 1 day or the first 4 days, handed with its
         covariance mapped by the STM to the filter for the rest.  The
         first perilune passage comes at day 3.28, so the 4-day batch
         contains it and the 1-day batch does not.

Reference.  The EKF run from the true state on noise-free measurements
never leaves the truth, so its covariance is the covariance recursion
linearised about the truth: the posterior Cramer-Rao bound for this
measurement history with this prior and process noise.  No filter
started from an error can beat it on average.

Consistency.  NEES (six degrees of freedom, needs the truth) and NIS
(two, does not) averaged over the runs at every measurement time, and
over time, against the 95 % chi-squared band for the mean of N runs,
as in rung 2.

Outputs: output/relative_navigation.csv (one row per filter and start)
and output/fig16_relative_navigation.png.
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import access, crtbp, estimation, frames, geometry, propagation, rendezvous
from engine import relative_navigation
from model.ephemeris import load_ephemeris
from model.family import load_families
from model.scenario import rendezvous_example
from model import runner

n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
EPOCH_UTC = "2026-01-01T00:00:00"
ARCSEC = 1.0 / 3600.0
NOISE = np.array([20.0 * ARCSEC, 20.0 * ARCSEC])            # degrees, per axis
CADENCE_S = 600.0
START_PHASE = 0.5                                           # apolune
SEPARATION_KM = 50.0
LAPS = 2.0
INITIAL_RANGE_SIGMA_KM = 5.0
INITIAL_CROSS_RANGE_SIGMA_KM = 0.5                          # per cross-range axis
INITIAL_VELOCITY_SIGMA_M_S = 0.10                           # per axis
ACCELERATION_SIGMA = 1e-9                                   # LU/TU^2
WARM_DAYS = [1.0, 4.0]

families = load_families()
orbit = families["L2 southern halo"][49]
period = orbit["period"]
ephemeris = load_ephemeris()
if ephemeris is None:
    raise SystemExit("run scripts/fetch_ephemeris.py first")

example = rendezvous_example()
target_object = example.spacecraft[0]
chaser_object = example.spacecraft[1]
camera = example.sensors[0]
camera.max_range_km = 0.0
camera.earth_exclusion_deg = 10.0
camera_constraints = runner.constraints_for(chaser_object, camera)

# ---- truth ----
target0 = crtbp.propagate(orbit["state0"], START_PHASE * period).y[:, -1]
chaser0 = rendezvous.state_from_lvlh_offset(target0, [0.0, -SEPARATION_KM, 0.0], [0.0, 0.0, 0.0], "moon")
times_s = np.arange(0.0, LAPS * crtbp.time_to_seconds(period), CADENCE_S)
times = crtbp.time_to_nondim(times_s)
truth = propagation.propagate_state(target0, times)
chaser = propagation.propagate_state(chaser0, times)
jd = frames.julian_dates_for_grid(EPOCH_UTC, times_s)
series = geometry.space_observation_geometry(chaser, truth, times_s, jd, target_object.diameter_m,
                                             target_object.albedo, ephemeris=ephemeris)
mask = access.access_mask(series, camera_constraints)
separation = crtbp.length_to_km(np.linalg.norm(truth[:, :3] - chaser[:, :3], axis=1))
days_grid = times_s / 86400.0
perilune_days = [(k + 0.5) * crtbp.time_to_days(period) for k in range(int(LAPS))]
print(f"two laps from apolune: separation {separation.min():.1f} to {separation.max():.1f} km, "
      f"camera excluded {100.0 * (1.0 - mask.mean()):.1f} % of the time, {int(mask.sum())} measurements, "
      f"perilune at day {', '.join(f'{d:.2f}' for d in perilune_days)}")

# ---- initial covariance, aligned with the line of sight ----
unit = relative_navigation.line_of_sight_unit(target0, chaser0)
across_projector = np.eye(3) - np.outer(unit, unit)
initial_covariance = np.zeros((6, 6))
initial_covariance[:3, :3] = crtbp.length_to_nondim(INITIAL_RANGE_SIGMA_KM) ** 2 * np.outer(unit, unit) \
    + crtbp.length_to_nondim(INITIAL_CROSS_RANGE_SIGMA_KM) ** 2 * across_projector
initial_covariance[3:, 3:] = np.eye(3) * crtbp.velocity_to_nondim(INITIAL_VELOCITY_SIGMA_M_S / 1000.0) ** 2


def measurements_for(seed):
    """One noisy measurement set of the whole two laps."""
    return relative_navigation.simulate_camera_measurements(truth, chaser, times, NOISE, mask=mask, seed=seed)


def run_filter(filter_name, state, covariance, measurements):
    """
    EKF or UKF through a measurement list starting at time zero.

    The UKF runs with alpha = 1 (sigma points at plus and minus sqrt(6)
    sigma, all covariance weights non-negative) rather than the engine's
    default 1e-3.  With the cross-range known to a fraction of a metre,
    the default's sigma points sit about 1e-12 LU apart, the size of the
    integrator's own error at its 1e-12 tolerance, and its central
    covariance weight of about -1e6 turns that noise into a covariance
    that is not positive definite (it failed at the second perilune).
    The ground pipeline's kilometre-level uncertainty never meets this.
    """
    if filter_name == "ekf":
        return estimation.extended_kalman_filter(state, covariance, measurements, ACCELERATION_SIGMA)
    return estimation.unscented_kalman_filter(state, covariance, measurements, ACCELERATION_SIGMA, alpha=1.0)


def score(result, measurements, time_offset):
    """
    Per-measurement range and cross-range errors and formal sigmas (km),
    NEES and NIS, for a filter result whose times start time_offset TU
    after the truth grid's zero.
    """
    shifted = dict(result, times=result["times"] + time_offset)
    nees = estimation.normalised_estimation_error_squared(shifted, truth, times)
    nis = estimation.normalised_innovation_squared(result)
    range_error = np.zeros(len(measurements))
    cross_error = np.zeros(len(measurements))
    range_sigma = np.zeros(len(measurements))
    cross_sigma = np.zeros(len(measurements))
    for k, measurement in enumerate(measurements):
        index = measurement["index"]
        range_error[k], cross_error[k] = relative_navigation.range_and_cross_range_errors_km(
            result["estimates"][k], truth[index], chaser[index])
        range_sigma[k], cross_sigma[k] = relative_navigation.range_and_cross_range_sigma_km(
            result["covariances"][k], truth[index], chaser[index])
    return {"days": days_grid[[m["index"] for m in measurements]], "range_error": range_error,
            "cross_error": cross_error, "range_sigma": range_sigma, "cross_sigma": cross_sigma,
            "nees": nees, "nis": nis}


# ---- reference: the posterior Cramer-Rao bound along the truth ----
noise_free = [dict(m, value=m["function"](truth[m["index"]])) for m in measurements_for(0)]
reference = score(estimation.extended_kalman_filter(truth[0], initial_covariance, noise_free, ACCELERATION_SIGMA),
                  noise_free, 0.0)
lap_index = int(np.searchsorted(reference["days"], crtbp.time_to_days(period))) - 1
print(f"posterior Cramer-Rao bound (EKF along the truth): range {reference['range_sigma'][lap_index]:.3f} km after "
      f"one lap, {reference['range_sigma'][-1]:.3f} km after two; cross-range "
      f"{1000.0 * reference['cross_sigma'][-1]:.2f} m")

# ---- Monte Carlo ----
rng = np.random.default_rng(2026)
cases = []
for run in range(n_runs):
    measurements = measurements_for(100 + run)
    initial_state = truth[0] + rng.multivariate_normal(np.zeros(6), initial_covariance)
    cases.append((measurements, initial_state))

starts = ["cold"] + [f"warm {days:g} d" for days in WARM_DAYS]
histories = {}
batch_notes = {}
for start in starts:
    handed = []
    started = time.perf_counter()
    batch_errors = []
    for measurements, initial_state in cases:
        if start == "cold":
            handed.append((initial_state, initial_covariance, measurements, 0.0))
            continue
        warm_days = float(start.split()[1])
        warm = [m for m in measurements if crtbp.time_to_days(m["time_nondim"]) <= warm_days]
        rest = [m for m in measurements if crtbp.time_to_days(m["time_nondim"]) > warm_days]
        batch = estimation.batch_least_squares_growing_arc(initial_state, warm, prior_covariance=initial_covariance,
                                                           stages=3)
        range_error, _ = relative_navigation.range_and_cross_range_errors_km(batch["state"], truth[0], chaser[0])
        batch_errors.append((range_error, batch["rms_normalised_residual"]))
        t_hand = warm[-1]["time_nondim"]
        solution = crtbp.propagate_with_stm(batch["state"], t_hand)
        state_hand, phi = crtbp.split_state_and_stm(solution.y[:, -1])
        covariance_hand = phi @ batch["covariance"] @ phi.T
        shifted = [dict(m, time_nondim=m["time_nondim"] - t_hand) for m in rest]
        handed.append((state_hand, covariance_hand, shifted, t_hand))
    if batch_errors:
        errors = np.array(batch_errors)
        batch_notes[start] = errors
        print(f"{start} batch: epoch range error rms {np.sqrt(np.mean(errors[:, 0] ** 2)):.3f} km, "
              f"normalised residual rms {np.mean(errors[:, 1]):.3f}, {time.perf_counter() - started:.0f} s")
    for filter_name in ("ekf", "ukf"):
        started = time.perf_counter()
        runs = []
        for state, covariance, measurements, t_hand in handed:
            result = run_filter(filter_name, state, covariance, measurements)
            runs.append(score(result, measurements, t_hand))
        histories[(filter_name, start)] = runs
        print(f"  {filter_name} {start}: {n_runs} runs in {time.perf_counter() - started:.0f} s")

# ---- summary ----
nees_lower, nees_upper = estimation.chi_squared_bounds(6, n_runs)
nis_lower, nis_upper = estimation.chi_squared_bounds(2, n_runs)
print(f"\n{n_runs}-run Monte Carlo.  Honest: NEES in [{nees_lower:.2f}, {nees_upper:.2f}], "
      f"NIS in [{nis_lower:.2f}, {nis_upper:.2f}]")
print(f"{'filter':>6s} {'start':>10s} {'mean NEES':>10s} {'mean NIS':>9s} {'NEES in band':>12s} "
      f"{'range km lap 1':>14s} {'range km end':>12s} {'1-sigma end':>11s} {'cross m end':>11s}")
rows = []
for (filter_name, start), runs in histories.items():
    nees_time = np.mean([run["nees"] for run in runs], axis=0)
    nis_time = np.mean([run["nis"] for run in runs], axis=0)
    days = runs[0]["days"]
    lap_end = int(np.searchsorted(days, crtbp.time_to_days(period))) - 1
    range_lap = float(np.sqrt(np.mean([run["range_error"][lap_end] ** 2 for run in runs])))
    range_end = float(np.sqrt(np.mean([run["range_error"][-1] ** 2 for run in runs])))
    cross_end = float(np.sqrt(np.mean([run["cross_error"][-1] ** 2 for run in runs])))
    row = {"filter": filter_name, "start": start, "runs": n_runs, "acceleration_sigma": ACCELERATION_SIGMA,
           "mean_nees": float(nees_time.mean()), "median_nees": float(np.median(nees_time)),
           "mean_nis": float(nis_time.mean()),
           "fraction_nees_in_band": float(np.mean((nees_time >= nees_lower) & (nees_time <= nees_upper))),
           "nees_lower": nees_lower, "nees_upper": nees_upper, "nis_lower": nis_lower, "nis_upper": nis_upper,
           "range_error_km_rms_lap1": range_lap, "range_error_km_rms_final": range_end,
           "range_sigma_km_final": float(np.mean([run["range_sigma"][-1] for run in runs])),
           "crlb_range_sigma_km_final": float(reference["range_sigma"][-1]),
           "cross_range_error_km_rms_final": cross_end,
           "cross_range_sigma_km_final": float(np.mean([run["cross_sigma"][-1] for run in runs])),
           "crlb_cross_range_sigma_km_final": float(reference["cross_sigma"][-1])}
    rows.append(row)
    print(f"{filter_name:>6s} {start:>10s} {row['mean_nees']:10.3g} {row['mean_nis']:9.2f} "
          f"{100.0 * row['fraction_nees_in_band']:11.0f}% {range_lap:14.3f} {range_end:12.3f} "
          f"{row['range_sigma_km_final']:11.3f} {1000.0 * cross_end:11.2f}")
print(f"{'CRLB':>6s} {'':>10s} {'':>10s} {'':>9s} {'':>12s} {reference['range_sigma'][lap_index]:14.3f} "
      f"{reference['range_sigma'][-1]:12.3f} {'':>11s} {1000.0 * reference['cross_sigma'][-1]:11.2f}")

csv_path = os.path.join("output", "relative_navigation.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {csv_path}")

# ---- figure ----
shown = [("ekf", "cold"), ("ukf", "cold"), ("ekf", f"warm {WARM_DAYS[-1]:g} d"), ("ukf", f"warm {WARM_DAYS[-1]:g} d")]
colours = {("ekf", "cold"): "#2a78d6", ("ukf", "cold"): "#eb6834",
           ("ekf", shown[2][1]): "#1baf7a", ("ukf", shown[3][1]): "#8a4fd8"}
figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for key in shown:
    runs = histories[key]
    days = runs[0]["days"]
    label = f"{key[0].upper()} {key[1]}"
    range_rms = np.sqrt(np.mean([run["range_error"] ** 2 for run in runs], axis=0))
    range_sigma = np.mean([run["range_sigma"] for run in runs], axis=0)
    cross_rms = np.sqrt(np.mean([run["cross_error"] ** 2 for run in runs], axis=0))
    cross_sigma = np.mean([run["cross_sigma"] for run in runs], axis=0)
    axes[0, 0].semilogy(days, range_rms, color=colours[key], lw=1.2, label=label)
    axes[0, 0].semilogy(days, range_sigma, color=colours[key], lw=0.8, ls="--")
    axes[0, 1].semilogy(days, 1000.0 * cross_rms, color=colours[key], lw=1.2, label=label)
    axes[0, 1].semilogy(days, 1000.0 * cross_sigma, color=colours[key], lw=0.8, ls="--")
    axes[1, 0].semilogy(days, np.mean([run["nees"] for run in runs], axis=0), color=colours[key], lw=1.0, label=label)
    axes[1, 1].semilogy(days, np.mean([run["nis"] for run in runs], axis=0), color=colours[key], lw=0.6, label=label)
axes[0, 0].semilogy(reference["days"], reference["range_sigma"], color="0.1", lw=1.4, ls=":",
                    label="Cramer-Rao bound (EKF along truth)")
axes[0, 1].semilogy(reference["days"], 1000.0 * reference["cross_sigma"], color="0.1", lw=1.4, ls=":",
                    label="Cramer-Rao bound")
axes[1, 0].axhspan(nees_lower, nees_upper, color="#43d19a", alpha=0.25, label="95 % band")
axes[1, 1].axhspan(nis_lower, nis_upper, color="#43d19a", alpha=0.25, label="95 % band")
for ax in axes.flat:
    for day in perilune_days:
        ax.axvline(day, color="0.6", lw=0.8, ls="-.")
    ax.grid(True, which="major", color="0.9", lw=0.6)
# Line-style key shared by the two error panels, drawn as proxy lines.
style_key = [matplotlib.lines.Line2D([], [], color="0.4", lw=1.2, label="solid: rms error over runs"),
             matplotlib.lines.Line2D([], [], color="0.4", lw=0.8, ls="--", label="dashed: filter's own 1-sigma")]
for ax, place in zip(axes[0], ("lower left", "upper center")):
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + style_key, fontsize=7, loc=place, ncol=2)
axes[0, 0].set_ylabel("range error [km]")
axes[0, 0].set_title("Range (along the line of sight)", fontsize=10)
axes[0, 1].set_ylabel("cross-range error [m]")
axes[0, 1].set_title("Cross-range (across the line of sight)", fontsize=10)
axes[1, 0].set_ylabel(f"NEES, mean of {n_runs} runs")
axes[1, 0].set_title("NEES against its chi-squared band (6 degrees of freedom)", fontsize=10)
axes[1, 0].legend(fontsize=7, loc="upper left")
axes[1, 1].set_ylabel(f"NIS, mean of {n_runs} runs")
axes[1, 1].set_title("NIS against its chi-squared band (2 degrees of freedom)", fontsize=10)
axes[1, 1].legend(fontsize=7, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel("days from the start at apolune (dash-dot lines: perilune)")
figure.suptitle(f"Angles-only relative navigation of a target on the 9:2 NRHO from a chaser {SEPARATION_KM:g} km "
                f"behind, 20 arcsec camera, {n_runs}-run Monte Carlo", fontsize=11)
figure.tight_layout()
figure_path = os.path.join("output", "fig16_relative_navigation.png")
figure.savefig(figure_path, dpi=150)
print(f"saved {figure_path}")
