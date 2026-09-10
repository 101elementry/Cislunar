"""
Rung 2: orbit determination of the NRHO from angles-only observations,
and the honesty of the filters.

    python scripts/orbit_determination.py [days] [monte carlo runs]

One spacecraft (member 49), the Sydney telescope, RA/Dec every 10 min
inside the access windows with 2 arcsecond noise, the DE440 ephemeris.
Three estimators on the same data:

  1. Batch least squares with a growing arc, the reference: its
     normalised residual rms must be 1 when the noise model is right,
     and its covariance is the Cramer-Rao bound.
  2. The extended Kalman filter, for several process-noise levels.
  3. The unscented Kalman filter, for the same levels.

The filters are started the way an operator starts them: a batch
solution of the first nights (initial orbit determination, from a
100 km and 1 m/s guess) hands its state and covariance to the filter,
which then runs through the remaining nights.  Started cold from the
100 km guess instead, the EKF diverges across the first perilune
passage and even the UKF stays optimistic by a factor of several;
that experiment is reported too, because it is the reason for the
warm start.

For each filter and level a Monte Carlo of N runs (fresh noise and
fresh initial error each time) gives the average NEES against the
truth and the average NIS of the innovations.  An honest filter has
NEES near 6 and NIS near 2, inside the chi-squared bounds printed with
them.  NEES above the bound means the covariance is too small
(optimistic); below it, too large.  The level that puts the filter
inside the bounds is the tuned one and is used for rung 3.

Outputs: output/od_consistency.csv, output/fig9_orbit_determination.png
(error and 3-sigma against time for the tuned filters, NEES against
time with the bounds).
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

from engine import crtbp, estimation, observability
from model.scenario import example_scenario
from model.family import load_families
from model.ephemeris import load_ephemeris
from model import runner

days = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
WARM_DAYS = 3.0                                             # nights handed to the batch initial orbit determination
ARCSEC = 1.0 / 3600.0
NOISE = [2.0 * ARCSEC, 2.0 * ARCSEC]
CADENCE_S = 600.0
PROCESS_NOISE_LEVELS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]     # LU/TU^2

families = load_families()
ephemeris = load_ephemeris()
if ephemeris is None:
    raise SystemExit("run scripts/fetch_ephemeris.py first")

scenario = example_scenario()
scenario.duration_days = days
results = runner.run_scenario(scenario, families, ephemeris=ephemeris)
key = list(results["observations"])[0]
truth = results["trajectories"][scenario.spacecraft[0].name]
times = results["times_nondim"]
jd = results["jd"]
access = results["observations"][key]["access"]
station = scenario.ground_stations[0]
station_tuple = (station.latitude_deg, station.longitude_deg, station.altitude_km)
every = max(1, int(round(CADENCE_S / scenario.time_step_s)))

position_sigma = 100.0 / crtbp.LENGTH_UNIT_KM
velocity_sigma = crtbp.velocity_to_nondim(1.0 / 1000.0)
initial_covariance = np.diag([position_sigma ** 2] * 3 + [velocity_sigma ** 2] * 3)
rng = np.random.default_rng(2026)


def draw_case(seed):
    measurements = estimation.simulate_measurements(truth, times, jd, station_tuple, "radec", NOISE, mask=access,
                                                    every=every, seed=seed, ephemeris=ephemeris)
    initial_state = truth[0] + rng.multivariate_normal(np.zeros(6), initial_covariance)
    return measurements, initial_state


# ---- batch reference ----
measurements, initial_state = draw_case(1)
print(f"{len(measurements)} measurements over {days:g} days")
started = time.perf_counter()
batch = estimation.batch_least_squares_growing_arc(initial_state, measurements, prior_covariance=initial_covariance * 100.0)
batch_error = crtbp.length_to_km(np.linalg.norm(batch["state"][:3] - truth[0][:3]))
batch_sigma = crtbp.length_to_km(np.sqrt(np.trace(batch["covariance"][:3, :3])))
information = observability.fisher_information(truth[0], measurements, prior_covariance=initial_covariance * 100.0)
_, crlb_position, crlb_velocity = observability.cramer_rao_bound(information)
print(f"batch least squares (growing arc): epoch position error {batch_error:.3f} km, formal {batch_sigma:.3f} km, "
      f"normalised residual rms {batch['rms_normalised_residual']:.3f} (1 if the noise model is right), "
      f"{time.perf_counter() - started:.1f} s")
print(f"Cramer-Rao bound at the epoch: position {crlb_position:.3f} km, velocity {crlb_velocity:.4f} m/s")

# ---- Monte Carlo consistency ----
nees_lower, nees_upper = estimation.chi_squared_bounds(6, n_runs)
nis_lower, nis_upper = estimation.chi_squared_bounds(2, n_runs)
print(f"\nMonte Carlo, {n_runs} runs per case.  Honest: NEES in [{nees_lower:.2f}, {nees_upper:.2f}], "
      f"NIS in [{nis_lower:.2f}, {nis_upper:.2f}]")


def run_filter(filter_name, state, covariance, measurements, level):
    if filter_name == "ekf":
        return estimation.extended_kalman_filter(state, covariance, measurements, level)
    return estimation.unscented_kalman_filter(state, covariance, measurements, level)


def score(result, offset=0.0):
    result = dict(result, times=result["times"] + offset)
    return (estimation.normalised_estimation_error_squared(result, truth, times),
            estimation.normalised_innovation_squared(result),
            estimation.position_errors_km(result, truth, times),
            estimation.formal_position_sigma_km(result),
            crtbp.time_to_days(result["times"]))


# Cold start: the filters from the 100 km, 1 m/s guess, all data.
print("\nCold start from the 100 km, 1 m/s guess (why the warm start is needed), q = 1e-8:")
cold_rows = []
for filter_name in ("ekf", "ukf"):
    nees_runs, final_errors = [], []
    for run in range(min(n_runs, 6)):
        measurements, initial_state = draw_case(100 + run)
        result = run_filter(filter_name, initial_state, initial_covariance, measurements, 1e-8)
        nees, nis, errors, sigmas, _ = score(result)
        nees_runs.append(nees.mean())
        final_errors.append(errors[-1])
    cold_rows.append({"filter": filter_name, "start": "cold", "mean_nees": float(np.mean(nees_runs)),
                      "final_error_km_rms": float(np.sqrt(np.mean(np.square(final_errors))))})
    print(f"  {filter_name}: mean NEES {np.mean(nees_runs):10.1f}, final position error rms {cold_rows[-1]['final_error_km_rms']:8.2f} km")

# Warm start: batch on the first WARM_DAYS, then the filters.
print(f"\nWarm start: batch initial orbit determination on the first {WARM_DAYS:g} days, then the filter")
print(f"{'filter':>6s} {'q [LU/TU^2]':>12s} {'mean NEES':>10s} {'mean NIS':>9s} {'final err km':>13s} {'final 1-sigma km':>17s}")
rows = []
histories = {}
cases = []
for run in range(n_runs):
    measurements, initial_state = draw_case(100 + run)
    warm = [m for m in measurements if crtbp.time_to_days(m["time_nondim"]) <= WARM_DAYS]
    rest = [m for m in measurements if crtbp.time_to_days(m["time_nondim"]) > WARM_DAYS]
    iod = estimation.batch_least_squares_growing_arc(initial_state, warm, prior_covariance=initial_covariance * 100.0,
                                                     stages=3)
    t_hand = warm[-1]["time_nondim"]
    sol = crtbp.propagate_with_stm(iod["state"], t_hand)
    state_hand, phi = crtbp.split_state_and_stm(sol.y[:, -1])
    covariance_hand = phi @ iod["covariance"] @ phi.T
    shifted = [dict(m, time_nondim=m["time_nondim"] - t_hand) for m in rest]
    cases.append((state_hand, covariance_hand, shifted, t_hand))

for filter_name in ("ekf", "ukf"):
    for level in PROCESS_NOISE_LEVELS:
        nees_runs, nis_runs, final_errors, final_sigmas = [], [], [], []
        for run, (state_hand, covariance_hand, shifted, t_hand) in enumerate(cases):
            result = run_filter(filter_name, state_hand, covariance_hand, shifted, level)
            nees, nis, errors, sigmas, days_axis = score(result, t_hand)
            nees_runs.append(nees)
            nis_runs.append(nis)
            final_errors.append(errors[-1])
            final_sigmas.append(sigmas[-1])
            if run == 0:
                histories[(filter_name, level)] = {"times_days": days_axis, "errors": errors, "sigmas": sigmas}
        nees_mean = float(np.mean([n.mean() for n in nees_runs]))
        nis_mean = float(np.mean([n.mean() for n in nis_runs]))
        histories[(filter_name, level)]["nees_time"] = np.mean(np.array(nees_runs), axis=0)
        row = {"filter": filter_name, "start": "warm", "acceleration_sigma": level, "mean_nees": nees_mean,
               "mean_nis": nis_mean, "nees_lower": nees_lower, "nees_upper": nees_upper,
               "nis_lower": nis_lower, "nis_upper": nis_upper,
               "final_error_km_rms": float(np.sqrt(np.mean(np.array(final_errors) ** 2))),
               "final_sigma_km_mean": float(np.mean(final_sigmas))}
        rows.append(row)
        honest = nees_lower <= nees_mean <= nees_upper
        print(f"{filter_name:>6s} {level:12.0e} {nees_mean:10.2f} {nis_mean:9.2f} {row['final_error_km_rms']:13.3f} "
              f"{row['final_sigma_km_mean']:17.3f}{'   <- consistent' if honest else ''}")

csv_path = os.path.join("output", "od_consistency.csv")
with open(csv_path, "w", newline="") as handle:
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows([dict({k: "" for k in fieldnames}, **row) for row in cold_rows])
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {csv_path}")


def tuned_level(filter_name):
    candidates = [row for row in rows if row["filter"] == filter_name]
    inside = [row for row in candidates if nees_lower <= row["mean_nees"] <= nees_upper]
    chosen = min(inside, key=lambda row: row["acceleration_sigma"]) if inside else \
        min(candidates, key=lambda row: abs(np.log(row["mean_nees"] / 6.0)))
    return chosen["acceleration_sigma"]


tuned = {name: tuned_level(name) for name in ("ekf", "ukf")}
print(f"tuned process noise: EKF {tuned['ekf']:.0e}, UKF {tuned['ukf']:.0e} LU/TU^2")

# ---- figure ----
figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
colors = {"ekf": "#2a78d6", "ukf": "#eb6834"}
for name in ("ekf", "ukf"):
    history = histories[(name, tuned[name])]
    axes[0].semilogy(history["times_days"], history["errors"], color=colors[name], lw=1.2,
                     label=f"{name.upper()} error (q = {tuned[name]:.0e})")
    axes[0].semilogy(history["times_days"], 3.0 * history["sigmas"], color=colors[name], lw=0.8, ls="--",
                     label=f"{name.upper()} 3-sigma")
    axes[1].semilogy(history["times_days"], history["nees_time"], color=colors[name], lw=1.2, label=f"{name.upper()} NEES")
axes[0].axhline(crlb_position, color="0.3", lw=0.8, ls=":", label=f"CRLB at epoch {crlb_position:.2f} km")
axes[0].set_ylabel("position error [km]")
axes[0].legend(fontsize=8, ncol=3)
axes[0].set_title(f"Angles-only OD of the 9:2 NRHO from Sydney, {days:g} days, {n_runs}-run Monte Carlo, "
                  f"batch warm start on the first {WARM_DAYS:g} days", fontsize=10)
axes[1].axhspan(nees_lower, nees_upper, color="#43d19a", alpha=0.2, label="95 % consistency band")
axes[1].axhline(6.0, color="0.3", lw=0.8, ls=":")
axes[1].set_ylabel("mean NEES")
axes[1].set_xlabel("days from epoch")
axes[1].legend(fontsize=8)
figure.tight_layout()
figure_path = os.path.join("output", "fig9_orbit_determination.png")
figure.savefig(figure_path, dpi=150)
print(f"saved {figure_path}")
