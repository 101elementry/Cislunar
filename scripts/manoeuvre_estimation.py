"""
Rung 4 (stretch): estimate the size and epoch of a burn once it has
been detected, by adding it to the batch solve.

    python scripts/manoeuvre_estimation.py [burn m/s] [location]

Defaults: 0.2 m/s along track at the first apolune (day 3.28, moved
into the gap between nights if needed), member 49, Sydney, 14 days,
2 arcsecond RA/Dec every 10 min, the DE440 ephemeris.

The batch solve gains three unknowns, the burn vector at a candidate
epoch, whose measurement partials are H Phi(t_k, t_b)[:, 3:6] for
measurements after the burn.  The epoch is not a linear parameter, so
the solve is repeated on a grid of candidate epochs spanning the gap
between the last night before and the first night after the burn, and
the cost curve's minimum gives the epoch estimate.  The width of that
minimum is the epoch's uncertainty: a burn between two nights can only
be timed to within the gap by measurements outside it, except through
the small change in the trajectory that timing makes.

Outputs: output/manoeuvre_estimation.csv (the cost curve) and
output/fig11_burn_estimation.png.
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

from engine import crtbp, detection, estimation
from model.scenario import example_scenario
from model.family import load_families
from model.ephemeris import load_ephemeris
from model import runner

burn_m_s = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
location = sys.argv[2] if len(sys.argv) > 2 else "apolune"
ARCSEC = 1.0 / 3600.0
NOISE = [2.0 * ARCSEC, 2.0 * ARCSEC]
CADENCE_S = 600.0

families = load_families()
ephemeris = load_ephemeris()
if ephemeris is None:
    raise SystemExit("run scripts/fetch_ephemeris.py first")

scenario = example_scenario()
scenario.duration_days = 14.0
results = runner.run_scenario(scenario, families, ephemeris=ephemeris)
key = list(results["observations"])[0]
truth = results["trajectories"][scenario.spacecraft[0].name]
times = results["times_nondim"]
jd = results["jd"]
access = results["observations"][key]["access"]
station = scenario.ground_stations[0]
station_tuple = (station.latitude_deg, station.longitude_deg, station.altitude_km)
every = max(1, int(round(CADENCE_S / scenario.time_step_s)))
orbit = families["L2 southern halo"][49]

nominal = estimation.simulate_measurements(truth, times, jd, station_tuple, "radec", NOISE, mask=access,
                                           every=every, seed=0, ephemeris=ephemeris)
nights = detection.group_into_nights(nominal)
night_bounds = [(nominal[i[0]]["time_nondim"], nominal[i[-1]]["time_nondim"]) for i in nights]
t_burn = orbit["period"] if location == "perilune" else 0.5 * orbit["period"]
for start, stop in night_bounds:
    if start <= t_burn <= stop:
        t_burn = stop + 0.01
before = max(stop for _, stop in night_bounds if stop < t_burn)
after = min(start for start, _ in night_bounds if start > t_burn)
print(f"burn of {burn_m_s:g} m/s along track at day {crtbp.time_to_days(t_burn):.3f} ({location}); "
      f"gap from day {crtbp.time_to_days(before):.3f} to {crtbp.time_to_days(after):.3f}")

state_at_burn = crtbp.propagate(truth[0], t_burn).y[:, -1]
direction = detection.velocity_direction(state_at_burn, "along_track")
delta_v_true = crtbp.velocity_to_nondim(burn_m_s / 1000.0) * direction
truth_burn = detection.trajectory_with_burn(truth[0], times, t_burn, delta_v_true)
measurements = estimation.simulate_measurements(truth_burn, times, jd, station_tuple, "radec", NOISE, mask=access,
                                                every=every, seed=7, ephemeris=ephemeris)

position_sigma = 10.0 / crtbp.LENGTH_UNIT_KM
velocity_sigma = crtbp.velocity_to_nondim(0.1 / 1000.0)
prior = np.diag([position_sigma ** 2] * 3 + [velocity_sigma ** 2] * 3)
rng = np.random.default_rng(3)
initial_state = truth[0] + rng.multivariate_normal(np.zeros(6), prior)

# Without a burn in the model: the residuals must show the burn.
plain = estimation.batch_least_squares_growing_arc(initial_state, measurements, prior_covariance=prior)
print(f"batch without a burn: normalised residual rms {plain['rms_normalised_residual']:.2f} (1 would mean no burn)")

# With a burn at candidate epochs across the gap.
candidates = np.linspace(before + 0.005, after - 0.005, 11)
started = time.perf_counter()
best_time, best, costs = detection.estimate_burn_epoch(initial_state, measurements, candidates, prior_covariance=prior)
elapsed = time.perf_counter() - started
true_m_s = crtbp.velocity_to_km_s(delta_v_true) * 1000.0
print(f"\nepoch estimate day {crtbp.time_to_days(best_time):.3f} (true {crtbp.time_to_days(t_burn):.3f}), "
      f"{elapsed:.0f} s for {len(candidates)} candidates")
print(f"burn estimate {np.round(best['delta_v_m_s'], 4)} m/s, one-sigma {np.round(best['delta_v_sigma_m_s'], 4)}")
print(f"burn truth    {np.round(true_m_s, 4)} m/s; |estimate| {np.linalg.norm(best['delta_v_m_s']):.4f}, "
      f"|truth| {burn_m_s:.4f} m/s")
print(f"normalised residual rms with the burn modelled: {best['rms_normalised_residual']:.3f}")

rows = [{"candidate_day": crtbp.time_to_days(t), "cost": c} for t, c in zip(candidates, costs)]
csv_path = os.path.join("output", "manoeuvre_estimation.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

figure, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(crtbp.time_to_days(candidates), costs, "o-", color="#2a78d6")
ax.axvline(crtbp.time_to_days(t_burn), color="#eb6834", ls="--", lw=1, label="true burn epoch")
ax.axvline(crtbp.time_to_days(best_time), color="#1baf7a", ls=":", lw=1.2, label="estimated epoch")
ax.set_xlabel("candidate burn epoch [days from epoch]")
ax.set_ylabel("weighted least-squares cost")
ax.set_title(f"Profile of the batch cost over the burn epoch ({burn_m_s:g} m/s at {location})", fontsize=10)
ax.legend(fontsize=8)
figure.tight_layout()
figure_path = os.path.join("output", "fig11_burn_estimation.png")
figure.savefig(figure_path, dpi=150)
print(f"wrote {csv_path}, saved {figure_path}")
