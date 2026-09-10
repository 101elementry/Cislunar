"""
Rung 1: simulated angles-only observations of one spacecraft from one
telescope over a lunar month, with the JPL DE440 ephemeris for the Sun,
Moon and Earth orientation.

    python scripts/simulate_observations.py [family index] [epoch UTC] [days]

Defaults: L2 southern halo member 49 (the 9:2 NRHO), 2026-01-01, one
synodic month (29.53 days), the Sydney station and telescope of the
example scenario, right ascension and declination every 10 minutes
inside the access windows with 2 arcsecond Gaussian noise.

Outputs
  output/observations.csv          one row per measurement: time, RA,
                                   Dec, truth, noise, geometry
  output/fig7_observation_schedule.png
                                   observation times over the month
                                   against Sun elevation, target
                                   elevation and lunar separation, with
                                   the windows shaded
  printed checks                   analytic Jacobian against central
                                   differences at every measurement, the
                                   window statistics with and without
                                   the ephemeris, and where the schedule
                                   comes from (which constraint bites)
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import crtbp, estimation
from model.scenario import example_scenario
from model.family import load_families
from model.ephemeris import load_ephemeris
from model import runner

family_index = int(sys.argv[1]) if len(sys.argv) > 1 else 49
epoch_utc = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01T00:00:00"
days = float(sys.argv[3]) if len(sys.argv) > 3 else 29.530589

ARCSEC = 1.0 / 3600.0
NOISE_ARCSEC = 2.0
CADENCE_S = 600.0

families = load_families()
ephemeris = load_ephemeris()
if ephemeris is None:
    raise SystemExit("run scripts/fetch_ephemeris.py first: the thesis observations need the real ephemeris")

scenario = example_scenario()
scenario.epoch_utc = epoch_utc
scenario.duration_days = days
scenario.spacecraft[0].family_index = family_index
spacecraft = scenario.spacecraft[0]
station = scenario.ground_stations[0]
station_tuple = (station.latitude_deg, station.longitude_deg, station.altitude_km)
observer_name = runner.observers(scenario)[0][0]

results = runner.run_scenario(scenario, families, ephemeris=ephemeris)
results_simple = runner.run_scenario(scenario, families, ephemeris=None)
key = (observer_name, spacecraft.name)
series = results["observations"][key]["geometry"]
access = results["observations"][key]["access"]
windows = results["windows"][key]

print(f"{spacecraft.name} (member {family_index}) from {station.name}, {days:.2f} days from {epoch_utc}")
print(f"  with DE440:              {len(windows):2d} windows, duty cycle {100 * results['duty_cycle'][key]:.1f}%, "
      f"total {sum(b - a for a, b in windows) / 3600:.1f} h")
windows_simple = results_simple["windows"][key]
print(f"  mean-longitude model:    {len(windows_simple):2d} windows, duty cycle {100 * results_simple['duty_cycle'][key]:.1f}%, "
      f"total {sum(b - a for a, b in windows_simple) / 3600:.1f} h")
kinds = results["observations"][key]["constraint_kinds"]
masks = results["observations"][key]["constraint_masks"]
print("  fraction of the span each constraint alone allows (DE440):")
for kind, column in zip(kinds, masks.T):
    print(f"    {kind:24s} {100 * column.mean():5.1f}%")

# ---- measurements ----
every = max(1, int(round(CADENCE_S / scenario.time_step_s)))
true_states = results["trajectories"][spacecraft.name]
measurements = estimation.simulate_measurements(true_states, results["times_nondim"], results["jd"], station_tuple,
                                                "radec", noise_sigma=[NOISE_ARCSEC * ARCSEC, NOISE_ARCSEC * ARCSEC],
                                                mask=access, every=every, seed=1, ephemeris=ephemeris)
print(f"\n{len(measurements)} RA/Dec measurements at {CADENCE_S / 60:.0f} min cadence, {NOISE_ARCSEC:g} arcsec noise")

# ---- Jacobian check at every measurement ----
worst = 0.0
for measurement in measurements:
    state = true_states[measurement["index"]]
    analytic = measurement["function"].jacobian(state)
    numeric = estimation.measurement_jacobian(measurement["function"], state)
    worst = max(worst, np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic))
print(f"analytic RA/Dec Jacobian vs central difference: worst relative error {worst:.2e} over all measurements")

# ---- CSV ----
rows = []
for measurement in measurements:
    index = measurement["index"]
    truth = measurement["function"](true_states[index])
    rows.append({"time_s": float(results["times_s"][index]),
                 "day": float(results["times_s"][index]) / 86400.0,
                 "jd_utc": float(results["jd"][index]),
                 "ra_deg": float(measurement["value"][0]),
                 "dec_deg": float(measurement["value"][1]),
                 "ra_true_deg": float(truth[0]),
                 "dec_true_deg": float(truth[1]),
                 "elevation_deg": float(series.elevation_deg[index]),
                 "sun_elevation_deg": float(series.sun_elevation_deg[index]),
                 "lunar_separation_deg": float(series.lunar_separation_deg[index]),
                 "range_km": float(series.range_km[index]),
                 "apparent_magnitude": float(series.apparent_magnitude[index])})
csv_path = os.path.join("output", "observations.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {csv_path}")

# ---- figure ----
day = results["times_s"] / 86400.0
obs_days = np.array([row["day"] for row in rows])
figure, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
for ax in axes:
    for start, stop in windows:
        ax.axvspan(start / 86400.0, stop / 86400.0, color="#43d19a", alpha=0.18, lw=0)
axes[0].plot(day, series.elevation_deg, color="#2a78d6", lw=1)
axes[0].axhline(station.min_elevation_deg, color="0.4", ls="--", lw=0.8)
axes[0].set_ylabel("target elevation [deg]")
axes[1].plot(day, series.sun_elevation_deg, color="#eb9b34", lw=1)
axes[1].axhline(station.max_sun_elevation_deg, color="0.4", ls="--", lw=0.8)
axes[1].set_ylabel("Sun elevation [deg]")
axes[2].plot(day, series.lunar_separation_deg, color="#1baf7a", lw=1)
axes[2].axhline(scenario.sensors[0].lunar_exclusion_deg, color="0.4", ls="--", lw=0.8)
axes[2].set_ylabel("lunar separation [deg]")
axes[2].set_xlabel(f"days from {epoch_utc} UTC")
axes[0].plot(obs_days, np.interp(obs_days, day, series.elevation_deg), ".", color="#0b0b0b", ms=3,
             label=f"{len(rows)} RA/Dec measurements")
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title(f"Observation schedule: {spacecraft.name} (member {family_index}) from {station.name}, DE440 ephemeris",
                  fontsize=11)
figure.tight_layout()
figure_path = os.path.join("output", "fig7_observation_schedule.png")
figure.savefig(figure_path, dpi=150)
print(f"saved {figure_path}")
