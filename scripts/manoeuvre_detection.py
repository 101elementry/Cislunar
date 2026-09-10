"""
Rung 3: the smallest impulsive burn detectable between observing
nights, as a function of burn size, burn location on the orbit and
the gap before the next observation.

    python scripts/manoeuvre_detection.py [filter] [process noise] [trials]

Defaults: the UKF with the process noise tuned in rung 2 (1e-7
LU/TU^2), 8 Monte Carlo trials per case.  One object (member 49), the
Sydney telescope, RA/Dec at 10 min cadence and 2 arcsecond noise
inside the access windows, 21 days from 2026-01-01.

The filter is warmed up on the nights before the burn from a 10 km,
0.1 m/s initial error (an object already in custody).  The burn is
along the rotating-frame velocity (the cheap way to change the
period), at the perilune crossing nearest day 6.5 or the apolune
nearest day 3.3 or 9.8, moved to the nearest gap between nights if
it falls inside one.  The "gap" is the number of nights skipped after
the burn before observing resumes: 1 means the very next night.  A
night is declared to show a manoeuvre when its normalised innovation
sum exceeds the chi-squared quantile at a 1e-3 false alarm
probability (engine/detection.py).

The zero-burn case is run too, so the false alarm rate the detector
actually delivers is measured, not assumed.

Outputs: output/manoeuvre_detection.csv and
output/fig10_minimum_detectable_burn.png.
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

filter_name = sys.argv[1] if len(sys.argv) > 1 else "ukf"
acceleration_sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-7
n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 8

ARCSEC = 1.0 / 3600.0
NOISE = [2.0 * ARCSEC, 2.0 * ARCSEC]
CADENCE_S = 600.0
BURN_SIZES_M_S = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
GAPS = [1, 2, 4]
FALSE_ALARM = 1e-3
TARGET_PROBABILITY = 0.9

families = load_families()
ephemeris = load_ephemeris()
if ephemeris is None:
    raise SystemExit("run scripts/fetch_ephemeris.py first")

scenario = example_scenario()
scenario.duration_days = 21.0
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
period = orbit["period"]

position_sigma = 10.0 / crtbp.LENGTH_UNIT_KM
velocity_sigma = crtbp.velocity_to_nondim(0.1 / 1000.0)
initial_covariance = np.diag([position_sigma ** 2] * 3 + [velocity_sigma ** 2] * 3)

# Nights of the nominal schedule, to place burns in gaps and to skip nights.
nominal = estimation.simulate_measurements(truth, times, jd, station_tuple, "radec", NOISE, mask=access,
                                           every=every, seed=0, ephemeris=ephemeris)
nights = detection.group_into_nights(nominal)
night_bounds = [(nominal[indices[0]]["time_nondim"], nominal[indices[-1]]["time_nondim"]) for indices in nights]


def move_into_gap(t):
    """The burn epoch, shifted to the nearest gap if it falls inside a night."""
    for start, stop in night_bounds:
        if start <= t <= stop:
            return stop + 0.01
    return t


burn_epochs = {"perilune": move_into_gap(period),           # first perilune crossing after the epoch
               "apolune": move_into_gap(0.5 * period)}       # first apolune
for name, t in burn_epochs.items():
    print(f"{name} burn at day {crtbp.time_to_days(t):.3f}")


def mask_skipping_nights(t_burn, gap):
    """Access mask with the first gap-1 nights after the burn removed."""
    mask = access.copy()
    after = [k for k, (start, _) in enumerate(night_bounds) if start > t_burn]
    for k in after[:gap - 1]:
        indices = nights[k]
        for i in indices:
            mask[nominal[i]["index"]] = False
        # Also blank the whole span of that night on the grid.
        start, stop = night_bounds[k]
        mask[(times >= start - 1e-9) & (times <= stop + 1e-9)] = False
    return mask


rows = []
print(f"\n{filter_name.upper()}, q = {acceleration_sigma:.0e}, {n_trials} trials, false alarm {FALSE_ALARM:g}")
print(f"{'location':>9s} {'gap':>4s} {'burn m/s':>9s} {'P(detect)':>10s} {'s':>6s}")
for location, t_burn in burn_epochs.items():
    for gap in GAPS:
        mask = mask_skipping_nights(t_burn, gap)
        for size in BURN_SIZES_M_S:
            started = time.perf_counter()
            probability, flags, night = detection.detection_probability(
                truth[0], times, jd, station_tuple, mask, t_burn, size, "along_track", NOISE, every,
                initial_covariance, acceleration_sigma, n_trials=n_trials, filter_name=filter_name,
                false_alarm_probability=FALSE_ALARM, seed=int(1000 * size + 10 * gap + (location == "apolune")),
                ephemeris=ephemeris)
            rows.append({"location": location, "gap_nights": gap, "burn_m_s": size,
                         "detection_probability": probability, "n_trials": n_trials, "filter": filter_name,
                         "acceleration_sigma": acceleration_sigma})
            print(f"{location:>9s} {gap:4d} {size:9.3f} {probability:10.2f} {time.perf_counter() - started:6.1f}")

csv_path = os.path.join("output", "manoeuvre_detection.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {csv_path}")


def minimum_detectable(location, gap):
    """Smallest burn whose detection probability reaches the target, by interpolation in log size."""
    cases = sorted([row for row in rows if row["location"] == location and row["gap_nights"] == gap and row["burn_m_s"] > 0.0],
                   key=lambda row: row["burn_m_s"])
    sizes = np.array([row["burn_m_s"] for row in cases])
    probabilities = np.array([row["detection_probability"] for row in cases])
    above = np.where(probabilities >= TARGET_PROBABILITY)[0]
    if len(above) == 0:
        return np.nan
    k = above[0]
    if k == 0:
        return sizes[0]
    # Log-linear interpolation between the last miss and the first hit.
    fraction = (TARGET_PROBABILITY - probabilities[k - 1]) / max(probabilities[k] - probabilities[k - 1], 1e-9)
    return float(np.exp(np.log(sizes[k - 1]) + fraction * (np.log(sizes[k]) - np.log(sizes[k - 1]))))


print(f"\nSmallest burn detected with probability >= {TARGET_PROBABILITY:g}:")
summary = {}
for location in burn_epochs:
    for gap in GAPS:
        summary[(location, gap)] = minimum_detectable(location, gap)
        print(f"  {location:>9s}, resume after {gap} night(s): {summary[(location, gap)]:.3f} m/s")
false_alarms = [row["detection_probability"] for row in rows if row["burn_m_s"] == 0.0]
print(f"false alarm rate measured with no burn: {np.mean(false_alarms):.3f} (design {FALSE_ALARM:g})")

# ---- figure ----
figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for location, color in (("perilune", "#eb6834"), ("apolune", "#2a78d6")):
    for gap, style in zip(GAPS, ("-", "--", ":")):
        cases = sorted([row for row in rows if row["location"] == location and row["gap_nights"] == gap and row["burn_m_s"] > 0.0],
                       key=lambda row: row["burn_m_s"])
        axes[0].semilogx([row["burn_m_s"] for row in cases], [row["detection_probability"] for row in cases],
                         style, marker="o", color=color, ms=4, label=f"{location}, resume after {gap}")
    axes[1].semilogy(GAPS, [summary[(location, gap)] for gap in GAPS], "o-", color=color, label=location)
axes[0].axhline(TARGET_PROBABILITY, color="0.4", lw=0.8, ls="--")
axes[0].set_xlabel("burn size [m/s]")
axes[0].set_ylabel("detection probability")
axes[0].legend(fontsize=7, ncol=2)
axes[0].set_title(f"Detection of an along-track burn, {filter_name.upper()}, false alarm {FALSE_ALARM:g}", fontsize=10)
axes[1].set_xlabel("nights skipped before observing resumes")
axes[1].set_ylabel(f"smallest burn detected with P >= {TARGET_PROBABILITY:g} [m/s]")
axes[1].set_xticks(GAPS)
axes[1].legend(fontsize=8)
axes[1].set_title("Minimum detectable burn", fontsize=10)
figure.tight_layout()
figure_path = os.path.join("output", "fig10_minimum_detectable_burn.png")
figure.savefig(figure_path, dpi=150)
print(f"saved {figure_path}")
