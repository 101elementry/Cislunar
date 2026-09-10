"""
Rung 2, observability: the best possible position knowledge from one
lunar month of angles from Sydney, along the orbit and across the
family, from the Fisher information (Cramer-Rao bound), plotted next
to the stability index.

    python scripts/observability_sweep.py [stride] [days]

For every stride-th member of the L2 southern halo family the access
windows are computed with the ephemeris, RA/Dec measurements at 10 min
cadence and 2 arcsecond noise are laid out inside them, and the Fisher
information about the epoch state is accumulated with the STM.  A weak
prior (1,000 km, 10 m/s) regularises the range direction.  The bound is
also traced along the NRHO by moving the epoch around the orbit with
the same observing schedule.

Outputs: output/observability.csv and output/fig8_observability.png.
"""

import csv
import os
import sys

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

stride = int(sys.argv[1]) if len(sys.argv) > 1 else 4
days = float(sys.argv[2]) if len(sys.argv) > 2 else 29.530589
ARCSEC = 1.0 / 3600.0
NOISE = [2.0 * ARCSEC, 2.0 * ARCSEC]
CADENCE_S = 600.0

families = load_families()
family = families["L2 southern halo"]
ephemeris = load_ephemeris()
if ephemeris is None:
    raise SystemExit("run scripts/fetch_ephemeris.py first")

prior_position = 1000.0 / crtbp.LENGTH_UNIT_KM
prior_velocity = crtbp.velocity_to_nondim(10.0 / 1000.0)
prior = np.diag([prior_position ** 2] * 3 + [prior_velocity ** 2] * 3)


def schedule_for(scenario):
    """Measurements inside this scenario's windows, and the truth trajectory."""
    results = runner.run_scenario(scenario, families, ephemeris=ephemeris)
    key = list(results["observations"])[0]
    truth = results["trajectories"][scenario.spacecraft[0].name]
    every = max(1, int(round(CADENCE_S / scenario.time_step_s)))
    measurements = estimation.simulate_measurements(truth, results["times_nondim"], results["jd"],
                                                    (scenario.ground_stations[0].latitude_deg,
                                                     scenario.ground_stations[0].longitude_deg,
                                                     scenario.ground_stations[0].altitude_km),
                                                    "radec", NOISE, mask=results["observations"][key]["access"],
                                                    every=every, ephemeris=ephemeris)
    return truth, measurements, results


# ---- across the family ----
rows = []
print(f"{'index':>5s} {'perilune km':>12s} {'nu':>8s} {'measurements':>12s} {'nights':>6s} {'CRLB pos km':>12s} {'CRLB vel m/s':>13s}")
for index in range(0, len(family), stride):
    scenario = example_scenario()
    scenario.duration_days = days
    scenario.time_step_s = 120.0
    scenario.spacecraft[0].family_index = index
    truth, measurements, results = schedule_for(scenario)
    information = observability.fisher_information(truth[0], measurements, prior_covariance=prior)
    _, position_bound, velocity_bound = observability.cramer_rao_bound(information)
    orbit = family[index]
    row = {"family_index": index,
           "perilune_km": crtbp.length_to_km(orbit["perilune_radius"]),
           "period_days": crtbp.time_to_days(orbit["period"]),
           "stability_index": orbit["stability_index"],
           "n_measurements": len(measurements),
           "n_windows": len(results["windows"][list(results["windows"])[0]]),
           "crlb_position_km": position_bound,
           "crlb_velocity_m_s": velocity_bound}
    rows.append(row)
    print(f"{index:5d} {row['perilune_km']:12.0f} {row['stability_index']:8.2f} {len(measurements):12d} "
          f"{row['n_windows']:6d} {position_bound:12.3f} {velocity_bound:13.4f}")

csv_path = os.path.join("output", "observability.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {csv_path}")

# ---- along the NRHO: move the epoch around the orbit, same schedule ----
scenario = example_scenario()
scenario.duration_days = days
scenario.time_step_s = 120.0
scenario.spacecraft[0].family_index = 49
truth, measurements, results = schedule_for(scenario)
orbit = family[49]
phases = np.linspace(0.0, 1.0, 13)[:-1]
bounds_along = np.zeros(len(phases))
for k, phase in enumerate(phases):
    # Same measurement times and geometry functions; only the epoch
    # state moves around the orbit.  The measurement functions carry
    # their own dates, so the sky is unchanged.
    state0 = crtbp.propagate(orbit["state0"], phase * orbit["period"]).y[:, -1] if phase > 0.0 else orbit["state0"]
    information = observability.fisher_information(state0, measurements, prior_covariance=prior)
    _, bounds_along[k], _ = observability.cramer_rao_bound(information)
print("\nNRHO position bound against epoch phase (0 = perilune crossing):")
for phase, bound in zip(phases, bounds_along):
    print(f"  phase {phase:4.2f}: {bound:8.3f} km")

# ---- figure ----
perilune = np.array([row["perilune_km"] for row in rows])
figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
ax.semilogy(perilune, [row["crlb_position_km"] for row in rows], "o-", color="#2a78d6", label="CRLB position [km]")
ax.set_xlabel("perilune radius [km]")
ax.set_ylabel("position bound after one month [km]", color="#2a78d6")
ax.invert_xaxis()
ax2 = ax.twinx()
ax2.semilogy(perilune, [row["stability_index"] for row in rows], "s--", color="#eb6834", label="stability index")
ax2.set_ylabel("stability index", color="#eb6834")
ax.set_title("Observability across the L2 southern halo family (Sydney, angles only)", fontsize=10)
ax = axes[1]
ax.plot(phases, bounds_along, "o-", color="#1baf7a")
ax.set_xlabel("epoch phase along the NRHO (0 = perilune)")
ax.set_ylabel("position bound [km]")
ax.set_title("Bound against where the epoch sits on the 9:2 NRHO", fontsize=10)
figure.tight_layout()
figure_path = os.path.join("output", "fig8_observability.png")
figure.savefig(figure_path, dpi=150)
print(f"saved {figure_path}")
