"""
Worked example: orbit determination of a spacecraft from ground-based
angle measurements with an extended Kalman filter on the CRTBP
dynamics.  No interface involved.

    python scripts/orbit_determination.py output/example_scenario.json output/orbit_determination.csv angles

The first spacecraft and the first ground station of the scenario are
used.  Angle measurements (azimuth and elevation, 2 arcsecond noise)
are taken every 10 minutes while the station has access according to
the scenario's own constraints, so the filter only sees the spacecraft
when a telescope could.  With the third argument "angles+range" a
range and range-rate measurement (10 m, 1 mm/s) is added at the same
times, as a transponder would give.  The filter starts 100 km and
1 m/s off the truth with a matching covariance and its position error
and formal one-sigma are tabulated against time.

What to look at: the error falls during a pass and grows between
passes.  With angles alone from one station the distance along the
line of sight is only weakly observed, so the true error settles above
the filter's formal sigma (the filter is optimistic); adding range
closes that gap.  Both effects are expected and worth explaining.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import crtbp, estimation
from model.scenario import Scenario
from model.family import load_families
from model import runner

scenario_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "example_scenario.json")
csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "orbit_determination.csv")
measurement_set = sys.argv[3] if len(sys.argv) > 3 else "angles"

scenario = Scenario.load(scenario_path)
families = load_families()
results = runner.run_scenario(scenario, families)

spacecraft = scenario.spacecraft[0]
station = scenario.ground_stations[0]
station_tuple = (station.latitude_deg, station.longitude_deg, station.altitude_km)
observer_name = runner.observers(scenario)[0][0]

true_states = results["trajectories"][spacecraft.name]
times = results["times_nondim"]
jd = results["jd"]
access = results["observations"][(observer_name, spacecraft.name)]["access"]

# Angle measurements every 10 minutes inside the access windows.
every = max(1, int(round(600.0 / scenario.time_step_s)))
arcsec_deg = 1.0 / 3600.0
measurements = estimation.simulate_measurements(true_states, times, jd, station_tuple, "angles",
                                                noise_sigma=[2.0 * arcsec_deg, 2.0 * arcsec_deg],
                                                mask=access, every=every)
if measurement_set == "angles+range":
    ranges = estimation.simulate_measurements(true_states, times, jd, station_tuple, "range",
                                              noise_sigma=[0.010, 1.0e-6], mask=access, every=every, seed=2)
    measurements = sorted(measurements + ranges, key=lambda m: (m["time_nondim"], m["function"].wraps_at_360))
print(f"{spacecraft.name} from {station.name}: {len(measurements)} measurements ({measurement_set}) in "
      f"{len(results['windows'][(observer_name, spacecraft.name)])} passes over {scenario.duration_days:g} days")

# Initial estimate: 100 km and 1 m/s off in every axis, covariance to match.
rng = np.random.default_rng(1)
position_sigma = 100.0 / crtbp.LENGTH_UNIT_KM
velocity_sigma = crtbp.velocity_to_nondim(1.0 / 1000.0)
initial_state = true_states[0].copy()
initial_state[:3] = initial_state[:3] + rng.normal(0.0, position_sigma, 3)
initial_state[3:] = initial_state[3:] + rng.normal(0.0, velocity_sigma, 3)
initial_covariance = np.diag([position_sigma ** 2] * 3 + [velocity_sigma ** 2] * 3)

result = estimation.extended_kalman_filter(initial_state, initial_covariance, measurements,
                                           acceleration_sigma=3e-7)
errors = estimation.position_errors_km(result, true_states, times)
sigmas = estimation.formal_position_sigma_km(result)

rows = []
print(f"\n{'day':>7s} {'pos error km':>13s} {'formal 1-sigma km':>18s} {'innovation 1':>16s} {'innovation 2':>16s}")
print("(innovations in arcseconds for angle measurements; range rows are in km and km/s times 3600)")
for k, t in enumerate(result["times"]):
    row = {"day": crtbp.time_to_days(t),
           "position_error_km": errors[k],
           "formal_sigma_km": sigmas[k],
           "innovation_1": float(result["residuals"][k][0]),
           "innovation_2": float(result["residuals"][k][1])}
    rows.append(row)
    if k % max(1, len(result["times"]) // 25) == 0 or k == len(result["times"]) - 1:
        print(f"{row['day']:7.3f} {errors[k]:13.2f} {sigmas[k]:18.2f} "
              f"{row['innovation_1'] * 3600.0:16.2f} {row['innovation_2'] * 3600.0:16.2f}")

print(f"\nfinal position error {errors[-1]:.2f} km, formal one-sigma {sigmas[-1]:.2f} km")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {csv_path}")
