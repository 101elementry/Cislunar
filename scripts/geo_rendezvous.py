"""
Worked example: a chaser inserted into a GEO parking orbit slightly
below and behind a target in GEO, then a two-impulse rendezvous.  The
relative motion is reported in the target's LVLH frame before and
during the transfer.  No interface involved.

    python scripts/geo_rendezvous.py 6 output/geo_rendezvous.csv

Arguments: transfer time in hours, output CSV.  Both spacecraft are
defined by Earth-centred elements against the equator at the scenario
epoch, converted with model.orbits exactly as the interface does, and
integrated in the CRTBP (so the Moon's pull is included).  The
transfer-time sweep at the end shows how the cost varies with the
allowed time.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import crtbp, frames, rendezvous
from model import orbits
from model.scenario import Spacecraft, ELEMENT_PRESETS

transfer_hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "geo_rendezvous.csv")

epoch_jd = frames.julian_date("2026-01-01T00:00:00")
centre, plane, geo = ELEMENT_PRESETS["Geostationary orbit"]

target = Spacecraft(name="GEO target", source="elements", centre=centre, reference_plane=plane,
                    elements=dict(geo))
# Chaser: 50 km lower (so it drifts forward), 2 degrees behind.
chaser_elements = dict(geo)
chaser_elements["a_km"] = geo["a_km"] - 50.0
chaser_elements["true_anomaly_deg"] = geo["true_anomaly_deg"] - 2.0
chaser = Spacecraft(name="chaser", source="elements", centre=centre, reference_plane=plane,
                    elements=chaser_elements)

target_state = orbits.initial_state(target, {}, epoch_jd)
chaser_state = orbits.initial_state(chaser, {}, epoch_jd)

# Relative motion during one day of free drift.
drift_days = 1.0
times = crtbp.time_to_nondim(np.arange(0.0, drift_days * crtbp.SECONDS_PER_DAY + 1.0, 300.0))
target_drift = crtbp.propagate(target_state, times[-1], t_eval=times).y.T
chaser_drift = crtbp.propagate(chaser_state, times[-1], t_eval=times).y.T
position_km, velocity_m_s = rendezvous.relative_motion_lvlh(target_drift, chaser_drift, times, centre="earth")
print("Free drift, chaser relative to target in LVLH (radial, along-track, cross-track):")
print(f"  start  {position_km[0].round(1)} km   {velocity_m_s[0].round(3)} m/s")
print(f"  +1 day {position_km[-1].round(1)} km   {velocity_m_s[-1].round(3)} m/s")
print(f"  a 50 km lower orbit drifts ahead by about {position_km[-1, 1] - position_km[0, 1]:.0f} km per day")

# Two-impulse rendezvous from the initial state.
transfer_time = crtbp.time_to_nondim(transfer_hours * 3600.0)
solution = rendezvous.two_impulse_rendezvous(chaser_state, target_state, transfer_time)
print(f"\nTwo-impulse rendezvous in {transfer_hours:g} h:")
print(f"  burn 1 {np.linalg.norm(solution['delta_v1_m_s']):.3f} m/s, burn 2 {np.linalg.norm(solution['delta_v2_m_s']):.3f} m/s, "
      f"total {solution['total_delta_v_m_s']:.3f} m/s, arrival miss {solution['arrival_position_error_km'] * 1000:.1f} m")

relative_km, relative_m_s = rendezvous.relative_motion_lvlh(solution["target_states"], solution["transfer_states"],
                                                            solution["transfer_times"], centre="earth")
rows = []
for k, t in enumerate(solution["transfer_times"]):
    rows.append({"hours": crtbp.time_to_seconds(t) / 3600.0,
                 "radial_km": relative_km[k, 0], "along_track_km": relative_km[k, 1], "cross_track_km": relative_km[k, 2],
                 "radial_m_s": relative_m_s[k, 0], "along_track_m_s": relative_m_s[k, 1],
                 "cross_track_m_s": relative_m_s[k, 2]})
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"  wrote {len(rows)} rows of relative motion to {csv_path}")

# Cost against transfer time.
sweep_hours = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
costs = rendezvous.transfer_time_sweep(chaser_state, target_state, crtbp.time_to_nondim(sweep_hours * 3600.0))
print("\nTotal delta-v against transfer time:")
for hours, cost in zip(sweep_hours, costs):
    print(f"  {hours:5.1f} h  {cost:8.3f} m/s")
