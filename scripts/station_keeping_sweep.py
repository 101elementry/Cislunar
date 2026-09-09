"""
Worked example: station-keeping cost along a family.  Every N-th member
is flown for a number of revolutions with impulsive targeting and the
delta-v per year is tabulated against the stability index.  No
interface involved.

    python scripts/station_keeping_sweep.py "L2 southern halo" 5 8 2 output/station_keeping.csv

Arguments: family name, stride through the family, revolutions flown,
manoeuvres per revolution, output CSV.  The navigation and execution
errors are the defaults of engine.stationkeeping.simulate (1 km, 1 cm/s
navigation; 1 % execution).  Classical halos with stability indices in
the hundreds need several manoeuvres per revolution to stay bounded;
the NRHOs manage with one.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import crtbp, stationkeeping
from model.family import load_families

family_name = sys.argv[1] if len(sys.argv) > 1 else "L2 southern halo"
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 5
n_revolutions = int(sys.argv[3]) if len(sys.argv) > 3 else 8
nodes_per_revolution = int(sys.argv[4]) if len(sys.argv) > 4 else 2
csv_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join("output", "station_keeping.csv")

family = load_families()[family_name]

rows = []
print(f"{'index':>5s} {'period d':>9s} {'perilune km':>12s} {'nu':>8s} {'dv/yr m/s':>10s} {'max err km':>11s} "
      f"{'largest dv m/s':>15s}")
for index in range(0, len(family), stride):
    orbit = family[index]
    result = stationkeeping.simulate(orbit, n_revolutions=n_revolutions, nodes_per_revolution=nodes_per_revolution)
    row = {"family_index": index,
           "period_days": crtbp.time_to_days(orbit["period"]),
           "perilune_km": crtbp.length_to_km(orbit["perilune_radius"]),
           "stability_index": orbit["stability_index"],
           "nodes_per_revolution": nodes_per_revolution,
           "delta_v_per_year_m_s": result["delta_v_per_year_m_s"],
           "max_position_error_km": float(result["position_error_km"].max()),
           "largest_manoeuvre_m_s": float(result["delta_v_magnitude_m_s"].max())}
    rows.append(row)
    print(f"{index:5d} {row['period_days']:9.2f} {row['perilune_km']:12.0f} {row['stability_index']:8.2f} "
          f"{row['delta_v_per_year_m_s']:10.2f} {row['max_position_error_km']:11.1f} {row['largest_manoeuvre_m_s']:15.3f}")

with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {len(rows)} rows to {csv_path}")
