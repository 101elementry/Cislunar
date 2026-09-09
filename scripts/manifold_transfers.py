"""
Worked example: trace the unstable manifold of a family member and rank
its branches as departure paths by how close they come to the Earth
and to the Moon.  No interface involved.

    python scripts/manifold_transfers.py "L2 southern halo" 0 12 40 output/manifold_transfers.csv

Arguments: family name, member index, number of departure points,
duration in days, output CSV.  A branch that reaches the Earth's
vicinity for free is a candidate return path; the stable manifold of
the same orbit (run with the engine's kind = "stable") is the matching
arrival path.  The NRHO members have small unstable eigenvalues, so
their branches take many revolutions to leave; use a long duration.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import crtbp, manifolds
from model.family import load_families

family_name = sys.argv[1] if len(sys.argv) > 1 else "L2 southern halo"
member = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n_branches = int(sys.argv[3]) if len(sys.argv) > 3 else 12
duration_days = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
csv_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join("output", "manifold_transfers.csv")

orbit = load_families()[family_name][member]
duration = crtbp.time_to_nondim(duration_days * crtbp.SECONDS_PER_DAY)
print(f"{family_name} member {member}: period {crtbp.time_to_days(orbit['period']):.3f} d, "
      f"stability index {orbit['stability_index']:.2f}, unstable eigenvalue "
      f"{max(abs(v) for v in orbit['eigenvalues']):.3f} per revolution")

rows = []
print(f"\n{'kind':>8s} {'depart d':>9s} {'sign':>4s} {'Earth min km':>13s} {'at day':>7s} {'Moon min km':>12s} {'impact':>7s}")
for kind in ("unstable", "stable"):
    branches = manifolds.manifold_branches(orbit, kind, n_branches=n_branches, duration=duration)
    earth_distance, earth_time = manifolds.closest_approach_to_body(branches, crtbp.earth_position())
    moon_distance, _ = manifolds.closest_approach_to_body(branches, crtbp.moon_position())
    for branch, d_earth, t_earth, d_moon in zip(branches, earth_distance, earth_time, moon_distance):
        row = {"kind": kind,
               "departure_day": crtbp.time_to_days(branch["departure_time"]),
               "sign": int(branch["sign"]),
               "earth_closest_km": crtbp.length_to_km(d_earth),
               "earth_closest_day": crtbp.time_to_days(t_earth),
               "moon_closest_km": crtbp.length_to_km(d_moon),
               "impact": branch["impact"] or ""}
        rows.append(row)
        print(f"{kind:>8s} {row['departure_day']:9.3f} {row['sign']:+4d} {row['earth_closest_km']:13.0f} "
              f"{row['earth_closest_day']:7.1f} {row['moon_closest_km']:12.0f} {row['impact']:>7s}")

with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {len(rows)} rows to {csv_path}")
