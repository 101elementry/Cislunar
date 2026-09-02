"""
Worked example: run the same station and sensor against every N-th
member of the halo family and tabulate how observability changes along
the family.  No interface involved.

    python scripts/compare_family_members.py output/example_scenario.json output/family_comparison.csv 5

The third argument is the stride through the family (default 5).  The
first spacecraft in the scenario is used as the template: its diameter,
albedo and propagation mode are kept, only the family index changes.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import crtbp
from model.scenario import Scenario
from model.family import load_family
from model import runner

scenario_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "example_scenario.json")
csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "family_comparison.csv")
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 5

scenario = Scenario.load(scenario_path)
family = load_family()

template = scenario.spacecraft[0]
template.source = "family"
scenario.spacecraft = [template]

rows = []
print(f"{'index':>5s} {'period d':>9s} {'perilune km':>12s} {'apolune km':>11s} {'nu':>7s} "
      f"{'duty %':>7s} {'windows':>8s} {'mag min':>8s} {'mag max':>8s}")
for index in range(0, len(family), stride):
    template.family_index = index
    template.name = f"Halo #{index}"
    orbit = family[index]
    results = runner.run_scenario(scenario, family)
    for (observer, spacecraft_name), windows in results["windows"].items():
        series = results["observations"][(observer, spacecraft_name)]["geometry"]
        duty = results["duty_cycle"][(observer, spacecraft_name)]
        row = {"family_index": index,
               "observer": observer,
               "period_days": crtbp.time_to_days(orbit["period"]),
               "perilune_km": crtbp.length_to_km(orbit["perilune_radius"]),
               "apolune_km": crtbp.length_to_km(orbit["apolune_radius"]),
               "jacobi": orbit["jacobi"],
               "stability_index": orbit["stability_index"],
               "duty_cycle": duty,
               "n_windows": len(windows),
               "magnitude_min": float(series.apparent_magnitude.min()),
               "magnitude_max": float(series.apparent_magnitude.max()),
               "lunar_separation_max_deg": float(series.lunar_separation_deg.max())}
        rows.append(row)
        print(f"{index:5d} {row['period_days']:9.2f} {row['perilune_km']:12.0f} {row['apolune_km']:11.0f} "
              f"{row['stability_index']:7.2f} {100 * duty:7.1f} {len(windows):8d} "
              f"{row['magnitude_min']:8.2f} {row['magnitude_max']:8.2f}")

with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {len(rows)} rows to {csv_path}")
