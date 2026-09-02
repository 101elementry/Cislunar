"""
Worked example with no interface involved: load a scenario from JSON,
sweep the minimum-elevation cutoff of every ground station, and write
the resulting duty cycles to a CSV file.

    python scripts/sweep_min_elevation.py output/example_scenario.json output/elevation_sweep.csv

Only model/ and engine/ are imported.  Everything the interface does is
available here in exactly the same form, which is the point of the
layering.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.scenario import Scenario
from model.family import load_family
from model import runner

scenario_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "example_scenario.json")
csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "elevation_sweep.csv")

scenario = Scenario.load(scenario_path)
family = load_family()

elevation_cutoffs_deg = np.arange(0.0, 61.0, 5.0)

rows = []
print(f"{'min elev':>9s} {'observer -> spacecraft':<48s} {'duty %':>7s} {'windows':>8s} {'longest h':>10s}")
for cutoff in elevation_cutoffs_deg:
    for station in scenario.ground_stations:
        station.min_elevation_deg = float(cutoff)
    results = runner.run_scenario(scenario, family)
    for (observer, spacecraft), windows in results["windows"].items():
        longest_h = max([(stop - start) for start, stop in windows], default=0.0) / 3600.0
        total_h = sum(stop - start for start, stop in windows) / 3600.0
        duty = results["duty_cycle"][(observer, spacecraft)]
        masks = results["observations"][(observer, spacecraft)]["constraint_masks"]
        kinds = results["observations"][(observer, spacecraft)]["constraint_kinds"]
        row = {"min_elevation_deg": cutoff,
               "observer": observer,
               "spacecraft": spacecraft,
               "duty_cycle": duty,
               "n_windows": len(windows),
               "total_hours": total_h,
               "longest_hours": longest_h}
        for kind, column in zip(kinds, masks.T):
            row[f"pass_fraction_{kind}"] = float(column.mean())
        rows.append(row)
        print(f"{cutoff:9.1f} {observer + ' -> ' + spacecraft:<48s} {100 * duty:7.1f} {len(windows):8d} {longest_h:10.2f}")

with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {len(rows)} rows to {csv_path}")
