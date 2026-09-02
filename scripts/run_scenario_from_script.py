"""
Run a scenario without the interface.  This is the pattern to use for
thesis work: load or build a Scenario, call analysis.run_scenario, then
do whatever you like with the arrays.

    python scripts/run_scenario_from_script.py [scenario.json]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import crtbp
from mission import analysis, figures
from mission.scenario import Scenario, example_scenario

if len(sys.argv) > 1:
    scenario = Scenario.load(sys.argv[1])
else:
    scenario = example_scenario()

results = analysis.run_scenario(scenario)

print(f"Scenario '{scenario.name}': {scenario.duration_days:g} days from {scenario.epoch_utc}, "
      f"{len(results['times_s']):,} samples at {scenario.time_step_s:g} s")
print()
for (observer, spacecraft_name), windows in results["windows"].items():
    observation = results["observations"][(observer, spacecraft_name)]
    print(f"{observer}  ->  {spacecraft_name}")
    print(f"  duty cycle {100.0 * results['duty_cycle'][(observer, spacecraft_name)]:.1f} %  "
          f"in {len(windows)} windows")
    print(f"  fraction of span passing each constraint: "
          f"horizon {observation['above_horizon'].mean():.2f}, dark {observation['station_dark'].mean():.2f}, "
          f"lit {observation['spacecraft_lit'].mean():.2f}, moon {observation['clear_of_moon'].mean():.2f}, "
          f"magnitude {observation['bright_enough'].mean():.2f}")
    print(f"  magnitude range {observation['apparent_magnitude'].min():.2f} to "
          f"{observation['apparent_magnitude'].max():.2f}, lunar separation up to "
          f"{observation['lunar_separation_deg'].max():.1f} deg")
    print(f"  {'#':>3s} {'start [h]':>10s} {'stop [h]':>10s} {'duration [h]':>13s}")
    for index, (start, stop) in enumerate(windows, start=1):
        print(f"  {index:3d} {start / 3600:10.2f} {stop / 3600:10.2f} {(stop - start) / 3600:13.2f}")
    print()

# The same figure builders the interface uses, written to standalone HTML.
os.makedirs("output", exist_ok=True)
figure = figures.rotating_frame_figure(results["trajectories"], analysis.fixed_geometry(),
                                       station_positions=results["stations"])
figure.write_html(os.path.join("output", "scenario_3d.html"))
print("wrote output/scenario_3d.html")
