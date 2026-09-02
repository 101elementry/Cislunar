"""Stage 1 check: build a scenario, save it, load it back, compare."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mission.scenario import Scenario, Spacecraft, example_scenario

scenario = example_scenario()
scenario.add(Spacecraft(name="Free-flyer", source="state",
                        initial_state=[1.0221, 0.0, -0.1821, 0.0, -0.1018, 0.0]))
scenario.add(Spacecraft(name="Free-flyer"))   # name collision -> renamed

path = os.path.join("output", "example_scenario.json")
os.makedirs("output", exist_ok=True)
scenario.save(path)
reloaded = Scenario.load(path)

print("saved to", path)
print(open(path).read())
print("round trip identical:", reloaded.to_dict() == scenario.to_dict())
print("objects:", [obj.name for obj in reloaded.all_objects()])
print("sensors of Sydney:", [s.name for s in reloaded.sensors_of("Sydney")])
grid = reloaded.time_grid_seconds()
print("time grid: %d samples, %.0f s to %.0f s, %.4f TU" % (len(grid), grid[0], grid[-1],
                                                            reloaded.time_grid_nondim()[-1]))
reloaded.remove("Sydney")
print("after removing Sydney:", [obj.name for obj in reloaded.all_objects()])
