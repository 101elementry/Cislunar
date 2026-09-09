"""
Worked example: how frozen is an elliptical lunar frozen orbit in the
CRTBP?  The ELFO preset (12 h, e = 0.6, i = 57 deg, apolune over the
south pole) is integrated for a number of days and its osculating
elements about the Moon are tabulated, along with a low lunar polar
orbit for contrast.  No interface involved.

    python scripts/elfo_drift.py 60 output/elfo_drift.csv

Arguments: days, output CSV.  The Earth's third-body pull is the
dominant perturbation on a high lunar orbit and is what the CRTBP
contains; lunar oblateness, which matters for the low orbit, is not
modelled, so the low orbit looks better behaved here than it would be.
The ELFO's argument of perilune should stay near 90 degrees (apolune
over the south pole) while e and i trade against each other slowly: that
exchange is the Lidov-Kozai mechanism.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import crtbp, kepler, propagation
from model import orbits
from model.scenario import Spacecraft, ELEMENT_PRESETS

days = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "elfo_drift.csv")

times = crtbp.time_to_nondim(np.arange(0.0, days * crtbp.SECONDS_PER_DAY + 1.0, 600.0))
rows = []
for preset_name in ("Elliptical lunar frozen orbit (12 h relay)", "Low lunar polar orbit (100 km)"):
    centre, plane, elements = ELEMENT_PRESETS[preset_name]
    spacecraft = Spacecraft(name=preset_name, source="elements", centre=centre, reference_plane=plane,
                            elements=dict(elements))
    state0 = orbits.initial_state(spacecraft, {}, None)
    states = propagation.propagate_state(state0, times)
    history = kepler.elements_history(states, centre="moon")

    print(f"\n{preset_name}")
    print(f"  {'day':>6s} {'a km':>8s} {'e':>7s} {'i deg':>7s} {'argp deg':>9s} {'RAAN deg':>9s} "
          f"{'perilune km':>12s} {'apolune km':>11s}")
    stride = max(1, len(times) // 12)
    for k in range(0, len(times), stride):
        print(f"  {crtbp.time_to_days(times[k]):6.1f} {history['a_km'][k]:8.0f} {history['e'][k]:7.4f} "
              f"{history['i_deg'][k]:7.2f} {history['argp_deg'][k]:9.2f} {history['raan_deg'][k]:9.2f} "
              f"{history['periapsis_km'][k]:12.0f} {history['apoapsis_km'][k]:11.0f}")
    print(f"  perilune stays between {history['periapsis_km'].min():.0f} and {history['periapsis_km'].max():.0f} km; "
          f"argument of perilune between {history['argp_deg'].min():.1f} and {history['argp_deg'].max():.1f} deg")

    for k in range(len(times)):
        rows.append({"orbit": preset_name, "day": crtbp.time_to_days(times[k]),
                     **{key: float(history[key][k]) for key in ("a_km", "e", "i_deg", "raan_deg", "argp_deg",
                                                               "periapsis_km", "apoapsis_km")}})

with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {len(rows)} rows to {csv_path}")
