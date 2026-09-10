"""
One-off: extract the Sun, Earth-Moon barycentre, Earth and Moon from the
JPL DE440s kernel into a small array file the engine can evaluate with
numpy alone.

    python scripts/fetch_ephemeris.py [data/de440s.bsp] [2020-01-01] [2041-01-01]

Downloads the kernel from NAIF if it is not present (32 MB), reads the
Chebyshev coefficient blocks of the four segments with jplephem, keeps
only the intervals covering the requested span, and writes
data/de440_ephemeris.npz.  jplephem is needed only here; the runtime
uses engine/ephemeris.py, which is a Chebyshev evaluator.

Segments (centre -> target, NAIF ids): 0 -> 3 solar-system barycentre to
Earth-Moon barycentre, 0 -> 10 to the Sun, 3 -> 399 barycentre to Earth,
3 -> 301 barycentre to Moon.  Positions are ICRF (J2000 equatorial)
kilometres; times are TDB Julian dates.
"""

import os
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from jplephem.spk import SPK

KERNEL_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
SEGMENTS = {"emb": (0, 3), "sun": (0, 10), "earth": (3, 399), "moon": (3, 301)}
JD_J2000 = 2451545.0


def julian_date(iso_utc):
    moment = datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc)
    return JD_J2000 + (moment - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0


kernel_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "de440s.bsp")
start_jd = julian_date(sys.argv[2] if len(sys.argv) > 2 else "2020-01-01")
stop_jd = julian_date(sys.argv[3] if len(sys.argv) > 3 else "2041-01-01")
output_path = os.path.join("data", "de440_ephemeris.npz")

os.makedirs("data", exist_ok=True)
if not os.path.exists(kernel_path):
    print(f"downloading {KERNEL_URL} ...")
    urllib.request.urlretrieve(KERNEL_URL, kernel_path)

kernel = SPK.open(kernel_path)
arrays = {"start_jd": start_jd, "stop_jd": stop_jd, "source": "JPL DE440s (de440s.bsp)"}
for name, (centre, target) in SEGMENTS.items():
    segment = kernel[centre, target]
    init, interval_days, coefficients = segment.load_array()
    # Keep the intervals that overlap the span, with one spare each side.
    first = max(0, int(np.floor((start_jd - init) / interval_days)) - 1)
    last = min(coefficients.shape[1], int(np.ceil((stop_jd - init) / interval_days)) + 1)
    arrays[f"{name}_init"] = init + first * interval_days
    arrays[f"{name}_interval_days"] = interval_days
    arrays[f"{name}_coefficients"] = np.array(coefficients[:, first:last, :], dtype=float)
    print(f"{name:5s} {centre:2d} -> {target:3d}: {last - first} intervals of {interval_days:g} days, "
          f"{coefficients.shape[2]} coefficients each")

np.savez_compressed(output_path, **arrays)
print(f"wrote {output_path} ({os.path.getsize(output_path) / 1e6:.2f} MB)")
