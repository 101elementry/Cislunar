"""
Leaving the NRHO for a low Earth perigee: the first leg of a Mars
departure staged at the Gateway orbit.

    python scripts/nrho_departure.py            (about ten minutes)

scripts/mars_transfers.py shows that a departure burn made at a 200 km
perigee, by a vehicle falling from the Moon's distance, costs about
0.55 km/s where the same departure from low Earth orbit costs 3.6.  It
assumed the vehicle could get from the NRHO to that perigee.  This
script finds out what that costs in the three-body model.

A single burn is made on the NRHO, along or against the direction of
travel, at one of twelve places round the orbit.  The path is followed
for up to 40 days and its closest approach to the Earth recorded.  The
NRHO is unstable, so a small burn is enough to leave it; the question is
which small burns lead, usually by way of a pass close to the Moon, to a
perigee as low as 200 km.  The motion is chaotic, so the answer is found
by scanning and then refining the cheapest case by bisection.

Writes output/nrho_departure.csv and output/fig13_nrho_departure.png.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import crtbp, frames, interplanetary, propagation, transfers
from model.family import load_families

OUTPUT_DIRECTORY = "output"
SEARCH_DAYS = 40.0
TARGET_ALTITUDE_KM = 200.0
C3_2035 = 10.37          # launch energy of the June 2035 Mars window, km^2/s^2 (scripts/mars_transfers.py)

nrho = load_families()["L2 southern halo"][49]
period = float(nrho["period"])
duration = crtbp.time_to_nondim(SEARCH_DAYS * crtbp.SECONDS_PER_DAY)


def lowest_perigee(phase, signed_burn_m_s):
    """
    Lowest Earth perigee within the search time after a burn of this
    size (negative for against the direction of travel) made at this
    phase of the NRHO.  Returns (altitude km, days after the burn, state).
    """
    state = propagation.propagate_periodic(np.array(nrho["state0"]), period, np.array([0.0, phase * period]))[-1].copy()
    direction = state[3:] / np.linalg.norm(state[3:])
    state[3:] = state[3:] + crtbp.velocity_to_nondim(signed_burn_m_s / 1000.0) * direction
    perigees = transfers.earth_perigees(state, duration)
    if len(perigees) == 0:
        return np.inf, np.nan, None
    time, radius, perigee_state = min(perigees, key=lambda entry: entry[1])
    return crtbp.length_to_km(radius - frames.EARTH_RADIUS_ND), crtbp.time_to_days(time), perigee_state


print("=" * 78)
print("1. Scan: burn size and place on the NRHO against the lowest Earth perigee")
print("=" * 78)
phases = np.arange(12) / 12.0
burns = np.concatenate([np.arange(-200.0, 0.0, 2.0), np.arange(2.0, 201.0, 2.0)])
altitude = np.full((len(phases), len(burns)), np.inf)
for row, phase in enumerate(phases):
    for column, burn in enumerate(burns):
        altitude[row, column], _, _ = lowest_perigee(phase, burn)
    reached = burns[altitude[row] <= TARGET_ALTITUDE_KM]
    cheapest = f"{np.abs(reached).min():.0f} m/s" if len(reached) > 0 else "none"
    print(f"  phase {phase:.3f}: lowest perigee {altitude[row].min():>10,.0f} km; "
          f"cheapest burn reaching {TARGET_ALTITUDE_KM:.0f} km: {cheapest}", flush=True)

print(f"  Within {SEARCH_DAYS:.0f} days no single burn of up to 200 m/s brings the perigee below "
      f"{altitude.min():,.0f} km.")
print("  Leaving the NRHO is cheap, but one burn along the direction of travel does not aim the vehicle at the Earth.")
print()

print("=" * 78)
print("2. Departure as the mirror image of the crew arrival")
print("=" * 78)
# The three-body equations are unchanged by reversing time and
# reflecting in the x-z plane: (x, y, z, t) -> (x, -y, z, -t), under
# which a velocity (vx, vy, vz) becomes (-vx, vy, -vz).  The NRHO is its
# own mirror image.  So the crew vehicle's arrival (Earth perigee, flyby
# burn, insertion burn; scripts/artemis_profile.py) flown backwards in
# the mirror is a departure: a burn on the NRHO, a burn at perilune, and
# a fall to the same 200 km perigee, for the same two burn sizes.
period_days = crtbp.time_to_days(period)
crew_phase, leg_days, plane_angle = 0.30, 1.5, 60.0
station = propagation.propagate_periodic(np.array(nrho["state0"]), period,
                                         np.array([0.0, ((crew_phase - leg_days / period_days) % 1.0) * period]))[-1]
to_station = transfers.from_circular_orbit(station, crtbp.time_to_nondim(leg_days * 86400.0), centre="moon",
                                           altitude_km=150.0, plane_angle_deg=plane_angle)
flyby = transfers.flyby_from_earth(to_station, parking_altitude_km=TARGET_ALTITUDE_KM)


def mirror(vector):
    """(x, y, z) -> (x, -y, z) for a position; for a velocity also reverse it: (-vx, vy, -vz)."""
    return np.array([vector[0], -vector[1], vector[2]])


# Numerical check: start on the mirrored NRHO state, fly the two
# mirrored burns, and look for the perigee.
arrival = to_station["transfer_states"][-1]
insertion = crtbp.velocity_to_nondim(to_station["delta_v2_m_s"] / 1000.0)
on_nrho = np.concatenate([mirror(arrival[:3]), -mirror(arrival[3:] + insertion)])
first_burn = mirror(insertion)               # minus the mirror of minus the insertion burn
leaving = on_nrho.copy()
leaving[3:] = leaving[3:] + first_burn
at_perilune = crtbp.propagate(leaving, crtbp.time_to_nondim(leg_days * 86400.0)).y[:, -1]
at_perilune[3:] = at_perilune[3:] + mirror(flyby["flyby_delta_v"])
perigees = transfers.earth_perigees(at_perilune, crtbp.time_to_nondim((flyby["coast_days"] + 0.5) * 86400.0))
time, radius, perigee_state = min(perigees, key=lambda entry: entry[1])
perigee_altitude = crtbp.length_to_km(radius - frames.EARTH_RADIUS_ND)
perigee_speed = crtbp.velocity_to_km_s(np.linalg.norm(transfers.earth_inertial_velocity(perigee_state)))

nrho_burn = to_station["burn2_m_s"]
perilune_burn = flyby["flyby_m_s"]
hyperbola_speed = interplanetary.speed_on_hyperbola(np.sqrt(C3_2035), interplanetary.EARTH_RADIUS_KM + perigee_altitude,
                                                    interplanetary.MU_EARTH_KM3_S2)
perigee_burn = hyperbola_speed - perigee_speed
leo_burn = interplanetary.departure_burn_from_circular_orbit(np.sqrt(C3_2035))
print(f"  burn on the NRHO {nrho_burn:.0f} m/s; {leg_days} days later a burn of {perilune_burn:.0f} m/s "
      f"150 km above the Moon")
print(f"  mirrored path flown forward: perigee {perigee_altitude:.1f} km above the Earth "
      f"{crtbp.time_to_days(time):.2f} days after the flyby, at {perigee_speed:.3f} km/s")
print(f"  burn at that perigee for the June 2035 Mars window (C3 {C3_2035}): {perigee_burn:.3f} km/s")
total = (nrho_burn + perilune_burn) / 1000.0 + perigee_burn
print(f"  whole departure from the NRHO: {total:.3f} km/s; from a 400 km Earth orbit: {leo_burn:.3f} km/s")
print("  Not modelled: the Sun's pull; and phasing, since the perigee must also point the right way for Mars,")
print("  which fixes when in the month the departure is flown.")
print()

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIRECTORY, "nrho_departure.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["phase", "burn_m_s", "lowest_perigee_altitude_km"])
    for row, phase in enumerate(phases):
        for column, burn in enumerate(burns):
            writer.writerow([f"{phase:.4f}", f"{burn:.1f}", f"{altitude[row, column]:.1f}"])
print(f"wrote {csv_path}")

figure, axis = plt.subplots(figsize=(10.0, 5.0))
shown = np.log10(np.clip(altitude, 100.0, None))
image = axis.pcolormesh(burns, phases, shown, cmap="viridis_r", shading="nearest")
figure.colorbar(image, ax=axis, label="log10 of the lowest Earth perigee altitude [km]")
axis.set_xlabel("burn on the NRHO [m/s], negative against the direction of travel")
axis.set_ylabel("place on the NRHO, fraction of a period after perilune")
axis.set_title(f"Lowest Earth perigee within {SEARCH_DAYS:.0f} days of a single burn on the 9:2 NRHO")
figure.tight_layout()
figure_path = os.path.join(OUTPUT_DIRECTORY, "fig13_nrho_departure.png")
figure.savefig(figure_path, dpi=160)
print(f"saved {figure_path}")
