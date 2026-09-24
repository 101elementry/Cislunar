"""
How far the 9:2 NRHO of the circular restricted three-body problem
drifts once it is flown with the real Earth, Moon and Sun.

    python scripts/ephemeris_divergence.py            (about a minute)

Every result in this repository comes from the CRTBP, where the Earth and
Moon sit on a fixed circle and there is no Sun.  This script takes member
49 of the L2 southern halo family, carries its starting state into the
real Earth-Moon geometry of the epoch (engine.ephemeris_dynamics.
rotating_to_ephemeris: scaled to that day's Earth-Moon distance, with
Kepler's time unit for that distance, centred on the Moon), and flies it in the ephemeris model
with no correction at all.  The number reported is how far the path is
from the CRTBP orbit, measured in the real rotating frame of each instant
to the nearest point of the orbit, together with the closest approach
to the Moon on each lap.

Three runs from the thesis epoch, 2026-01-01 00:00 UTC:
  * the CRTBP itself, flown straight through instead of forced to repeat
    and carried through the same frame change and back, which checks the
    measurement: it should read (almost) zero
  * the ephemeris model with the Earth and Moon on their real orbits but
    no Sun, which isolates what the circular-orbit assumption costs
  * the full ephemeris model, Earth, Moon and Sun

and then the full model from eight epochs spread over one synodic month,
because the answer depends on where the Moon is in its orbit.

The drift of an uncorrected state is not the quality of the CRTBP orbit
as a design: a real NRHO is the CRTBP orbit corrected into the ephemeris
model, which this script does not do.  It is the size of the gap such a
correction has to close, and how quickly an unmaintained spacecraft
leaves.

Writes output/ephemeris_divergence.csv and
output/fig14_ephemeris_divergence.png.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import crtbp, frames, propagation
from engine import ephemeris_dynamics as ed
from engine.ephemeris import utc_to_tdb
from model.ephemeris import load_ephemeris
from model.family import load_families

OUTPUT_DIRECTORY = "output"
EPOCH_UTC = "2026-01-01T00:00:00"
DAYS = 30.0
SAMPLE_HOURS = 1.0
MEMBER = 49

ephemeris = load_ephemeris()
if ephemeris is None:
    sys.exit("data/de440_ephemeris.npz is missing; run python scripts/fetch_ephemeris.py")
orbit = load_families()["L2 southern halo"][MEMBER]
state0 = np.array(orbit["state0"])
period_days = float(crtbp.time_to_days(orbit["period"]))


def moon_impact(time_s, state, *args):
    """Zero at the Moon's surface: stops a run that hits the Moon."""
    return np.linalg.norm(state[:3]) - crtbp.MOON_RADIUS_KM


moon_impact.terminal = True
moon_impact.direction = -1


def closest_to_moon(time_s, state, *args):
    """Zero when the distance from the Moon stops falling and starts rising: each perilune."""
    return np.dot(state[:3], state[3:])


closest_to_moon.direction = 1


# The CRTBP orbit, densely sampled once, non-dimensional.  Distances to it
# are measured to the nearest sample, so they carry a floor of a few tens
# of kilometres near perilune, where samples are furthest apart.
ORBIT_SAMPLES = 20000
orbit_points = propagation.propagate_periodic(state0, float(orbit["period"]),
                                              np.linspace(0.0, float(orbit["period"]), ORBIT_SAMPLES))[:, :3]


def distance_from_orbit_km(epoch_jd, times_s, positions_km):
    """
    How far each position is from the CRTBP orbit, wherever along the
    orbit the nearest point is: the position is carried into the real
    rotating frame of its own instant (ephemeris_to_rotating), the
    nearest orbit point found there, and the gap scaled by that day's
    Earth-Moon distance.  Measuring to the nearest point rather than to
    where the CRTBP clock says the spacecraft should be keeps a timing
    slip from counting as distance: this answers "has it left the orbit".
    """
    distances = np.empty(len(times_s))
    for k, (t, position) in enumerate(zip(times_s, positions_km)):
        jd = epoch_jd + t / ed.SECONDS_PER_DAY
        rotating = ed.ephemeris_to_rotating(np.concatenate([position, np.zeros(3)]), ephemeris, jd)[:3]
        _, day_distance_km, _, _ = ed.instantaneous_frame(ephemeris, jd)
        distances[k] = np.min(np.linalg.norm(orbit_points - rotating, axis=1)) * day_distance_km
    return distances


def ephemeris_run(epoch_utc, days, bodies):
    """
    Fly member 49's carried-over state in the ephemeris model.  Returns
    (times_s, positions_km, hit_moon, perilune radii km in order).
    """
    epoch_jd = float(utc_to_tdb(frames.julian_date(epoch_utc)))
    start = ed.rotating_to_ephemeris(state0, ephemeris, epoch_jd)
    times_s = np.arange(0.0, days * ed.SECONDS_PER_DAY + 1.0, SAMPLE_HOURS * 3600.0)
    result = ed.propagate(start, times_s[-1], ephemeris, epoch_jd, t_eval=times_s, bodies=bodies,
                          events=[moon_impact, closest_to_moon])
    perilunes_km = np.linalg.norm(result.y_events[1][:, :3], axis=1) if len(result.t_events[1]) else np.array([])
    return result.t, result.y[:3].T, len(result.t_events[0]) > 0, perilunes_km


def crtbp_straight_through_km(epoch_jd, times_s):
    """The CRTBP orbit flown without forcing it to repeat, carried into the real frame (positions, km)."""
    result = crtbp.propagate(state0, crtbp.time_to_nondim(times_s[-1]), t_eval=crtbp.time_to_nondim(times_s))
    return np.array([ed.rotating_to_ephemeris(state, ephemeris, epoch_jd + t / ed.SECONDS_PER_DAY)[:3]
                     for t, state in zip(times_s, result.y.T)])


def first_time_above(times_s, errors_km, threshold_km):
    """Days until the error first exceeds the threshold, or NaN if it never does."""
    above = np.nonzero(errors_km > threshold_km)[0]
    return times_s[above[0]] / ed.SECONDS_PER_DAY if len(above) else np.nan


def largest_before(times_s, errors_km, day):
    """Largest distance from the orbit up to this day, or NaN if the run stopped before it."""
    if times_s[-1] < day * ed.SECONDS_PER_DAY - 1.0:
        return np.nan
    return np.max(errors_km[times_s <= day * ed.SECONDS_PER_DAY])


# ---- the three runs from the thesis epoch -----------------------------------

epoch_jd = float(utc_to_tdb(frames.julian_date(EPOCH_UTC)))
axes, distance_km, distance_rate, rate = ed.instantaneous_frame(ephemeris, epoch_jd)
scaled_perilune_km = crtbp.length_to_km(float(orbit["perilune_radius"])) * distance_km / crtbp.LENGTH_UNIT_KM
print(f"Member {MEMBER}, period {period_days:.3f} d, epoch {EPOCH_UTC} UTC")
print(f"  Earth-Moon distance on the day {distance_km:,.0f} km (CRTBP: {crtbp.LENGTH_UNIT_KM:,.0f} km); "
      f"the Earth-Moon line is turning {rate * crtbp.TIME_UNIT_S:.3f} times the CRTBP mean motion")
print(f"  perilune radius {crtbp.length_to_km(float(orbit['perilune_radius'])):,.0f} km in the CRTBP, "
      f"{scaled_perilune_km:,.0f} km scaled to the day's distance")

runs = {}
times_s = np.arange(0.0, DAYS * ed.SECONDS_PER_DAY + 1.0, SAMPLE_HOURS * 3600.0)
runs["CRTBP, flown straight through"] = (times_s, crtbp_straight_through_km(epoch_jd, times_s), False, None)
for label, bodies in [("Earth and Moon on real orbits, no Sun", ("earth",)),
                      ("Earth, Moon and Sun (full model)", ("earth", "sun"))]:
    print(f"  flying: {label}")
    runs[label] = ephemeris_run(EPOCH_UTC, DAYS, bodies)

rows = []
curves = {}
print()
print("Largest distance from the CRTBP orbit (km) by the end of lap 1 and lap 2, and days until it first "
      "passes 100 and 1,000 km")
print(f"{'run':40s} {'lap 1':>9s} {'lap 2':>9s} {'>100 km':>8s} {'>1,000 km':>10s}  closest approaches to the Moon (km)")
for label, (t, positions, hit_moon, perilunes) in runs.items():
    distances = distance_from_orbit_km(epoch_jd, t, positions)
    curves[label] = (t, distances)
    lap1, lap2 = largest_before(t, distances, period_days), largest_before(t, distances, 2 * period_days)
    over_100, over_1000 = first_time_above(t, distances, 100.0), first_time_above(t, distances, 1000.0)
    perilune_text = "the check run" if perilunes is None else ", ".join(f"{r:,.0f}" for r in perilunes)
    note = f"; hit the Moon on day {t[-1] / ed.SECONDS_PER_DAY:.1f}" if hit_moon else ""
    print(f"{label:40s} {lap1:9,.0f} {lap2:9,.0f} {over_100:8.2f} {over_1000:10.2f}  {perilune_text}{note}")
    rows.append({"epoch_utc": EPOCH_UTC, "run": label, "largest_distance_lap_1_km": lap1,
                 "largest_distance_lap_2_km": lap2, "days_to_100_km": over_100, "days_to_1000_km": over_1000,
                 "perilune_radii_km": perilune_text, "hit_moon": hit_moon})

# ---- the full model from eight epochs across one synodic month --------------

print()
print("Full model from eight epochs across a synodic month")
print(f"{'epoch UTC':22s} {'Earth-Moon km':>14s} {'lap 1 km':>9s} {'>100 km':>8s} {'>1,000 km':>10s}")
start = np.datetime64(EPOCH_UTC)
for k in range(8):
    epoch = str(start + np.timedelta64(int(round(k * crtbp.SYNODIC_MONTH_DAYS / 8.0 * 86400)), "s"))
    jd = float(utc_to_tdb(frames.julian_date(epoch)))
    _, day_distance, _, _ = ed.instantaneous_frame(ephemeris, jd)
    t, positions, hit_moon, _ = ephemeris_run(epoch, 2.2 * period_days, ("earth", "sun"))
    distances = distance_from_orbit_km(jd, t, positions)
    lap1 = largest_before(t, distances, period_days)
    over_100, over_1000 = first_time_above(t, distances, 100.0), first_time_above(t, distances, 1000.0)
    print(f"{epoch:22s} {day_distance:14,.0f} {lap1:9,.0f} {over_100:8.2f} {over_1000:10.2f}")
    rows.append({"epoch_utc": epoch, "run": "Earth, Moon and Sun (full model)", "largest_distance_lap_1_km": lap1,
                 "largest_distance_lap_2_km": largest_before(t, distances, 2 * period_days),
                 "days_to_100_km": over_100, "days_to_1000_km": over_1000, "perilune_radii_km": "",
                 "hit_moon": hit_moon})

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIRECTORY, "ephemeris_divergence.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# ---- figure -------------------------------------------------------------------

figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.4, 1]})
for label, (t, distances) in curves.items():
    left.semilogy(t / ed.SECONDS_PER_DAY, np.maximum(distances, 1.0), label=label)
for lap in np.arange(period_days, DAYS, period_days):
    left.axvline(lap, color="0.85", linewidth=0.8, zorder=0)
left.set_xlabel("days from 2026-01-01 00:00 UTC (grey lines: whole laps)")
left.set_ylabel("distance from the CRTBP orbit [km]")
left.set_title(f"Member {MEMBER} (9:2 NRHO) flown uncorrected")
left.legend(fontsize=9)
left.grid(True, which="both", alpha=0.3)

# The full-model path in the real rotating frame of each instant, side-on,
# against the CRTBP orbit, for the first two laps.
t, positions, _, _ = runs["Earth, Moon and Sun (full model)"]
shown = t <= 2.0 * period_days * ed.SECONDS_PER_DAY
side = np.array([ed.ephemeris_to_rotating(np.concatenate([p, np.zeros(3)]), ephemeris,
                                          epoch_jd + s / ed.SECONDS_PER_DAY)[:3]
                 for s, p in zip(t[shown], positions[shown])])
side_km = crtbp.length_to_km(side - crtbp.moon_position())
orbit_km = crtbp.length_to_km(orbit_points - crtbp.moon_position())
right.plot(orbit_km[:, 0], orbit_km[:, 2], color="0.2", linewidth=2.0, label="CRTBP orbit")
right.plot(side_km[:, 0], side_km[:, 2], linewidth=0.9, label="full model, first two laps")
right.add_patch(plt.Circle((0.0, 0.0), crtbp.MOON_RADIUS_KM, color="0.6"))
right.set_aspect("equal")
right.set_xlabel("x from the Moon, away from Earth [km, CRTBP units]")
right.set_ylabel("z from the Moon, north [km, CRTBP units]")
right.set_title("Side-on, in the real rotating frame of each instant")
right.legend(fontsize=9, loc="lower left")
right.grid(True, alpha=0.3)
figure.tight_layout()
figure_path = os.path.join(OUTPUT_DIRECTORY, "fig14_ephemeris_divergence.png")
figure.savefig(figure_path, dpi=160)
print(f"\nwrote {csv_path} and {figure_path}")
