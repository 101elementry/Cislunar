"""
Earth-Mars transfer windows, and what staging in the NRHO is worth.

    python scripts/mars_transfers.py            (about one minute)

1. Scans departures from 2030 to 2042 for the cheapest transfer in each
   26-month launch window (patched conics, JPL DE440 planet positions,
   Lambert's problem for each pair of dates).
2. For each window compares the departure burn from a 400 km circular
   Earth orbit with the burn at a 200 km perigee reached by falling
   from the Moon's distance, which is how a vehicle staged in the NRHO
   would leave.  The difference is the Oberth effect.
3. Draws the porkchop plot of one window (fig12).
4. Compares a long-stay mission (wait at Mars for the next return
   window) with a 30-sol short stay for the same window.

Writes output/mars_windows.csv and output/fig12_mars_porkchop.png.
"""

import csv
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import frames, interplanetary
from model.ephemeris import load_ephemeris

OUTPUT_DIRECTORY = "output"
SYNODIC_PERIOD_DAYS = 779.94
PORKCHOP_WINDOW = 2035          # the launch year drawn in the figure and used for the stay comparison


def calendar_date(jd):
    """ISO date of a Julian date."""
    return (datetime(2000, 1, 1, 12) + timedelta(days=float(jd) - 2451545.0)).strftime("%Y-%m-%d")


ephemeris = load_ephemeris()
if ephemeris is None or "mars" not in ephemeris.segments:
    raise SystemExit("data/de440_ephemeris.npz has no Mars; run python scripts/fetch_ephemeris.py")

# --------------------------------------------------------------------------
# 1. The launch windows
# --------------------------------------------------------------------------
print("=" * 78)
print("1. Cheapest transfer in each launch window, 2030 to 2042")
print("=" * 78)
scan_start = frames.julian_date("2030-03-01T00:00:00")
scan_stop = frames.julian_date("2042-06-01T00:00:00")
departures = np.arange(scan_start, scan_stop, 5.0)
flight_times = np.arange(120.0, 420.0, 5.0)

best_per_departure = []
for departure_jd in departures:
    best = None
    for flight_days in flight_times:
        try:
            result = interplanetary.transfer(ephemeris, departure_jd, departure_jd + flight_days)
        except ValueError:
            continue
        # Cost: excess speed to be supplied at the Earth plus excess speed
        # to be removed at Mars.
        cost = np.sqrt(result["c3_km2_s2"]) + np.linalg.norm(result["v_infinity_arrive"])
        if best is None or cost < best[0]:
            best = (cost, departure_jd, flight_days, result)
    best_per_departure.append(best)

# One minimum per synodic period: a departure date is a window's
# optimum if nothing within half a synodic period either side is cheaper.
costs = np.array([entry[0] for entry in best_per_departure])
half_window = int(0.5 * SYNODIC_PERIOD_DAYS / 5.0)
windows = []
for index in range(len(costs)):
    neighbours = costs[max(0, index - half_window):index + half_window + 1]
    at_edge = index < 10 or index > len(costs) - 10
    if costs[index] == neighbours.min() and not at_edge:
        windows.append(best_per_departure[index])

rows = []
header = (f"  {'depart':>10s} {'arrive':>10s} {'days':>5s} {'C3':>6s} {'v_inf arr':>9s} "
          f"{'LEO burn':>9s} {'NRHO burn':>9s} {'saving':>7s} {'capture':>8s}")
print(header)
for cost, departure_jd, flight_days, result in windows:
    v_infinity = np.sqrt(result["c3_km2_s2"])
    v_arrive = np.linalg.norm(result["v_infinity_arrive"])
    leo_burn = interplanetary.departure_burn_from_circular_orbit(v_infinity)
    staged_burn = interplanetary.departure_burn_from_lunar_distance(v_infinity)
    capture = interplanetary.capture_burn(v_arrive)
    rows.append([calendar_date(departure_jd), calendar_date(departure_jd + flight_days), f"{flight_days:.0f}",
                 f"{result['c3_km2_s2']:.2f}", f"{v_arrive:.3f}", f"{leo_burn:.3f}", f"{staged_burn:.3f}",
                 f"{leo_burn - staged_burn:.3f}", f"{capture:.3f}"])
    print(f"  {rows[-1][0]:>10s} {rows[-1][1]:>10s} {flight_days:5.0f} {result['c3_km2_s2']:6.2f} {v_arrive:9.3f} "
          f"{leo_burn:9.3f} {staged_burn:9.3f} {leo_burn - staged_burn:7.3f} {capture:8.3f}")
print("  C3 in km^2/s^2, speeds and burns in km/s.")
print("  LEO burn: from a 400 km circular orbit.  NRHO burn: at a 200 km perigee after falling from")
print("  the Moon's distance; the few hundred m/s to leave the NRHO and lower the perigee are extra.")
print("  capture: into the one-sol 250 x 33,800 km Mars orbit.")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIRECTORY, "mars_windows.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["depart", "arrive", "flight_days", "c3_km2_s2", "v_infinity_arrive_km_s",
                     "burn_from_400km_leo_km_s", "burn_at_perigee_from_lunar_distance_km_s",
                     "saving_km_s", "mars_capture_burn_km_s"])
    writer.writerows(rows)
print(f"  wrote {csv_path}")
print()

# --------------------------------------------------------------------------
# 2. Porkchop plot of one window
# --------------------------------------------------------------------------
print("=" * 78)
print(f"2. Porkchop plot of the {PORKCHOP_WINDOW} window")
print("=" * 78)
chosen = min(windows, key=lambda entry: abs(entry[1] - frames.julian_date(f"{PORKCHOP_WINDOW}-07-01T00:00:00")))
centre_departure = chosen[1]
grid_departures = np.arange(centre_departure - 110.0, centre_departure + 110.0, 3.0)
grid_arrivals = np.arange(centre_departure + 100.0, centre_departure + 480.0, 4.0)
c3, v_arrive_grid = interplanetary.porkchop(ephemeris, grid_departures, grid_arrivals)

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
departure_days = grid_departures - grid_departures[0]
arrival_days = grid_arrivals - grid_departures[0]
panels = [(c3, [8, 10, 12, 15, 20, 25, 30, 40, 50], "Launch energy C3 [km$^2$/s$^2$]"),
          (v_arrive_grid, [2.5, 3, 3.5, 4, 5, 6, 7, 8], "Arrival excess speed at Mars [km/s]")]
for axis, (values, levels, title) in zip(axes, panels):
    contours = axis.contour(departure_days, arrival_days, values, levels=levels, cmap="viridis", linewidths=1.2)
    axis.clabel(contours, fmt="%g", fontsize=8)
    flight = arrival_days[:, np.newaxis] - departure_days[np.newaxis, :]
    flight_lines = axis.contour(departure_days, arrival_days, flight, levels=[150, 200, 250, 300, 350],
                                colors="0.6", linewidths=0.7, linestyles="dashed")
    axis.clabel(flight_lines, fmt="%g d", fontsize=7)
    axis.plot(chosen[1] - grid_departures[0], chosen[1] + chosen[2] - grid_departures[0], "k+", markersize=12)
    axis.set_title(title)
    axis.set_xlabel(f"departure, days after {calendar_date(grid_departures[0])}")
    axis.grid(alpha=0.2)
axes[0].set_ylabel(f"arrival, days after {calendar_date(grid_departures[0])}")
figure.suptitle(f"Earth to Mars, {PORKCHOP_WINDOW} window (JPL DE440, patched conics). "
                f"Cross: depart {calendar_date(chosen[1])}, {chosen[2]:.0f} days", fontsize=11)
figure.tight_layout()
figure_path = os.path.join(OUTPUT_DIRECTORY, "fig12_mars_porkchop.png")
figure.savefig(figure_path, dpi=160)
print(f"  saved {figure_path}")
print()

# --------------------------------------------------------------------------
# 3. Long stay against a 30-sol short stay
# --------------------------------------------------------------------------
print("=" * 78)
print("3. Round trip: long stay against a 30-sol stay, same window")
print("=" * 78)


def cheapest_return(mars_departures, flight_times_days):
    """Cheapest Mars-to-Earth transfer among the given Mars departure dates: (cost, jd, days, result)."""
    best = None
    for mars_departure in mars_departures:
        for flight_days in flight_times_days:
            try:
                result = interplanetary.transfer(ephemeris, mars_departure, mars_departure + flight_days,
                                                 origin="mars", destination="earth")
            except ValueError:
                continue
            cost = np.linalg.norm(result["v_infinity_depart"])
            if best is None or cost < best[0]:
                best = (cost, mars_departure, flight_days, result)
    return best


def mission_budget(outbound, inbound):
    """Burns of a round trip in km/s: Earth departure from the staged perigee, Mars capture, Mars departure."""
    depart = interplanetary.departure_burn_from_lunar_distance(np.sqrt(outbound["c3_km2_s2"]))
    capture = interplanetary.capture_burn(np.linalg.norm(outbound["v_infinity_arrive"]))
    # Leaving the one-sol orbit is the capture burn run backwards.
    leave = interplanetary.capture_burn(np.linalg.norm(inbound["v_infinity_depart"]))
    return depart, capture, leave


_, out_departure, out_flight, outbound = chosen
mars_arrival = out_departure + out_flight

# Long stay: the cheap outbound, then wait for the cheap way home.
long_return = cheapest_return(np.arange(mars_arrival + 300.0, mars_arrival + 650.0, 10.0), np.arange(150.0, 400.0, 10.0))
long_budget = mission_budget(outbound, long_return[3])

# Short stay: leave 30 sols (30.8 days) after arriving, whatever that
# costs, and choose the outbound that makes the total least.  Done
# twice: with no limit on the mission length, and holding the whole
# mission under 650 days, which is what "short stay" usually means.
best_short = None
best_fast = None
for departure_jd in np.arange(out_departure - 250.0, out_departure + 150.0, 10.0):
    for flight_days in np.arange(120.0, 360.0, 10.0):
        try:
            candidate_out = interplanetary.transfer(ephemeris, departure_jd, departure_jd + flight_days)
        except ValueError:
            continue
        mars_departure = departure_jd + flight_days + 30.8
        for limit, slot in ((460.0, "any"), (650.0 - flight_days - 30.8, "fast")):
            if limit < 150.0:
                continue
            candidate_return = cheapest_return([mars_departure], np.arange(150.0, limit + 1.0, 10.0))
            if candidate_return is None:
                continue
            budget = mission_budget(candidate_out, candidate_return[3])
            entry = (budget, departure_jd, flight_days, candidate_return)
            if slot == "any" and (best_short is None or sum(budget) < sum(best_short[0])):
                best_short = entry
            if slot == "fast" and (best_fast is None or sum(budget) < sum(best_fast[0])):
                best_fast = entry

for label, budget, depart_jd, flight, back in (
        ("long stay", long_budget, out_departure, out_flight, long_return),
        ("30-sol stay", best_short[0], best_short[1], best_short[2], best_short[3]),
        ("30 sol <650d", best_fast[0], best_fast[1], best_fast[2], best_fast[3])):
    stay = back[1] - (depart_jd + flight)
    total_days = flight + stay + back[2]
    entry_speed = interplanetary.speed_on_hyperbola(np.linalg.norm(back[3]["v_infinity_arrive"]),
                                                    interplanetary.EARTH_RADIUS_KM + 125.0,
                                                    interplanetary.MU_EARTH_KM3_S2)
    print(f"  {label:12s} depart {calendar_date(depart_jd)}, out {flight:.0f} d, stay {stay:.0f} d, "
          f"back {back[2]:.0f} d, total {total_days:.0f} d")
    print(f"  {'':12s} burns: Earth departure {budget[0]:.2f}, Mars capture {budget[1]:.2f}, "
          f"Mars departure {budget[2]:.2f}, sum {sum(budget):.2f} km/s; Earth entry speed {entry_speed:.1f} km/s")
print("  The short stay leaves Mars when the Earth is badly placed, so the way home is expensive and the")
print("  entry is fast; real short-stay designs add a Venus flyby to tame it, which this does not model.")
