"""
A 30-day stay at Mars, with and without a Venus flyby.

    python scripts/mars_short_stay.py           (about three minutes)

scripts/mars_transfers.py shows that leaving Mars a month after arriving
is expensive, because the Earth is then badly placed: the way home is
long, costly, and ends in a very fast entry.  Short-stay ("opposition
class") mission designs get round this by passing Venus on one of the
two legs.  Venus is nearer the Sun and moves faster, so a pass behind or
in front of it bends and re-times the path for no propellant.

For every Earth departure date in the search and every way of flying
each leg (direct, or by way of Venus), this script adds up the burns:

    Earth departure   at a 200 km perigee, staged from lunar distance
    Venus flyby       free if the speeds in and out match, otherwise a
                      burn at closest approach (engine.interplanetary
                      .gravity_assist); passes below 300 km are rejected
    Mars capture      into the one-sol orbit, and the same to leave it
and records the speed of the final entry at the Earth.  Entries faster
than 13 km/s are rejected, the limit NASA's reference studies use.

Dates are on a 10-day grid, so the results are good to a few per cent.
Writes output/mars_short_stay.csv.
"""

import csv
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import frames, interplanetary
from model.ephemeris import load_ephemeris

OUTPUT_DIRECTORY = "output"
STEP_DAYS = 10
STAY_DAYS = 30
ENTRY_LIMIT_KM_S = 13.0
ENTRY_ALTITUDE_KM = 125.0

ephemeris = load_ephemeris()
if ephemeris is None or "venus" not in ephemeris.segments:
    raise SystemExit("data/de440_ephemeris.npz has no Venus; run python scripts/fetch_ephemeris.py")

base_jd = frames.julian_date("2034-01-01T00:00:00")


def calendar_date(day):
    """ISO date of a day count from the base date."""
    return (datetime(2034, 1, 1) + timedelta(days=int(day))).strftime("%Y-%m-%d")


def leg(origin, destination, depart_day, arrive_day):
    """Excess velocities (out of the origin, into the destination) of one leg, or None."""
    try:
        result = interplanetary.transfer(ephemeris, base_jd + depart_day, base_jd + arrive_day, origin, destination)
    except ValueError:
        return None
    return result["v_infinity_depart"], result["v_infinity_arrive"]


def via_venus(origin, destination, depart_day, first_days, second_days):
    """
    Two legs joined at Venus.  Returns (v_infinity out of the origin,
    v_infinity into the destination, flyby burn km/s, flyby altitude km)
    or None if either leg fails or the pass would be too low.
    """
    first = leg(origin, "venus", depart_day, depart_day + first_days)
    second = leg("venus", destination, depart_day + first_days, depart_day + first_days + second_days)
    if first is None or second is None:
        return None
    altitude, burn, feasible = interplanetary.gravity_assist(first[1], second[0])
    if not feasible:
        return None
    return first[0], second[1], burn, altitude


def entry_speed(v_infinity):
    return interplanetary.speed_on_hyperbola(np.linalg.norm(v_infinity), interplanetary.EARTH_RADIUS_KM + ENTRY_ALTITUDE_KM,
                                             interplanetary.MU_EARTH_KM3_S2)


# --------------------------------------------------------------------------
# Outbound: the cheapest way to arrive at Mars on each date
# --------------------------------------------------------------------------
print("searching outbound legs ...", flush=True)
outbound = {}          # arrival day -> (burns km/s, description, departure day)
for depart_day in range(0, 1100, STEP_DAYS):
    for flight in range(120, 370, STEP_DAYS):
        found = leg("earth", "mars", depart_day, depart_day + flight)
        if found is not None:
            cost = (interplanetary.departure_burn_from_lunar_distance(np.linalg.norm(found[0]))
                    + interplanetary.capture_burn(np.linalg.norm(found[1])))
            arrive = depart_day + flight
            if arrive not in outbound or cost < outbound[arrive][0]:
                outbound[arrive] = (cost, "direct", depart_day)
    for first_days in range(80, 230, STEP_DAYS):
        for second_days in range(100, 310, STEP_DAYS):
            found = via_venus("earth", "mars", depart_day, first_days, second_days)
            if found is None:
                continue
            cost = (interplanetary.departure_burn_from_lunar_distance(np.linalg.norm(found[0])) + found[2]
                    + interplanetary.capture_burn(np.linalg.norm(found[1])))
            arrive = depart_day + first_days + second_days
            if arrive not in outbound or cost < outbound[arrive][0]:
                outbound[arrive] = (cost, f"Venus flyby at {found[3]:,.0f} km", depart_day)

# --------------------------------------------------------------------------
# Return: the cheapest way home from each Mars departure date
# --------------------------------------------------------------------------
print("searching return legs ...", flush=True)
inbound = {"direct": {}, "venus": {}}      # kind -> Mars departure day -> (burns, description, Earth arrival day, entry)
for leave_day in sorted(arrive + STAY_DAYS for arrive in outbound):
    for flight in range(120, 410, STEP_DAYS):
        found = leg("mars", "earth", leave_day, leave_day + flight)
        if found is None or entry_speed(found[1]) > ENTRY_LIMIT_KM_S:
            continue
        cost = interplanetary.capture_burn(np.linalg.norm(found[0]))
        if leave_day not in inbound["direct"] or cost < inbound["direct"][leave_day][0]:
            inbound["direct"][leave_day] = (cost, "direct", leave_day + flight, entry_speed(found[1]))
    for first_days in range(100, 270, STEP_DAYS):
        for second_days in range(80, 230, STEP_DAYS):
            found = via_venus("mars", "earth", leave_day, first_days, second_days)
            if found is None or entry_speed(found[1]) > ENTRY_LIMIT_KM_S:
                continue
            cost = interplanetary.capture_burn(np.linalg.norm(found[0])) + found[2]
            if leave_day not in inbound["venus"] or cost < inbound["venus"][leave_day][0]:
                inbound["venus"][leave_day] = (cost, f"Venus flyby at {found[3]:,.0f} km",
                                               leave_day + first_days + second_days, entry_speed(found[1]))

# --------------------------------------------------------------------------
# Whole missions
# --------------------------------------------------------------------------
missions = []
for arrive, (out_cost, out_kind, depart_day) in outbound.items():
    for kind in ("direct", "venus"):
        back = inbound[kind].get(arrive + STAY_DAYS)
        if back is None:
            continue
        missions.append({"total": out_cost + back[0], "depart": depart_day, "arrive": arrive, "home": back[2],
                         "outbound": out_kind, "return": back[1], "entry": back[3],
                         "out_cost": out_cost, "back_cost": back[0]})


def best(condition):
    chosen = [mission for mission in missions if condition(mission)]
    return min(chosen, key=lambda mission: mission["total"]) if chosen else None


print()
print("=" * 78)
print(f"30-day stay, entry no faster than {ENTRY_LIMIT_KM_S:g} km/s, Earth departures 2034 to 2036")
print("=" * 78)
cases = [("no Venus flyby", lambda m: m["outbound"] == "direct" and m["return"] == "direct"),
         ("Venus on the way home", lambda m: m["outbound"] == "direct" and m["return"] != "direct"),
         ("Venus on the way out", lambda m: m["outbound"] != "direct" and m["return"] == "direct"),
         ("best of all", lambda m: True),
         ("best under 650 days", lambda m: m["home"] - m["depart"] <= 650)]
rows = []
for label, condition in cases:
    mission = best(condition)
    if mission is None:
        print(f"  {label:24s} no mission meets the limits")
        continue
    print(f"  {label:24s} depart {calendar_date(mission['depart'])}, at Mars {calendar_date(mission['arrive'])}, "
          f"home {calendar_date(mission['home'])}, {mission['home'] - mission['depart']} days")
    print(f"  {'':24s} out: {mission['outbound']}, {mission['out_cost']:.2f} km/s; back: {mission['return']}, "
          f"{mission['back_cost']:.2f} km/s; total {mission['total']:.2f} km/s; entry {mission['entry']:.1f} km/s")
    rows.append([label, calendar_date(mission["depart"]), calendar_date(mission["arrive"]), calendar_date(mission["home"]),
                 mission["home"] - mission["depart"], mission["outbound"], mission["return"],
                 f"{mission['out_cost']:.3f}", f"{mission['back_cost']:.3f}", f"{mission['total']:.3f}",
                 f"{mission['entry']:.2f}"])
print("  Burns counted: Earth departure at a 200 km perigee staged from lunar distance, any burn at Venus,")
print("  Mars capture into the one-sol orbit and departure from it.  For comparison a long-stay mission")
print("  in the same window costs about 2.8 km/s (scripts/mars_transfers.py).")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIRECTORY, "mars_short_stay.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["case", "depart", "arrive_mars", "home", "total_days", "outbound", "return",
                     "outbound_burns_km_s", "return_burns_km_s", "total_km_s", "entry_speed_km_s"])
    writer.writerows(rows)
print(f"wrote {csv_path}")
