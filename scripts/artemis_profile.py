"""
The two crewed legs of an Artemis-style mission, flown to the 9:2 NRHO
in the three-body model:

  lander  a 100 km polar low lunar orbit up to the NRHO (the ascent leg
          of a Human Landing System; the descent is the same in
          reverse), arriving at a hold point 30 km behind the station
          and closing through a second hold to 500 m
  crew    a 200 km low Earth orbit to the NRHO with a powered lunar
          flyby: injection, a braking burn at 150 km above the Moon,
          insertion.  The direct two-burn transfer is kept for
          comparison.

    python scripts/artemis_profile.py           (about two minutes)
    python scripts/artemis_profile.py --scan    (also repeats the wide scans, about 20 minutes)

Every transfer is a Lambert seed corrected in the full three-body
equations (engine/transfers.py); the flyby burn is the smallest one
whose incoming path came from a 200 km Earth perigee.

Writes output/artemis_profile.csv and two scenario files the interface
can load (Examples menu, or Load): scenarios/lander_to_nrho.json and
scenarios/crew_to_nrho.json, each with the station on the NRHO, the
vehicle with all its burns, and the Sydney telescope watching both.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import crtbp, propagation, rendezvous, transfers
from model.family import load_families
from model.scenario import Scenario, Spacecraft, GroundStation, OpticalSensor

OUTPUT_DIRECTORY = "output"
SCENARIO_DIRECTORY = "scenarios"
NRHO_INDEX = 49

nrho = load_families()["L2 southern halo"][NRHO_INDEX]
period = float(nrho["period"])
period_days = crtbp.time_to_days(period)


def target_state_at_phase(phase):
    """NRHO state a fraction `phase` of a period after perilune (phase 0 is the family's initial state)."""
    return propagation.propagate_periodic(np.array(nrho["state0"]), period, np.array([0.0, (phase % 1.0) * period]))[-1]


def scan(centre, altitude_km, arrival_phases, transfer_days, plane_angles):
    """Every converged transfer of the scan as rows, cheapest first."""
    found = []
    for arrival_phase in arrival_phases:
        for days in transfer_days:
            # The target is this far round its orbit when the vehicle departs.
            start_phase = arrival_phase - days / period_days
            target0 = target_state_at_phase(start_phase)
            for plane_angle in plane_angles:
                result = transfers.from_circular_orbit(target0, crtbp.time_to_nondim(days * 86400.0), centre=centre,
                                                       altitude_km=altitude_km, plane_angle_deg=plane_angle)
                if result is None:
                    continue
                found.append({"arrival_phase": arrival_phase, "days": days, "plane_angle": plane_angle,
                              "start_phase": start_phase % 1.0, "result": result})
                print(f"    arrive at phase {arrival_phase:+.2f}, {days:.2f} d, plane {plane_angle:+.0f} deg: "
                      f"{result['burn1_m_s']:6.0f} + {result['burn2_m_s']:6.0f} = {result['total_delta_v_m_s']:6.0f} m/s",
                      flush=True)
    return sorted(found, key=lambda entry: entry["result"]["total_delta_v_m_s"])


def to_m_s(delta_v):
    """A burn in LU/TU as a list of m/s for a scenario file."""
    return [float(value) for value in crtbp.velocity_to_km_s(np.asarray(delta_v)) * 1000.0]


def scenario_for(name, vehicle_name, start_phase, departed_state, burns, duration_days):
    """
    A scenario with the station on the NRHO, the vehicle just after its
    first burn with the rest of its burns [(day, delta-v m/s)], and the
    Sydney telescope.
    """
    scenario = Scenario(name=name, epoch_utc="2026-01-01T00:00:00", duration_days=duration_days, time_step_s=120.0)
    # The station is written as a state on the periodic orbit so that it starts at the right phase.
    scenario.add(Spacecraft(name="Gateway", source="state",
                            initial_state=[float(v) for v in target_state_at_phase(start_phase)],
                            period_tu=period, propagation="periodic", diameter_m=6.0, albedo=0.25,
                            keep_out_radius_km=10.0))
    scenario.add(Spacecraft(name=vehicle_name, source="state", initial_state=[float(v) for v in departed_state],
                            propagation="integrate", diameter_m=5.0, albedo=0.3,
                            burns=[{"time_days": float(day), "delta_v_m_s": [float(v) for v in delta_v]}
                                   for day, delta_v in burns]))
    scenario.add(GroundStation(name="Sydney", latitude_deg=-33.87, longitude_deg=151.21, altitude_km=0.05,
                               min_elevation_deg=15.0, max_sun_elevation_deg=-12.0))
    scenario.add(OpticalSensor(name="Sydney 0.5 m telescope", station="Sydney", limiting_magnitude=18.5,
                               lunar_exclusion_deg=2.0))
    return scenario


os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(SCENARIO_DIRECTORY, exist_ok=True)
wide_scan = "--scan" in sys.argv
rows = []


def hours(value):
    return crtbp.time_to_nondim(value * 3600.0)


# --------------------------------------------------------------------------
# 1. Lander
# --------------------------------------------------------------------------
print("=" * 78)
print("1. Lander: 100 km polar low lunar orbit to the NRHO")
print("=" * 78)
if wide_scan:
    lander = scan("moon", 100.0, arrival_phases=[-0.04, -0.02, 0.0, 0.02, 0.04, 0.08],
                  transfer_days=[0.3, 0.5, 0.75, 1.0], plane_angles=[90.0, -90.0])
    for entry in lander:
        result = entry["result"]
        rows.append(["lander scan", f"{entry['arrival_phase']:.2f}", f"{entry['days']:.2f}", f"{entry['plane_angle']:.0f}",
                     f"{result['burn1_m_s']:.1f}", "", f"{result['burn2_m_s']:.1f}", f"{result['total_delta_v_m_s']:.1f}"])
    arrival_phase, lander_days, lander_plane = lander[0]["arrival_phase"], lander[0]["days"], lander[0]["plane_angle"]
else:
    # The cheapest case of the wide scan: arrive just after perilune, half a day, polar.
    arrival_phase, lander_days, lander_plane = 0.08, 0.5, -90.0

lander_start_phase = (arrival_phase - lander_days / period_days) % 1.0
station0 = target_state_at_phase(lander_start_phase)
first_hold = [0.0, -30.0, 0.0]
leg = transfers.from_circular_orbit(station0, crtbp.time_to_nondim(lander_days * 86400.0), centre="moon",
                                    altitude_km=100.0, plane_angle_deg=lander_plane, arrival_offset_km=first_hold)
print(f"  to a hold point 30 km behind the station: departure {leg['burn1_m_s']:.0f} m/s, "
      f"stop {leg['burn2_m_s']:.0f} m/s, {lander_days * 24.0:.0f} h")

# Stepped approach: wait an hour, move to 12 km in an hour, wait half an
# hour, close to 500 m in an hour and a half.  12 km is just outside the
# 10 km keep-out sphere; the last hop is the final approach inside it.
lander_at_hold = leg["transfer_states"][-1].copy()
lander_at_hold[3:] = lander_at_hold[3:] + crtbp.velocity_to_nondim(leg["delta_v2_m_s"] / 1000.0)
station_at_hold = crtbp.propagate(station0, crtbp.time_to_nondim(lander_days * 86400.0)).y[:, -1]
approach, _, _, approach_time = rendezvous.approach_sequence(
    lander_at_hold, station_at_hold,
    [([0.0, -12.0, 0.0], hours(1.0), hours(1.0)), ([0.0, -0.5, 0.0], hours(0.5), hours(1.5))])
lander_burns = [(lander_days, leg["delta_v2_m_s"])]
for burn_time, delta_v in approach:
    lander_burns.append((lander_days + crtbp.time_to_days(burn_time), to_m_s(delta_v)))
approach_m_s = sum(np.linalg.norm(delta_v) for _, delta_v in lander_burns[1:])
print(f"  stepped approach through 12 km to 500 m: {len(approach)} burns, {approach_m_s:.1f} m/s, "
      f"{crtbp.time_to_days(approach_time) * 24.0:.1f} h")
lander_total = leg["total_delta_v_m_s"] + approach_m_s
print(f"  lander total {lander_total:.0f} m/s")
rows.append(["lander", f"{arrival_phase:.2f}", f"{lander_days:.2f}", f"{lander_plane:.0f}",
             f"{leg['burn1_m_s']:.1f}", "", f"{leg['burn2_m_s']:.1f} + {approach_m_s:.1f} approach", f"{lander_total:.1f}"])
lander_departed = leg["start_state"].copy()
lander_departed[3:] = lander_departed[3:] + crtbp.velocity_to_nondim(leg["delta_v1_m_s"] / 1000.0)
print()

# --------------------------------------------------------------------------
# 2. Crew vehicle
# --------------------------------------------------------------------------
print("=" * 78)
print("2. Crew: 200 km low Earth orbit to the NRHO with a powered lunar flyby")
print("=" * 78)
best_crew = None
for crew_phase in (0.25, 0.3, 0.35):
    for leg_days in (1.5, 2.0):
        for plane_angle in (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0):
            station_at_flyby = target_state_at_phase(crew_phase - leg_days / period_days)
            to_station = transfers.from_circular_orbit(station_at_flyby, crtbp.time_to_nondim(leg_days * 86400.0),
                                                       centre="moon", altitude_km=150.0, plane_angle_deg=plane_angle)
            if to_station is None:
                continue
            flyby = transfers.flyby_from_earth(to_station, parking_altitude_km=200.0)
            if flyby is None:
                continue
            after_injection = flyby["flyby_m_s"] + to_station["burn2_m_s"]
            print(f"    arrive at phase {crew_phase:.2f}, {leg_days} d after the flyby, plane {plane_angle:+.0f}: "
                  f"injection {flyby['injection_m_s']:.0f}, flyby {flyby['flyby_m_s']:.0f}, "
                  f"insertion {to_station['burn2_m_s']:.0f} m/s; parking orbit tilted "
                  f"{flyby['parking_inclination_deg']:.0f} deg", flush=True)
            rows.append(["crew flyby", f"{crew_phase:.2f}", f"{flyby['coast_days'] + leg_days:.2f}", f"{plane_angle:.0f}",
                         f"{flyby['injection_m_s']:.1f}", f"{flyby['flyby_m_s']:.1f}", f"{to_station['burn2_m_s']:.1f}",
                         f"{flyby['injection_m_s'] + after_injection:.1f}"])
            if best_crew is None or after_injection < best_crew[0]:
                best_crew = (after_injection, crew_phase, leg_days, to_station, flyby)

after_injection, crew_phase, leg_days, to_station, flyby = best_crew
print(f"  cheapest: injection {flyby['injection_m_s']:.0f} m/s, flyby {flyby['flyby_m_s']:.0f} m/s, "
      f"insertion {to_station['burn2_m_s']:.0f} m/s; {flyby['coast_days']:.1f} d to the Moon, "
      f"{leg_days} d on to the station")
print(f"  after leaving Earth orbit {after_injection:.0f} m/s, against 921 m/s for the direct two-burn transfer")
print("  (direct: injection 3144 m/s, insertion 921 m/s, 4 days; python scripts/artemis_profile.py --scan repeats it)")
if wide_scan:
    crew_direct = scan("earth", 200.0, arrival_phases=[0.3, 0.5, 0.7], transfer_days=[4.0, 5.0],
                       plane_angles=[-25.0, -50.0])
    for entry in crew_direct:
        result = entry["result"]
        rows.append(["crew direct", f"{entry['arrival_phase']:.2f}", f"{entry['days']:.2f}", f"{entry['plane_angle']:.0f}",
                     f"{result['burn1_m_s']:.1f}", "", f"{result['burn2_m_s']:.1f}", f"{result['total_delta_v_m_s']:.1f}"])
print()

crew_days = flyby["coast_days"] + leg_days
crew_burns = [(flyby["coast_days"], to_m_s(flyby["flyby_delta_v"])), (crew_days, to_station["delta_v2_m_s"])]
crew_start_phase = (crew_phase - crew_days / period_days) % 1.0

csv_path = os.path.join(OUTPUT_DIRECTORY, "artemis_profile.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["leg", "arrival_phase", "transfer_days", "plane_angle_deg", "burn1_m_s", "flyby_m_s",
                     "arrival_m_s", "total_m_s"])
    writer.writerows(rows)
print(f"wrote {csv_path}")

for file_name, scenario in (
        ("lander_to_nrho.json", scenario_for("Lander to the NRHO", "Lander", lander_start_phase, lander_departed,
                                             lander_burns, lander_days + 1.0)),
        ("crew_to_nrho.json", scenario_for("Crew vehicle to the NRHO", "Crew vehicle", crew_start_phase,
                                           flyby["perigee_state"], crew_burns, crew_days + 2.0))):
    path = os.path.join(SCENARIO_DIRECTORY, file_name)
    with open(path, "w") as handle:
        handle.write(scenario.to_json())
    print(f"wrote {path}")
