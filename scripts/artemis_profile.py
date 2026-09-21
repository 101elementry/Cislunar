"""
The two crewed legs of an Artemis-style mission, flown to the 9:2 NRHO
in the three-body model:

  lander  a 100 km polar low lunar orbit up to the NRHO (the ascent leg
          of a Human Landing System; the descent is the same in reverse)
  crew    a 200 km low Earth orbit to the NRHO (the outbound leg of a
          crew vehicle), by a direct two-burn transfer

    python scripts/artemis_profile.py           (about 20 minutes)

Each leg is scanned over where on the NRHO the vehicle arrives, how long
the transfer takes and, for the crew leg, the parking orbit plane; the
cheapest of each is kept.  Every case is a Lambert seed corrected in
the full three-body equations (engine/transfers.py).

Writes output/artemis_profile.csv and two scenario files the interface
can load (Examples menu, or Load): scenarios/lander_to_nrho.json and
scenarios/crew_to_nrho.json, each with the target on the NRHO, the
vehicle with its two burns, and the Sydney telescope watching both.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import crtbp, propagation, transfers
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


def scenario_for(name, vehicle_name, best, duration_days):
    """A scenario with the NRHO target, the vehicle and its two burns, and the Sydney telescope."""
    result = best["result"]
    scenario = Scenario(name=name, epoch_utc="2026-01-01T00:00:00", duration_days=duration_days, time_step_s=120.0)
    # The target is written as a state on the periodic orbit so that it starts at the right phase.
    scenario.add(Spacecraft(name="Gateway", source="state", initial_state=[float(v) for v in target_state_at_phase(best["start_phase"])],
                            period_tu=period, propagation="periodic", diameter_m=6.0, albedo=0.25,
                            keep_out_radius_km=10.0))
    departed = result["start_state"].copy()
    departed[3:] = departed[3:] + crtbp.velocity_to_nondim(result["delta_v1_m_s"] / 1000.0)
    scenario.add(Spacecraft(name=vehicle_name, source="state", initial_state=[float(v) for v in departed],
                            propagation="integrate", diameter_m=5.0, albedo=0.3,
                            burns=[{"time_days": float(best["days"]),
                                    "delta_v_m_s": [float(v) for v in result["delta_v2_m_s"]]}]))
    scenario.add(GroundStation(name="Sydney", latitude_deg=-33.87, longitude_deg=151.21, altitude_km=0.05,
                               min_elevation_deg=15.0, max_sun_elevation_deg=-12.0))
    scenario.add(OpticalSensor(name="Sydney 0.5 m telescope", station="Sydney", limiting_magnitude=18.5,
                               lunar_exclusion_deg=2.0))
    return scenario


os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(SCENARIO_DIRECTORY, exist_ok=True)
rows = []

print("=" * 78)
print("1. Lander: 100 km polar low lunar orbit to the NRHO")
print("=" * 78)
lander = scan("moon", 100.0, arrival_phases=[-0.04, -0.02, 0.0, 0.02, 0.04, 0.08],
              transfer_days=[0.3, 0.5, 0.75, 1.0], plane_angles=[90.0, -90.0])
best_lander = lander[0]
print(f"  cheapest: {best_lander['result']['total_delta_v_m_s']:.0f} m/s, {best_lander['days']} d, "
      f"arriving at phase {best_lander['arrival_phase']:+.2f} of the NRHO (0 is perilune)")
print()

print("=" * 78)
print("2. Crew: 200 km low Earth orbit to the NRHO, direct")
print("=" * 78)
# Each crew case takes about a minute: the coast starts in low Earth
# orbit, where the integrator needs very small steps.
crew = scan("earth", 200.0, arrival_phases=[0.3, 0.5, 0.7],
            transfer_days=[4.0, 5.0], plane_angles=[-25.0, -50.0])
best_crew = crew[0]
print(f"  cheapest: trans-lunar injection {best_crew['result']['burn1_m_s']:.0f} m/s, "
      f"NRHO insertion {best_crew['result']['burn2_m_s']:.0f} m/s, {best_crew['days']} d")
print("  A flown crew mission adds a powered lunar flyby on the way in, which roughly halves the insertion;")
print("  this direct transfer is the simpler two-burn version.")
print()

for leg, entries in (("lander", lander), ("crew", crew)):
    for entry in entries:
        result = entry["result"]
        rows.append([leg, f"{entry['arrival_phase']:.2f}", f"{entry['days']:.2f}", f"{entry['plane_angle']:.0f}",
                     f"{result['parking_inclination_deg']:.1f}", f"{result['burn1_m_s']:.1f}",
                     f"{result['burn2_m_s']:.1f}", f"{result['total_delta_v_m_s']:.1f}",
                     f"{result['arrival_position_error_km']:.4f}"])
csv_path = os.path.join(OUTPUT_DIRECTORY, "artemis_profile.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["leg", "arrival_phase", "transfer_days", "plane_angle_deg", "parking_inclination_deg",
                     "burn1_m_s", "burn2_m_s", "total_m_s", "arrival_miss_km"])
    writer.writerows(rows)
print(f"wrote {csv_path}")

for file_name, scenario in (
        ("lander_to_nrho.json", scenario_for("Lander to the NRHO", "Lander", best_lander, best_lander["days"] + 1.5)),
        ("crew_to_nrho.json", scenario_for("Crew vehicle to the NRHO", "Crew vehicle", best_crew, best_crew["days"] + 2.0))):
    path = os.path.join(SCENARIO_DIRECTORY, file_name)
    with open(path, "w") as handle:
        handle.write(scenario.to_json())
    print(f"wrote {path}")
