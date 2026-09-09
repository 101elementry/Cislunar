"""
Write a GMAT script that propagates every spacecraft of a scenario in
GMAT's high-fidelity model, for cross-checking the CRTBP results.

    python scripts/export_gmat.py output/example_scenario.json output/example_scenario.script

The initial states are handed to GMAT in an Earth-Moon rotating
coordinate system centred on the Earth-Moon barycentre, defined in the
script with GMAT's ObjectReferenced axes (x from the Earth to the
Moon, z along their orbital angular momentum).  That is the CRTBP
frame, so GMAT does the exact conversion to its inertial frame using
the real Moon position at the epoch; the only approximation is the
fixed length unit (384,400 km) against the Moon's true distance on
that date, a difference of a few percent that any comparison should
keep in mind.  Kilometres and km/s are what GMAT wants.

The script propagates for the scenario's duration with the Earth,
Moon and Sun as point masses plus the lunar gravity field, reports the
state in the same rotating system every time step, and draws it.  Open
the .script in GMAT and press Run.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import crtbp, frames
from model import orbits
from model.family import load_families
from model.scenario import Scenario

scenario_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "example_scenario.json")
script_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "example_scenario.script")

scenario = Scenario.load(scenario_path)
families = load_families()
epoch_jd = frames.julian_date(scenario.epoch_utc)
epoch = datetime.fromisoformat(scenario.epoch_utc)
gmat_epoch = epoch.strftime("%d %b %Y %H:%M:%S.000")


def gmat_name(name):
    """GMAT object names are identifiers: letters, digits, underscores."""
    cleaned = "".join(character if character.isalnum() else "_" for character in name)
    if cleaned[0].isdigit():
        cleaned = "S_" + cleaned
    return cleaned


lines = [f"% Generated from {scenario_path} by scripts/export_gmat.py",
         "% Earth-Moon rotating, barycentric coordinate system: the CRTBP frame.",
         "",
         "Create Barycenter EarthMoonBary;",
         "GMAT EarthMoonBary.BodyNames = {Earth, Luna};",
         "",
         "Create CoordinateSystem EarthMoonRot;",
         "GMAT EarthMoonRot.Origin = EarthMoonBary;",
         "GMAT EarthMoonRot.Axes = ObjectReferenced;",
         "GMAT EarthMoonRot.XAxis = R;",
         "GMAT EarthMoonRot.ZAxis = N;",
         "GMAT EarthMoonRot.Primary = Earth;",
         "GMAT EarthMoonRot.Secondary = Luna;",
         ""]

names = []
for spacecraft in scenario.spacecraft:
    state = orbits.initial_state(spacecraft, families, epoch_jd)
    position_km = crtbp.length_to_km(state[:3])
    velocity_km_s = crtbp.velocity_to_km_s(state[3:])
    name = gmat_name(spacecraft.name)
    names.append(name)
    lines.extend([f"Create Spacecraft {name};",
                  f"GMAT {name}.DateFormat = UTCGregorian;",
                  f"GMAT {name}.Epoch = '{gmat_epoch}';",
                  f"GMAT {name}.CoordinateSystem = EarthMoonRot;",
                  f"GMAT {name}.DisplayStateType = Cartesian;",
                  f"GMAT {name}.X = {position_km[0]:.9f};",
                  f"GMAT {name}.Y = {position_km[1]:.9f};",
                  f"GMAT {name}.Z = {position_km[2]:.9f};",
                  f"GMAT {name}.VX = {velocity_km_s[0]:.12f};",
                  f"GMAT {name}.VY = {velocity_km_s[1]:.12f};",
                  f"GMAT {name}.VZ = {velocity_km_s[2]:.12f};",
                  ""])

lines.extend(["Create ForceModel CislunarForces;",
              "GMAT CislunarForces.CentralBody = Earth;",
              "GMAT CislunarForces.PrimaryBodies = {Earth};",
              "GMAT CislunarForces.PointMasses = {Luna, Sun};",
              "GMAT CislunarForces.GravityField.Earth.Degree = 4;",
              "GMAT CislunarForces.GravityField.Earth.Order = 4;",
              "GMAT CislunarForces.Drag = None;",
              "GMAT CislunarForces.SRP = Off;",
              "",
              "Create Propagator Cislunar;",
              "GMAT Cislunar.FM = CislunarForces;",
              "GMAT Cislunar.Type = RungeKutta89;",
              "GMAT Cislunar.InitialStepSize = 60;",
              "GMAT Cislunar.Accuracy = 1e-12;",
              "GMAT Cislunar.MinStep = 0.001;",
              "GMAT Cislunar.MaxStep = 600;",
              "",
              "Create ReportFile RotatingStates;",
              f"GMAT RotatingStates.Filename = '{os.path.basename(script_path).replace('.script', '')}_rotating.txt';",
              "GMAT RotatingStates.WriteHeaders = true;",
              "GMAT RotatingStates.Add = {" + ", ".join(
                  f"{name}.UTCModJulian, {name}.EarthMoonRot.X, {name}.EarthMoonRot.Y, {name}.EarthMoonRot.Z, "
                  f"{name}.EarthMoonRot.VX, {name}.EarthMoonRot.VY, {name}.EarthMoonRot.VZ" for name in names) + "};",
              "",
              "Create OrbitView RotatingView;",
              "GMAT RotatingView.Add = {" + ", ".join(names + ["Earth", "Luna"]) + "};",
              "GMAT RotatingView.CoordinateSystem = EarthMoonRot;",
              "GMAT RotatingView.ViewPointReference = Luna;",
              "GMAT RotatingView.ViewDirection = Luna;",
              "GMAT RotatingView.ViewUpAxis = Z;",
              "",
              "BeginMissionSequence;",
              f"Propagate Cislunar({', '.join(names)}) {{{names[0]}.ElapsedDays = {scenario.duration_days:g}}};",
              ""])

with open(script_path, "w") as handle:
    handle.write("\n".join(lines))
print(f"wrote GMAT script for {len(names)} spacecraft to {script_path}")
print("The CRTBP states are given in EarthMoonRot (km, km/s); GMAT converts them with the real ephemeris.")
