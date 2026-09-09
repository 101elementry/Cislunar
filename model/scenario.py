"""
Scenario data model and JSON save/load.

A scenario is a time span plus a tree of objects: spacecraft, ground
stations, and optical sensors attached to ground stations.  This module
holds only the description of the scenario.  The engine does the
physics (see model/runner.py for the translation) and app/ only reads
and edits these objects.

Conventions
  * Spacecraft initial states are in the non-dimensional Earth-Moon
    rotating frame of crtbp.py: [x, y, z, vx, vy, vz] in LU and LU/TU.
  * Station coordinates are geodetic latitude and longitude in degrees
    and altitude in kilometres above the reference sphere.
  * The epoch is a UTC ISO-8601 string.  It fixes where the Sun and the
    Earth's rotation are at scenario time zero; see frames.py.
"""

import json
from dataclasses import dataclass, field, asdict, fields

import numpy as np

from engine import crtbp


@dataclass
class Spacecraft:
    """
    A spacecraft and the orbit it starts on.  Three ways to define it:

    source = "state"     a rotating-frame initial state [x, y, z, vx, vy,
                         vz] in LU and LU/TU.  If period_tu > 0 the state
                         has been corrected to a periodic orbit with that
                         period and can be propagated as "periodic".
    source = "family"    member family_index of the named orbit family
                         (see model/family.py); families are periodic.
    source = "elements"  osculating two-body elements about the Moon or
                         the Earth (see engine/kepler.py): a lunar relay
                         on a frozen orbit, a GEO parking orbit.  Always
                         integrated.

    propagation  : "integrate" propagates the initial state through the
                   whole span with the full equations of motion.
                   "periodic" repeats one converged period exactly, which
                   is what a station-kept spacecraft on the orbit does; it
                   needs a known period (family member or corrected state).
    diameter_m   : diameter of the equivalent diffuse sphere, metres.
    albedo       : geometric albedo of that sphere, dimensionless.
    manifolds    : "none", "unstable", "stable" or "both": invariant
                   manifold branches to compute for a periodic orbit.
    manifold_branches, manifold_time_days : how many departure points
                   along the orbit and how long each branch is followed.
    """
    name: str
    source: str = "state"
    initial_state: list = field(default_factory=lambda: [1.0221, 0.0, -0.1821, 0.0, -0.1018, 0.0])
    period_tu: float = 0.0
    family_name: str = "L2 southern halo"
    family_index: int = 49
    elements: dict = field(default_factory=lambda: dict(ELFO_ELEMENTS))
    centre: str = "moon"
    reference_plane: str = "moon orbit"
    propagation: str = "integrate"
    diameter_m: float = 4.0
    albedo: float = 0.2
    manifolds: str = "none"
    manifold_branches: int = 8
    manifold_time_days: float = 10.0

    kind = "spacecraft"

    def is_periodic(self):
        """True if the spacecraft can be propagated as a periodic orbit."""
        return self.source == "family" or (self.source == "state" and self.period_tu > 0.0)


# Osculating elements of the ESA Lunar Pathfinder style elliptical lunar
# frozen orbit: about 12 hours, e = 0.6, inclination in the mid fifties,
# perilune argument 90 degrees so apolune sits over the south pole.
ELFO_ELEMENTS = {"a_km": 6142.0, "e": 0.6, "i_deg": 57.0, "raan_deg": 0.0,
                 "argp_deg": 90.0, "true_anomaly_deg": 180.0}

# Presets offered by the interface: name -> (centre, reference plane, elements).
ELEMENT_PRESETS = {
    "Elliptical lunar frozen orbit (12 h relay)": ("moon", "moon orbit", dict(ELFO_ELEMENTS)),
    "Low lunar polar orbit (100 km)": ("moon", "moon orbit",
                                       {"a_km": 1837.0, "e": 0.0, "i_deg": 90.0, "raan_deg": 0.0,
                                        "argp_deg": 0.0, "true_anomaly_deg": 0.0}),
    "Geostationary orbit": ("earth", "earth equator",
                            {"a_km": 42164.0, "e": 0.0, "i_deg": 0.0, "raan_deg": 0.0,
                             "argp_deg": 0.0, "true_anomaly_deg": 0.0}),
    "GEO transfer orbit (185 x 35786 km)": ("earth", "earth equator",
                                            {"a_km": 24371.0, "e": 0.7306, "i_deg": 27.0, "raan_deg": 0.0,
                                             "argp_deg": 180.0, "true_anomaly_deg": 0.0}),
    "Low Earth orbit (400 km, 51.6 deg)": ("earth", "earth equator",
                                           {"a_km": 6771.0, "e": 0.0, "i_deg": 51.6, "raan_deg": 0.0,
                                            "argp_deg": 0.0, "true_anomaly_deg": 0.0}),
}


@dataclass
class GroundStation:
    """
    An Earth ground station.

    min_elevation_deg     : the spacecraft must be at least this high
                            above the local horizon to be visible.
    max_sun_elevation_deg : the Sun must be below this elevation for the
                            station to count as dark; -6 is civil,
                            -12 nautical, -18 astronomical twilight.
    """
    name: str
    latitude_deg: float = -33.87
    longitude_deg: float = 151.21
    altitude_km: float = 0.05
    min_elevation_deg: float = 10.0
    max_sun_elevation_deg: float = -12.0

    kind = "ground_station"


@dataclass
class OpticalSensor:
    """
    An optical telescope at a ground station.

    station              : name of the GroundStation it sits on.
    limiting_magnitude   : faintest apparent magnitude it can detect.
    lunar_exclusion_deg  : minimum angle between the line of sight and
                           the Moon, to keep lunar glare out of the field.
                           Note that from Earth an L2 NRHO never gets
                           more than about 10 degrees from the Moon, so
                           values above that give no access at all.
    """
    name: str
    station: str = ""
    limiting_magnitude: float = 18.0
    lunar_exclusion_deg: float = 20.0

    kind = "optical_sensor"


def known_fields(cls, entry):
    """The keys of a saved dictionary that the dataclass still has, so
    files written by older versions of the tool keep loading."""
    names = {f.name for f in fields(cls)}
    return {key: value for key, value in entry.items() if key in names}


OBJECT_CLASSES = {"spacecraft": Spacecraft,
                  "ground_station": GroundStation,
                  "optical_sensor": OpticalSensor}


@dataclass
class Scenario:
    """
    A time span and the objects in it.

    epoch_utc     : UTC ISO-8601 time of scenario time zero.
    duration_days : length of the span in days.
    time_step_s   : spacing of the analysis grid in seconds.  Access
                    windows are resolved to this step.
    """
    name: str = "untitled"
    epoch_utc: str = "2026-01-01T00:00:00"
    duration_days: float = 14.0
    time_step_s: float = 60.0
    spacecraft: list = field(default_factory=list)
    ground_stations: list = field(default_factory=list)
    sensors: list = field(default_factory=list)

    # ---- time grid -------------------------------------------------------

    def time_grid_seconds(self):
        """Analysis times in seconds from the epoch, inclusive of both ends."""
        n_steps = int(np.floor(self.duration_days * crtbp.SECONDS_PER_DAY / self.time_step_s))
        return np.arange(n_steps + 1) * self.time_step_s

    def time_grid_nondim(self):
        """Analysis times in non-dimensional time units (TU) from the epoch."""
        return crtbp.time_to_nondim(self.time_grid_seconds())

    # ---- object tree -----------------------------------------------------

    def all_objects(self):
        """Every object in the scenario, stations before their sensors."""
        return list(self.spacecraft) + list(self.ground_stations) + list(self.sensors)

    def find(self, name):
        """Return the object with this name, or None."""
        for obj in self.all_objects():
            if obj.name == name:
                return obj
        return None

    def sensors_of(self, station_name):
        """Sensors attached to a given station."""
        return [sensor for sensor in self.sensors if sensor.station == station_name]

    def unique_name(self, base):
        """A name not already used in the scenario, e.g. 'Station 2'."""
        if self.find(base) is None:
            return base
        counter = 2
        while self.find(f"{base} {counter}") is not None:
            counter = counter + 1
        return f"{base} {counter}"

    def add(self, obj):
        """Add an object, renaming it if the name is already taken."""
        obj.name = self.unique_name(obj.name)
        if isinstance(obj, Spacecraft):
            self.spacecraft.append(obj)
        elif isinstance(obj, GroundStation):
            self.ground_stations.append(obj)
        elif isinstance(obj, OpticalSensor):
            self.sensors.append(obj)
        else:
            raise TypeError(f"unknown object type {type(obj)}")
        return obj

    def remove(self, name):
        """Remove an object by name.  Removing a station removes its sensors."""
        self.spacecraft = [obj for obj in self.spacecraft if obj.name != name]
        self.sensors = [obj for obj in self.sensors if obj.name != name and obj.station != name]
        self.ground_stations = [obj for obj in self.ground_stations if obj.name != name]

    def rename(self, old_name, new_name):
        """Rename an object and keep sensor-to-station links consistent."""
        obj = self.find(old_name)
        if obj is None:
            return
        obj.name = new_name
        for sensor in self.sensors:
            if sensor.station == old_name:
                sensor.station = new_name

    # ---- serialisation ---------------------------------------------------

    def to_dict(self):
        """Plain dictionary with only JSON-compatible values."""
        return {"name": self.name,
                "epoch_utc": self.epoch_utc,
                "duration_days": self.duration_days,
                "time_step_s": self.time_step_s,
                "spacecraft": [asdict(obj) for obj in self.spacecraft],
                "ground_stations": [asdict(obj) for obj in self.ground_stations],
                "sensors": [asdict(obj) for obj in self.sensors]}

    @classmethod
    def from_dict(cls, data):
        """Rebuild a Scenario from the dictionary produced by to_dict."""
        scenario = cls(name=data.get("name", "untitled"),
                       epoch_utc=data.get("epoch_utc", "2026-01-01T00:00:00"),
                       duration_days=float(data.get("duration_days", 14.0)),
                       time_step_s=float(data.get("time_step_s", 60.0)))
        for entry in data.get("spacecraft", []):
            scenario.spacecraft.append(Spacecraft(**known_fields(Spacecraft, entry)))
        for entry in data.get("ground_stations", []):
            scenario.ground_stations.append(GroundStation(**known_fields(GroundStation, entry)))
        for entry in data.get("sensors", []):
            scenario.sensors.append(OpticalSensor(**known_fields(OpticalSensor, entry)))
        return scenario

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def save(self, path):
        """Write the scenario to a JSON file."""
        with open(path, "w") as handle:
            handle.write(self.to_json())

    @classmethod
    def load(cls, path):
        """Read a scenario from a JSON file."""
        with open(path) as handle:
            return cls.from_json(handle.read())


def example_scenario():
    """
    A ready-made scenario: a spacecraft on the 9:2-type NRHO from the
    family, repeated periodically, observed by a telescope near Sydney.
    """
    scenario = Scenario(name="NRHO from Sydney", epoch_utc="2026-01-01T00:00:00",
                        duration_days=14.0, time_step_s=60.0)
    scenario.add(Spacecraft(name="Gateway-like NRHO", source="family", family_index=49,
                            propagation="periodic", diameter_m=6.0, albedo=0.25))
    scenario.add(GroundStation(name="Sydney", latitude_deg=-33.87, longitude_deg=151.21,
                               altitude_km=0.05, min_elevation_deg=15.0,
                               max_sun_elevation_deg=-12.0))
    scenario.add(OpticalSensor(name="Sydney 0.5 m telescope", station="Sydney",
                               limiting_magnitude=18.5, lunar_exclusion_deg=2.0))
    return scenario
