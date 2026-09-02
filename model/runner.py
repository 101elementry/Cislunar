"""
Run a Scenario through the engine.

This is the only place that understands both the model objects and the
engine's array interface.  It resolves each spacecraft to a trajectory,
builds the constraint list for each observer from the station and
sensor settings, and hands everything to the engine.  Results come
back as plain arrays and lists keyed by object names, so app/ and
scripts/ can use them identically.
"""

import numpy as np

from engine import access, constraints, frames, geometry, propagation
from model.scenario import OpticalSensor


def spacecraft_trajectory(spacecraft, times_nondim, family=None):
    """
    (n, 6) rotating-frame states for one Spacecraft on the grid.

    "family" spacecraft take their initial state (and, for periodic
    propagation, their period) from the family list; "state" spacecraft
    integrate their own initial state.
    """
    if spacecraft.source == "family":
        if family is None:
            raise ValueError(f"spacecraft {spacecraft.name!r} needs the halo family")
        orbit = family[int(spacecraft.family_index)]
        if spacecraft.propagation == "periodic":
            return propagation.propagate_periodic(orbit["state0"], orbit["period"], times_nondim)
        return propagation.propagate_state(orbit["state0"], times_nondim)
    return propagation.propagate_state(spacecraft.initial_state, times_nondim)


def constraints_for(station, sensor):
    """
    The access constraints implied by a station and an optional sensor.
    Order is only cosmetic: the engine treats the list as a set.
    """
    constraint_list = [constraints.elevation_cutoff(station.min_elevation_deg),
                       constraints.station_darkness(station.max_sun_elevation_deg),
                       constraints.target_illumination()]
    if sensor is not None:
        constraint_list.append(constraints.limiting_magnitude(sensor.limiting_magnitude))
        constraint_list.append(constraints.lunar_exclusion(sensor.lunar_exclusion_deg))
    return constraint_list


def observers(scenario):
    """
    (observer name, station, sensor) for every observer.  A station
    without sensors observes on its own with sensor = None, so geometry
    only constraints still produce windows.
    """
    result = []
    for station in scenario.ground_stations:
        sensors = scenario.sensors_of(station.name)
        if len(sensors) == 0:
            result.append((station.name, station, None))
        for sensor in sensors:
            result.append((sensor.name, station, sensor))
    return result


def run_scenario(scenario, family=None, extra_constraints=None):
    """
    Propagate every spacecraft and evaluate every observer-spacecraft
    pair.

    extra_constraints : optional list of additional constraint functions
                        (see engine/constraints.py) applied to every pair.

    Returns a dictionary
      times_s, times_nondim, jd : (n,) grid arrays
      trajectories : {spacecraft name: (n, 6)}
      stations     : {station name: (n, 3) rotating-frame positions}
      observations : {(observer, spacecraft): {
                          "geometry": GeometrySeries,
                          "constraint_kinds": [str],
                          "constraint_names": [str],
                          "constraint_masks": (n, k) bool,
                          "access": (n,) bool}}
      windows      : {(observer, spacecraft): [(start_s, stop_s), ...]}
      duty_cycle   : {(observer, spacecraft): fraction}
    """
    times_s = scenario.time_grid_seconds()
    times_nondim = scenario.time_grid_nondim()
    jd = frames.julian_dates_for_grid(scenario.epoch_utc, times_s)

    trajectories = {}
    for spacecraft in scenario.spacecraft:
        trajectories[spacecraft.name] = spacecraft_trajectory(spacecraft, times_nondim, family)

    stations = {}
    for station in scenario.ground_stations:
        position, _ = frames.station_position_rotating(
            station.latitude_deg, station.longitude_deg, station.altitude_km, jd)
        stations[station.name] = position

    observations = {}
    windows = {}
    duty = {}
    for observer_name, station, sensor in observers(scenario):
        constraint_list = constraints_for(station, sensor) + list(extra_constraints or [])
        for spacecraft in scenario.spacecraft:
            key = (observer_name, spacecraft.name)
            series = geometry.observation_geometry(
                station.latitude_deg, station.longitude_deg, station.altitude_km,
                trajectories[spacecraft.name], times_s, jd,
                spacecraft.diameter_m, spacecraft.albedo)
            masks = access.evaluate_constraints(series, constraint_list)
            passed = np.all(masks, axis=1) if masks.shape[1] > 0 else np.ones(len(series), dtype=bool)
            observations[key] = {"geometry": series,
                                 "constraint_kinds": [c.kind for c in constraint_list],
                                 "constraint_names": [c.name for c in constraint_list],
                                 "constraint_masks": masks,
                                 "access": passed}
            windows[key] = access.windows_from_mask(times_s, passed)
            duty[key] = access.duty_cycle(windows[key], times_s[0], times_s[-1])

    return {"times_s": times_s,
            "times_nondim": times_nondim,
            "jd": jd,
            "trajectories": trajectories,
            "stations": stations,
            "observations": observations,
            "windows": windows,
            "duty_cycle": duty}


def observer_settings(scenario, observer_name):
    """(station, sensor or None) for an observer name; display code uses this for thresholds."""
    for name, station, sensor in observers(scenario):
        if name == observer_name:
            return station, sensor
    return None, None
