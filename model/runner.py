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

from engine import access, constraints, crtbp, frames, geometry, manifolds, propagation
from model import orbits
from model.family import DEFAULT_FAMILY_NAME
from model.scenario import OpticalSensor


def as_families(family_or_families):
    """
    Accept either the {name: family} dictionary or, for older callers,
    the bare L2 southern halo family list, and return the dictionary.
    """
    if family_or_families is None:
        return {}
    if isinstance(family_or_families, dict):
        return family_or_families
    return {DEFAULT_FAMILY_NAME: family_or_families}


def spacecraft_trajectory(spacecraft, times_nondim, families=None, epoch_jd=None, ephemeris=None):
    """
    (n, 6) rotating-frame states for one Spacecraft on the grid.

    The initial state comes from model.orbits whatever the source.
    Periodic propagation is used when asked for and the orbit has a
    known period; otherwise the state is integrated.
    """
    families = as_families(families)
    state0 = orbits.initial_state(spacecraft, families, epoch_jd, ephemeris)
    period = orbits.period(spacecraft, families)
    if spacecraft.propagation == "periodic" and period is not None:
        return propagation.propagate_periodic(state0, period, times_nondim)
    return propagation.propagate_state(state0, times_nondim)


def spacecraft_manifolds(spacecraft, families, epoch_jd, ephemeris=None):
    """
    Manifold branches asked for by a periodic spacecraft, as
    {"unstable": [...], "stable": [...]} (either may be missing).
    A linearly stable orbit has none; the engine's ValueError is turned
    into an empty result so the run does not fail.
    """
    orbit = {"state0": orbits.initial_state(spacecraft, families, epoch_jd, ephemeris),
             "period": orbits.period(spacecraft, families)}
    duration = crtbp.time_to_nondim(spacecraft.manifold_time_days * crtbp.SECONDS_PER_DAY)
    kinds = ["unstable", "stable"] if spacecraft.manifolds == "both" else [spacecraft.manifolds]
    result = {}
    for kind in kinds:
        try:
            result[kind] = manifolds.manifold_branches(orbit, kind, n_branches=int(spacecraft.manifold_branches),
                                                       duration=duration)
        except ValueError:
            result[kind] = []
    return result


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
        if sensor.max_range_km > 0.0:
            constraint_list.append(constraints.maximum_range(sensor.max_range_km))
        if sensor.max_slew_rate_deg_s > 0.0:
            constraint_list.append(constraints.maximum_slew_rate(sensor.max_slew_rate_deg_s))
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


def run_scenario(scenario, families=None, extra_constraints=None, ephemeris=None):
    """
    Propagate every spacecraft and evaluate every observer-spacecraft
    pair.

    families          : {name: list of orbit dictionaries} from
                        model.family.load_families (a bare list is taken
                        as the L2 southern halo family)
    extra_constraints : optional list of additional constraint functions
                        (see engine/constraints.py) applied to every pair.
    ephemeris         : optional engine.ephemeris.Ephemeris (from
                        model.ephemeris.load_ephemeris); None uses the
                        mean-longitude sky model

    Returns a dictionary
      times_s, times_nondim, jd : (n,) grid arrays
      trajectories : {spacecraft name: (n, 6)}
      manifolds    : {spacecraft name: {"unstable": [branch, ...],
                                        "stable": [branch, ...]}}
                     (see engine/manifolds.py; only for periodic
                     spacecraft that ask for them)
      stations     : {station name: (n, 3) rotating-frame positions}
      observations : {(observer, spacecraft): {
                          "geometry": GeometrySeries,
                          "constraint_kinds": [str],
                          "constraint_names": [str],
                          "constraint_masks": (n, k) bool,
                          "access": (n,) bool}}
      windows      : {(observer, spacecraft): [(start_s, stop_s), ...]}
      duty_cycle   : {(observer, spacecraft): fraction}
      coverage     : {spacecraft: {"count": (n,) observers with access,
                                   "windows": [(start_s, stop_s), ...]
                                   (any observer), "duty_cycle": fraction}}
    """
    families = as_families(families)
    times_s = scenario.time_grid_seconds()
    times_nondim = scenario.time_grid_nondim()
    jd = frames.julian_dates_for_grid(scenario.epoch_utc, times_s)

    trajectories = {}
    manifold_branches = {}
    for spacecraft in scenario.spacecraft:
        trajectories[spacecraft.name] = spacecraft_trajectory(spacecraft, times_nondim, families, jd[0], ephemeris)
        if spacecraft.manifolds != "none" and orbits.period(spacecraft, families) is not None:
            manifold_branches[spacecraft.name] = spacecraft_manifolds(spacecraft, families, jd[0], ephemeris)

    stations = {}
    for station in scenario.ground_stations:
        position, _ = frames.station_position_rotating(
            station.latitude_deg, station.longitude_deg, station.altitude_km, jd, ephemeris)
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
                spacecraft.diameter_m, spacecraft.albedo, ephemeris=ephemeris)
            masks = access.evaluate_constraints(series, constraint_list)
            passed = np.all(masks, axis=1) if masks.shape[1] > 0 else np.ones(len(series), dtype=bool)
            observations[key] = {"geometry": series,
                                 "constraint_kinds": [c.kind for c in constraint_list],
                                 "constraint_names": [c.name for c in constraint_list],
                                 "constraint_masks": masks,
                                 "access": passed}
            windows[key] = access.windows_from_mask(times_s, passed)
            duty[key] = access.duty_cycle(windows[key], times_s[0], times_s[-1])

    # Multi-station coverage: how many observers see each spacecraft at
    # each step, and the windows in which at least one does.
    coverage = {}
    for spacecraft in scenario.spacecraft:
        masks = [observations[key]["access"] for key in observations if key[1] == spacecraft.name]
        if len(masks) == 0:
            continue
        any_mask = access.coverage_mask(masks, 1)
        coverage[spacecraft.name] = {"count": access.coverage_count(masks),
                                     "windows": access.windows_from_mask(times_s, any_mask),
                                     "duty_cycle": access.duty_cycle(access.windows_from_mask(times_s, any_mask),
                                                                     times_s[0], times_s[-1])}

    return {"times_s": times_s,
            "times_nondim": times_nondim,
            "jd": jd,
            "sky_model": "JPL DE440" if ephemeris is not None else "mean-longitude model",
            "trajectories": trajectories,
            "manifolds": manifold_branches,
            "coverage": coverage,
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
