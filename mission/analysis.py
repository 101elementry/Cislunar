"""
Propagation and observation analysis for a Scenario.

Nothing here knows about Dash.  Every function takes plain arrays or
scenario objects and returns arrays or dictionaries, so the same code
runs from a script.  See scripts/run_scenario_from_script.py.

Units: positions in LU, times in seconds past the epoch unless a name
says otherwise, angles in degrees at the interface of each function and
radians inside.
"""

import os

import numpy as np

import crtbp
import corrector
from mission import frames

FAMILY_FILE = os.path.join("output", "halo_family.npz")

# Apparent visual magnitude of the Sun, used as the reference for the
# reflected-light magnitude of the spacecraft.
SUN_APPARENT_MAGNITUDE = -26.74


# --------------------------------------------------------------------------
# Halo family
# --------------------------------------------------------------------------

def load_family(path=FAMILY_FILE):
    """
    The L2 southern halo family from validate.py.  Built on the spot if
    the file is missing (about twenty seconds).
    """
    if os.path.exists(path):
        data = np.load(path)
        return corrector.arrays_to_family({key: data[key] for key in data.files})
    family = corrector.build_l2_southern_family(
        stop_perilune_radius=crtbp.length_to_nondim(1800.0),
        initial_step=0.004, max_step=0.01, max_vy_change=0.03, verbose=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **corrector.family_to_arrays(family))
    return family


def family_summary(family):
    """One short label per family member for menus and tables."""
    labels = []
    for index, orbit in enumerate(family):
        labels.append(f"{index}: T = {crtbp.time_to_days(orbit['period']):.2f} d, "
                      f"perilune {crtbp.length_to_km(orbit['perilune_radius']):,.0f} km, "
                      f"C = {orbit['jacobi']:.4f}")
    return labels


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------

def propagate_spacecraft(spacecraft, times_nondim, family=None):
    """
    Rotating-frame states of a spacecraft on the analysis grid.

    times_nondim : array of times in TU past the epoch.
    Returns an array of shape (n, 6).

    For source = "family" with propagation = "periodic" the converged
    orbit is sampled at time modulo the period, so the spacecraft stays
    on the periodic orbit for the whole span exactly as a perfectly
    station-kept vehicle would.  Otherwise the initial state is
    integrated with the full equations of motion, which for an unstable
    halo will eventually leave the orbit.
    """
    times_nondim = np.asarray(times_nondim, dtype=float)

    if spacecraft.source == "family":
        if family is None:
            family = load_family()
        orbit = family[int(spacecraft.family_index)]
        initial_state = np.asarray(orbit["state0"], dtype=float)
        if spacecraft.propagation == "periodic":
            period = float(orbit["period"])
            one_period = crtbp.propagate(initial_state, period, dense_output=True)
            phase_times = np.mod(times_nondim, period)
            return one_period.sol(phase_times).T
    else:
        initial_state = np.asarray(spacecraft.initial_state, dtype=float)

    if times_nondim[-1] == 0.0:
        return np.tile(initial_state, (len(times_nondim), 1))
    sol = crtbp.propagate(initial_state, times_nondim[-1], t_eval=times_nondim)
    return sol.y.T


def fixed_geometry():
    """Positions that do not move in the rotating frame."""
    libration = crtbp.collinear_libration_points()
    return {"earth": crtbp.earth_position(),
            "moon": crtbp.moon_position(),
            "L1": np.array([libration["L1"], 0.0, 0.0]),
            "L2": np.array([libration["L2"], 0.0, 0.0])}


# --------------------------------------------------------------------------
# Observation geometry
# --------------------------------------------------------------------------

def unit_vectors(vectors):
    """Normalise each row of an (n, 3) array."""
    lengths = np.linalg.norm(vectors, axis=1)
    return vectors / lengths[:, np.newaxis], lengths


def angle_between_deg(a, b):
    """Angle in degrees between the rows of two (n, 3) unit-vector arrays."""
    cosine = np.sum(a * b, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def elevation_deg(line_of_sight_unit, up_unit):
    """
    Elevation of a target above the local horizon, degrees.  Ninety
    degrees minus the angle between the local vertical and the line of
    sight, written as an arcsine of the vertical component.
    """
    vertical_component = np.sum(line_of_sight_unit * up_unit, axis=1)
    return np.degrees(np.arcsin(np.clip(vertical_component, -1.0, 1.0)))


def in_cylindrical_shadow(positions, sun_direction, body_position, body_radius):
    """
    True where a point lies inside the shadow cylinder of a body.

    The cylinder has the body's radius and extends from the body away
    from the Sun.  Umbra and penumbra are not distinguished; for a body
    as far from the Sun as the Earth the cone half-angle is a quarter of
    a degree, so the cylinder is a fine approximation at cislunar ranges.
    """
    relative = positions - body_position
    # Component along the Sun direction: negative means the point is on
    # the night side of the body.
    along_sun = np.sum(relative * sun_direction, axis=1)
    perpendicular = relative - along_sun[:, np.newaxis] * sun_direction
    perpendicular_distance = np.linalg.norm(perpendicular, axis=1)
    return (along_sun < 0.0) & (perpendicular_distance < body_radius)


def phase_angle_deg(sun_direction, observer_direction_from_target):
    """
    Sun-target-observer phase angle in degrees.  Zero means the observer
    sees a fully lit disc, 180 means the target is backlit.
    """
    return angle_between_deg(sun_direction, observer_direction_from_target)


def diffuse_sphere_phase_function(phase_angle_deg_array):
    """
    Fraction of the full-phase brightness a Lambertian sphere shows at a
    given phase angle:  [(pi - alpha) cos(alpha) + sin(alpha)] / pi.
    Equals 1 at alpha = 0 and 0 at alpha = 180 degrees.
    """
    alpha = np.radians(phase_angle_deg_array)
    return ((np.pi - alpha) * np.cos(alpha) + np.sin(alpha)) / np.pi


def apparent_magnitude(range_km, diameter_m, albedo, phase_angle_deg_array):
    """
    Apparent visual magnitude of a diffuse sphere lit by the Sun.

    The sphere reflects a fraction of the sunlight falling on it; at zero
    phase the flux ratio to the Sun seen from the same place is
        (2/3) * albedo * (radius / range)^2,
    and the phase function scales that for other geometries.  Inverse
    square falloff is the (radius / range)^2 factor.  The Sun is assumed
    at 1 au from the spacecraft, which is true to a quarter of a percent
    in cislunar space.
    """
    radius_m = 0.5 * diameter_m
    range_m = np.asarray(range_km) * 1000.0
    flux_ratio = (2.0 / 3.0) * albedo * (radius_m / range_m) ** 2 \
        * diffuse_sphere_phase_function(phase_angle_deg_array)
    return SUN_APPARENT_MAGNITUDE - 2.5 * np.log10(flux_ratio)


def observe(station, sensor, spacecraft, spacecraft_states, jd):
    """
    Everything a ground station and sensor see of one spacecraft over the
    analysis grid.

    station           : GroundStation
    sensor            : OpticalSensor or None (then no magnitude or lunar
                        exclusion constraints are applied)
    spacecraft        : Spacecraft (for diameter and albedo)
    spacecraft_states : (n, 6) rotating-frame states
    jd                : (n,) Julian dates of the grid

    Returns a dictionary of (n,) arrays:
      elevation_deg, sun_elevation_deg, range_km, lunar_separation_deg,
      phase_angle_deg, apparent_magnitude, plus the boolean constraints
      above_horizon, station_dark, spacecraft_lit, clear_of_moon,
      bright_enough and their conjunction `access`.
    """
    geometry = fixed_geometry()
    sun_direction = frames.sun_direction_rotating(jd)
    station_position, station_up = frames.station_position_rotating(
        station.latitude_deg, station.longitude_deg, station.altitude_km, jd)

    positions = spacecraft_states[:, :3]
    line_of_sight_unit, range_nd = unit_vectors(positions - station_position)
    range_km = crtbp.length_to_km(range_nd)

    elevation = elevation_deg(line_of_sight_unit, station_up)
    sun_elevation = elevation_deg(sun_direction, station_up)

    moon_direction_unit, _ = unit_vectors(geometry["moon"] - station_position)
    lunar_separation = angle_between_deg(line_of_sight_unit, moon_direction_unit)

    shadowed = (in_cylindrical_shadow(positions, sun_direction, geometry["earth"], frames.EARTH_RADIUS_ND)
                | in_cylindrical_shadow(positions, sun_direction, geometry["moon"], crtbp.MOON_RADIUS_ND))

    phase_angle = phase_angle_deg(sun_direction, -line_of_sight_unit)
    magnitude = apparent_magnitude(range_km, spacecraft.diameter_m, spacecraft.albedo, phase_angle)

    above_horizon = elevation >= station.min_elevation_deg
    station_dark = sun_elevation <= station.max_sun_elevation_deg
    spacecraft_lit = ~shadowed
    if sensor is None:
        clear_of_moon = np.ones_like(above_horizon)
        bright_enough = np.ones_like(above_horizon)
    else:
        clear_of_moon = lunar_separation >= sensor.lunar_exclusion_deg
        bright_enough = magnitude <= sensor.limiting_magnitude

    return {"elevation_deg": elevation,
            "sun_elevation_deg": sun_elevation,
            "range_km": range_km,
            "lunar_separation_deg": lunar_separation,
            "phase_angle_deg": phase_angle,
            "apparent_magnitude": magnitude,
            "above_horizon": above_horizon,
            "station_dark": station_dark,
            "spacecraft_lit": spacecraft_lit,
            "clear_of_moon": clear_of_moon,
            "bright_enough": bright_enough,
            "access": above_horizon & station_dark & spacecraft_lit & clear_of_moon & bright_enough}


# --------------------------------------------------------------------------
# Windows and duty cycle
# --------------------------------------------------------------------------

def windows_from_mask(times_seconds, mask):
    """
    Contiguous runs where `mask` is True, as a list of (start, stop)
    in seconds.  Edges are resolved to the grid spacing: a window is
    taken to start at the first True sample and end at the last one, so
    a single True sample has zero duration.
    """
    mask = np.asarray(mask, dtype=bool)
    windows = []
    if not mask.any():
        return windows
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0] - 1
    for start_index, stop_index in zip(starts, stops):
        windows.append((float(times_seconds[start_index]), float(times_seconds[stop_index])))
    return windows


def duty_cycle(windows, span_start_seconds, span_stop_seconds):
    """Fraction of [span_start, span_stop] covered by the windows."""
    covered = 0.0
    for start, stop in windows:
        overlap = min(stop, span_stop_seconds) - max(start, span_start_seconds)
        if overlap > 0.0:
            covered = covered + overlap
    span = span_stop_seconds - span_start_seconds
    if span <= 0.0:
        return 0.0
    return covered / span


# --------------------------------------------------------------------------
# Whole scenario
# --------------------------------------------------------------------------

def run_scenario(scenario, family=None):
    """
    Propagate every spacecraft and evaluate every (sensor, spacecraft)
    pair.  Stations without sensors are evaluated with sensor = None so
    that pure geometry access still appears.

    Returns a dictionary:
      times_s          : (n,) seconds past the epoch
      times_nondim     : (n,) TU
      jd               : (n,) Julian dates
      trajectories     : {spacecraft name: (n, 6) states}
      stations         : {station name: (n, 3) positions}
      observations     : {(observer name, spacecraft name): observe() output}
      windows          : {(observer name, spacecraft name): [(start_s, stop_s), ...]}
      duty_cycle       : {(observer name, spacecraft name): fraction}
    """
    if family is None and any(s.source == "family" for s in scenario.spacecraft):
        family = load_family()

    times_s = scenario.time_grid_seconds()
    times_nondim = scenario.time_grid_nondim()
    jd = frames.julian_dates_for_grid(scenario.epoch_utc, times_s)

    trajectories = {}
    for spacecraft in scenario.spacecraft:
        trajectories[spacecraft.name] = propagate_spacecraft(spacecraft, times_nondim, family)

    stations = {}
    for station in scenario.ground_stations:
        position, _ = frames.station_position_rotating(
            station.latitude_deg, station.longitude_deg, station.altitude_km, jd)
        stations[station.name] = position

    observations = {}
    windows = {}
    duty = {}
    for station in scenario.ground_stations:
        observers = scenario.sensors_of(station.name)
        if len(observers) == 0:
            observers = [None]
        for sensor in observers:
            observer_name = station.name if sensor is None else sensor.name
            for spacecraft in scenario.spacecraft:
                key = (observer_name, spacecraft.name)
                result = observe(station, sensor, spacecraft, trajectories[spacecraft.name], jd)
                observations[key] = result
                windows[key] = windows_from_mask(times_s, result["access"])
                duty[key] = duty_cycle(windows[key], times_s[0], times_s[-1])

    return {"times_s": times_s,
            "times_nondim": times_nondim,
            "jd": jd,
            "trajectories": trajectories,
            "stations": stations,
            "observations": observations,
            "windows": windows,
            "duty_cycle": duty}
