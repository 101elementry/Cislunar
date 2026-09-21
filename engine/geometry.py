"""
Observer-to-target geometry on a time grid.

`observation_geometry` turns a station, a spacecraft trajectory and the
Julian dates of the grid into a GeometrySeries: one array per quantity.
`space_observation_geometry` does the same for an observer that is
itself a spacecraft (a camera on a chaser looking at a target).
`GeometrySeries.at(index)` gives the scalar StepGeometry for one time
step, which is what the access constraints consume.
"""

from dataclasses import dataclass

import numpy as np

from engine import crtbp, frames, photometry, propagation


@dataclass
class StepGeometry:
    """
    Everything known about one observer-target pair at one instant.
    Angles in degrees, range in kilometres.  This is the argument of
    every access constraint.
    """
    time_s: float
    elevation_deg: float
    sun_elevation_deg: float
    range_km: float
    lunar_separation_deg: float
    phase_angle_deg: float
    in_shadow: bool
    apparent_magnitude: float
    los_rate_deg_s: float
    # Angles at the observer between the line of sight and the Sun and
    # the Earth.  A camera cannot look close to the Sun, and a bright
    # Earth behind the target washes it out.  For an observer in space
    # elevation_deg and sun_elevation_deg have no meaning and are NaN.
    sun_separation_deg: float = float("nan")
    earth_separation_deg: float = float("nan")


@dataclass
class GeometrySeries:
    """The same quantities as StepGeometry, as arrays over the grid."""
    time_s: np.ndarray
    elevation_deg: np.ndarray
    sun_elevation_deg: np.ndarray
    range_km: np.ndarray
    lunar_separation_deg: np.ndarray
    phase_angle_deg: np.ndarray
    in_shadow: np.ndarray
    apparent_magnitude: np.ndarray
    los_rate_deg_s: np.ndarray
    sun_separation_deg: np.ndarray = None
    earth_separation_deg: np.ndarray = None

    def __post_init__(self):
        # Series built before these two angles existed carry NaN for them.
        if self.sun_separation_deg is None:
            self.sun_separation_deg = np.full(len(self.time_s), np.nan)
        if self.earth_separation_deg is None:
            self.earth_separation_deg = np.full(len(self.time_s), np.nan)

    def __len__(self):
        return len(self.time_s)

    def at(self, index):
        """Scalar geometry for one time step."""
        return StepGeometry(time_s=float(self.time_s[index]),
                            elevation_deg=float(self.elevation_deg[index]),
                            sun_elevation_deg=float(self.sun_elevation_deg[index]),
                            range_km=float(self.range_km[index]),
                            lunar_separation_deg=float(self.lunar_separation_deg[index]),
                            phase_angle_deg=float(self.phase_angle_deg[index]),
                            in_shadow=bool(self.in_shadow[index]),
                            apparent_magnitude=float(self.apparent_magnitude[index]),
                            los_rate_deg_s=float(self.los_rate_deg_s[index]),
                            sun_separation_deg=float(self.sun_separation_deg[index]),
                            earth_separation_deg=float(self.earth_separation_deg[index]))


# --------------------------------------------------------------------------
# Vector helpers
# --------------------------------------------------------------------------

def unit_vectors(vectors):
    """Normalise each row of an (n, 3) array; returns (units, lengths)."""
    lengths = np.linalg.norm(vectors, axis=1)
    return vectors / lengths[:, np.newaxis], lengths


def angle_between_deg(a, b):
    """Angle in degrees between the rows of two (n, 3) unit-vector arrays."""
    cosine = np.sum(a * b, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def elevation_deg(direction_unit, up_unit):
    """
    Elevation of a direction above the local horizon, degrees: the
    arcsine of its component along the local vertical.
    """
    vertical_component = np.sum(direction_unit * up_unit, axis=1)
    return np.degrees(np.arcsin(np.clip(vertical_component, -1.0, 1.0)))


def angular_rate_deg_s(unit_directions, times_s):
    """
    Rate at which a unit direction turns, degrees per second, from the
    angle between neighbouring samples divided by the time between
    them (central difference inside the grid, one-sided at the ends).
    This is the slew rate a telescope must follow to track the target.
    """
    n = len(times_s)
    rate = np.zeros(n)
    if n < 2:
        return rate
    angles = angle_between_deg(unit_directions[:-1], unit_directions[1:])
    intervals = np.diff(np.asarray(times_s, dtype=float))
    step_rate = angles / intervals
    rate[0] = step_rate[0]
    rate[-1] = step_rate[-1]
    rate[1:-1] = 0.5 * (step_rate[:-1] + step_rate[1:])
    return rate


def in_cylindrical_shadow(positions, sun_direction, body_position, body_radius):
    """
    True where a point lies inside a body's shadow cylinder: on the
    night side of the body and closer to the anti-Sun axis than the body
    radius.  Umbra and penumbra are not distinguished; the real cone
    half-angle is a quarter of a degree, negligible at these ranges.
    """
    relative = positions - body_position
    along_sun = np.sum(relative * sun_direction, axis=1)
    perpendicular = relative - along_sun[:, np.newaxis] * sun_direction
    perpendicular_distance = np.linalg.norm(perpendicular, axis=1)
    return (along_sun < 0.0) & (perpendicular_distance < body_radius)


# --------------------------------------------------------------------------
# The geometry series
# --------------------------------------------------------------------------

def observation_geometry(station_latitude_deg, station_longitude_deg, station_altitude_km,
                         spacecraft_states, times_s, jd, diameter_m, albedo, mu=crtbp.MU, ephemeris=None):
    """
    Geometry of one ground station observing one spacecraft over a grid.

    spacecraft_states : (n, 6) rotating-frame states in LU, LU/TU
    times_s           : (n,) seconds past the epoch (carried through)
    jd                : (n,) Julian dates of the grid
    diameter_m, albedo: diffuse-sphere parameters for the magnitude
    ephemeris         : optional engine.ephemeris.Ephemeris for the real
                        Sun, Moon and Earth orientation (see frames.py)

    Returns a GeometrySeries.
    """
    points = propagation.fixed_points(mu)
    sun_direction = frames.sun_direction_rotating(jd, ephemeris)
    station_position, station_up = frames.station_position_rotating(
        station_latitude_deg, station_longitude_deg, station_altitude_km, jd, ephemeris)

    positions = spacecraft_states[:, :3]
    line_of_sight_unit, range_nd = unit_vectors(positions - station_position)

    moon_direction_unit, _ = unit_vectors(points["moon"] - station_position)

    shadowed = (in_cylindrical_shadow(positions, sun_direction, points["earth"], frames.EARTH_RADIUS_ND)
                | in_cylindrical_shadow(positions, sun_direction, points["moon"], crtbp.MOON_RADIUS_ND))

    # Phase angle is measured at the target between the Sun and the
    # observer, so the observer direction is the reversed line of sight.
    phase_angle = angle_between_deg(sun_direction, -line_of_sight_unit)
    range_km = crtbp.length_to_km(range_nd)

    return GeometrySeries(time_s=np.asarray(times_s, dtype=float),
                          elevation_deg=elevation_deg(line_of_sight_unit, station_up),
                          sun_elevation_deg=elevation_deg(sun_direction, station_up),
                          range_km=range_km,
                          lunar_separation_deg=angle_between_deg(line_of_sight_unit, moon_direction_unit),
                          phase_angle_deg=phase_angle,
                          in_shadow=shadowed,
                          apparent_magnitude=photometry.apparent_magnitude(range_km, diameter_m, albedo, phase_angle),
                          los_rate_deg_s=angular_rate_deg_s(line_of_sight_unit, times_s),
                          sun_separation_deg=angle_between_deg(line_of_sight_unit, sun_direction),
                          earth_separation_deg=np.full(len(positions), np.nan))


def space_observation_geometry(observer_states, target_states, times_s, jd, diameter_m, albedo,
                               mu=crtbp.MU, ephemeris=None):
    """
    Geometry of one spacecraft observing another over a grid: a camera
    on a chaser looking at a target.

    observer_states, target_states : (n, 6) rotating-frame states on the
                                     same grid, LU and LU/TU
    times_s, jd                    : (n,) seconds past epoch, Julian dates
    diameter_m, albedo             : diffuse-sphere parameters of the target
    ephemeris                      : optional engine.ephemeris.Ephemeris

    The line of sight runs from the observer to the target.  The Sun is
    so far away that its direction is the same from both spacecraft.
    There is no horizon and no night in space, so elevation_deg and
    sun_elevation_deg are NaN; what limits a camera instead is how close
    to the Sun, the Moon or the Earth it has to point, and whether the
    target is lit.

    The line-of-sight rate is measured in the rotating frame, which
    turns once in 27.3 days (0.00015 deg/s); next to the rates of a
    close approach that is negligible.

    Returns a GeometrySeries.
    """
    points = propagation.fixed_points(mu)
    sun_direction = frames.sun_direction_rotating(jd, ephemeris)

    observer_positions = observer_states[:, :3]
    target_positions = target_states[:, :3]
    line_of_sight_unit, range_nd = unit_vectors(target_positions - observer_positions)
    moon_direction_unit, _ = unit_vectors(points["moon"] - observer_positions)
    earth_direction_unit, _ = unit_vectors(points["earth"] - observer_positions)

    shadowed = (in_cylindrical_shadow(target_positions, sun_direction, points["earth"], frames.EARTH_RADIUS_ND)
                | in_cylindrical_shadow(target_positions, sun_direction, points["moon"], crtbp.MOON_RADIUS_ND))

    # Phase angle at the target between the Sun and the observer.
    phase_angle = angle_between_deg(sun_direction, -line_of_sight_unit)
    range_km = crtbp.length_to_km(range_nd)
    nan = np.full(len(target_positions), np.nan)

    return GeometrySeries(time_s=np.asarray(times_s, dtype=float),
                          elevation_deg=nan,
                          sun_elevation_deg=nan.copy(),
                          range_km=range_km,
                          lunar_separation_deg=angle_between_deg(line_of_sight_unit, moon_direction_unit),
                          phase_angle_deg=phase_angle,
                          in_shadow=shadowed,
                          apparent_magnitude=photometry.apparent_magnitude(range_km, diameter_m, albedo, phase_angle),
                          los_rate_deg_s=angular_rate_deg_s(line_of_sight_unit, times_s),
                          sun_separation_deg=angle_between_deg(line_of_sight_unit, sun_direction),
                          earth_separation_deg=angle_between_deg(line_of_sight_unit, earth_direction_unit))
