"""
Observer-to-target geometry on a time grid.

`observation_geometry` turns a station, a spacecraft trajectory and the
Julian dates of the grid into a GeometrySeries: one array per quantity.
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
                            apparent_magnitude=float(self.apparent_magnitude[index]))


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
                         spacecraft_states, times_s, jd, diameter_m, albedo, mu=crtbp.MU):
    """
    Geometry of one ground station observing one spacecraft over a grid.

    spacecraft_states : (n, 6) rotating-frame states in LU, LU/TU
    times_s           : (n,) seconds past the epoch (carried through)
    jd                : (n,) Julian dates of the grid
    diameter_m, albedo: diffuse-sphere parameters for the magnitude

    Returns a GeometrySeries.
    """
    points = propagation.fixed_points(mu)
    sun_direction = frames.sun_direction_rotating(jd)
    station_position, station_up = frames.station_position_rotating(
        station_latitude_deg, station_longitude_deg, station_altitude_km, jd)

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
                          apparent_magnitude=photometry.apparent_magnitude(range_km, diameter_m, albedo, phase_angle))
