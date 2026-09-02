"""
Time and frame conversions that connect the idealised CRTBP rotating
frame to the Sun, the Earth's rotation, and ground stations.

The CRTBP rotating frame has no notion of calendar time, Sun, or Earth
spin.  To do access analysis we need all three, so this module adds a
deliberately simple astronomical model on top:

  * The lunar orbit plane is taken as the ecliptic (the real inclination
    of 5.1 degrees is ignored) and the Moon moves uniformly along it at
    its mean longitude.  The rotating frame is the ecliptic frame turned
    about z by the Moon's mean longitude, so the Moon stays on +x.
  * The Sun is infinitely far away in the direction of its apparent
    ecliptic longitude.  Parallax across the Earth-Moon system (about
    0.15 degrees) is ignored.
  * The Earth is a sphere of mean radius 6371 km spinning about an axis
    tilted 23.44 degrees from the ecliptic pole, with Greenwich mean
    sidereal time from the standard linear formula.

These give Sun and station directions to roughly a degree, which is
enough for lighting and elevation constraints in a proof of concept.
Anything needing better should replace this module with an ephemeris.

All angles are radians unless the name says degrees.  Positions returned
in the rotating frame are non-dimensional (LU) and barycentric so they
can be compared directly with spacecraft states from crtbp.py.
"""

from datetime import datetime, timezone

import numpy as np

import crtbp

EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_ND = EARTH_RADIUS_KM / crtbp.LENGTH_UNIT_KM
OBLIQUITY_RAD = np.radians(23.4393)
JD_J2000 = 2451545.0


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

def julian_date(iso_utc):
    """Julian date of a UTC ISO-8601 string such as 2026-01-01T00:00:00."""
    moment = datetime.fromisoformat(iso_utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds_since_j2000 = (moment - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds()
    return JD_J2000 + seconds_since_j2000 / crtbp.SECONDS_PER_DAY


def julian_dates_for_grid(epoch_utc, times_seconds):
    """Julian dates for an array of seconds past the epoch."""
    return julian_date(epoch_utc) + np.asarray(times_seconds) / crtbp.SECONDS_PER_DAY


# --------------------------------------------------------------------------
# Sun, Moon, Earth rotation
# --------------------------------------------------------------------------

def sun_ecliptic_longitude(jd):
    """
    Apparent ecliptic longitude of the Sun, radians.  Low-precision
    formula (mean longitude plus the equation of centre), good to about
    0.01 degrees, which is far better than the rest of this model.
    """
    days = jd - JD_J2000
    mean_longitude = np.radians(280.460 + 0.9856474 * days)
    mean_anomaly = np.radians(357.528 + 0.9856003 * days)
    return mean_longitude + np.radians(1.915) * np.sin(mean_anomaly) + np.radians(0.020) * np.sin(2.0 * mean_anomaly)


def moon_mean_longitude(jd):
    """
    Mean ecliptic longitude of the Moon, radians.  This is the uniformly
    increasing angle that defines the rotating frame; the true Moon can
    lead or lag it by up to about 6 degrees because of the eccentricity
    the CRTBP ignores.
    """
    days = jd - JD_J2000
    return np.radians(218.316 + 13.176396 * days)


def greenwich_sidereal_angle(jd):
    """Greenwich mean sidereal time as an angle, radians."""
    days = jd - JD_J2000
    return np.radians(280.46061837 + 360.98564736629 * days)


# --------------------------------------------------------------------------
# Rotations, written out by component so they work on arrays of angles
# --------------------------------------------------------------------------

def rotate_about_z(x, y, z, angle):
    """Rotate the vector (x, y, z) by `angle` about the z-axis (right-handed)."""
    x_new = x * np.cos(angle) - y * np.sin(angle)
    y_new = x * np.sin(angle) + y * np.cos(angle)
    return x_new, y_new, z


def rotate_about_x(x, y, z, angle):
    """Rotate the vector (x, y, z) by `angle` about the x-axis (right-handed)."""
    y_new = y * np.cos(angle) - z * np.sin(angle)
    z_new = y * np.sin(angle) + z * np.cos(angle)
    return x, y_new, z_new


def ecliptic_to_rotating(x, y, z, jd):
    """
    Turn an ecliptic-frame vector into the CRTBP rotating frame at the
    given Julian date(s).  The rotating x-axis points at the Moon, which
    sits at its mean longitude, so this is a rotation about z by minus
    that longitude.  Origins are not shifted (use for directions, or add
    the barycentric offset separately).
    """
    return rotate_about_z(x, y, z, -moon_mean_longitude(jd))


def sun_direction_rotating(jd):
    """
    Unit vector toward the Sun in the rotating frame, shape (n, 3) for an
    array of Julian dates.  In this frame the Sun goes round once per
    synodic month, backwards.
    """
    jd = np.atleast_1d(jd)
    longitude = sun_ecliptic_longitude(jd)
    x, y, z = ecliptic_to_rotating(np.cos(longitude), np.sin(longitude), np.zeros_like(longitude), jd)
    return np.column_stack([x, y, z])


# --------------------------------------------------------------------------
# Ground stations
# --------------------------------------------------------------------------

def station_position_rotating(latitude_deg, longitude_deg, altitude_km, jd):
    """
    Barycentric rotating-frame position (LU) and local vertical of a
    ground station at the given Julian date(s).

    The chain is: Earth-fixed spherical coordinates, spin by Greenwich
    sidereal angle into the equatorial inertial frame, tilt by the
    obliquity into the ecliptic frame, turn into the rotating frame, then
    shift from the Earth's centre to the barycentre.

    Returns (position, up) with shapes (n, 3).  `up` is the geocentric
    vertical, which differs from the geodetic vertical by at most 0.2
    degrees on a spherical-Earth model like this one.
    """
    jd = np.atleast_1d(jd)
    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    radius_nd = (EARTH_RADIUS_KM + altitude_km) / crtbp.LENGTH_UNIT_KM

    # Earth-fixed direction of the station.
    x_fixed = np.cos(latitude) * np.cos(longitude) * np.ones_like(jd)
    y_fixed = np.cos(latitude) * np.sin(longitude) * np.ones_like(jd)
    z_fixed = np.sin(latitude) * np.ones_like(jd)

    # Earth spin: Earth-fixed to equatorial inertial.
    x_eq, y_eq, z_eq = rotate_about_z(x_fixed, y_fixed, z_fixed, greenwich_sidereal_angle(jd))

    # Equatorial to ecliptic: tilt back by the obliquity about the
    # equinox direction (the shared x-axis).
    x_ecl, y_ecl, z_ecl = rotate_about_x(x_eq, y_eq, z_eq, -OBLIQUITY_RAD)

    # Ecliptic to rotating frame.
    x_rot, y_rot, z_rot = ecliptic_to_rotating(x_ecl, y_ecl, z_ecl, jd)

    up = np.column_stack([x_rot, y_rot, z_rot])
    position = crtbp.earth_position() + radius_nd * up
    return position, up
