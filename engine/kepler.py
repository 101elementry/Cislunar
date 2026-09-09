"""
Keplerian elements about the Moon or the Earth and their conversion to
and from the CRTBP rotating-frame state.

Lunar orbiters (a polar mapping orbit, a relay on an elliptical lunar
frozen orbit) and Earth orbiters (a GEO parking orbit, a phasing orbit
before a rendezvous) are naturally described by two-body elements about
their central body.  The CRTBP integrates them with the other body's
pull included, which is what makes a lunar frozen orbit frozen, so the
only work here is the change of frame.

Conventions
  * By default the reference plane is the Moon's orbit plane, which is
    the xy-plane of the rotating frame, and the reference direction for
    the right ascension of the ascending node is the rotating +x axis
    at the instant of conversion (Earth to Moon direction).  A different
    reference frame, such as the Earth's equator, is handled by passing
    the 3x3 rotation from that frame into the rotating frame; frames.py
    builds it for a given date.
  * The Moon's equator is tilted 6.7 degrees from its orbit plane; that
    tilt and the oblateness of both bodies are not modelled.
  * Elements are osculating: the two-body values the state would have
    if the other body were switched off at that instant.
  * The gravitational parameter of the central body in non-dimensional
    units is mu for the Moon and 1 - mu for the Earth (LU^3 / TU^2).
"""

import numpy as np

from engine import crtbp
from engine.crtbp import MU


def rotation_perifocal_to_reference(i, raan, argp):
    """
    3x3 matrix taking perifocal coordinates (x toward perilune, z along
    the angular momentum) to the reference frame, the classical 3-1-3
    sequence: rotate by argp about z, then i about x, then raan about z.
    Angles in radians.
    """
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_argp = np.cos(argp)
    sin_argp = np.sin(argp)
    return np.array([
        [cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
         -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
         -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
         -cos_raan * sin_i],
        [sin_argp * sin_i,
         cos_argp * sin_i,
         cos_i]])


def central_body(centre, mu=MU):
    """(position, gravitational parameter) of "moon" or "earth", non-dimensional."""
    if centre == "moon":
        return crtbp.moon_position(mu), mu
    if centre == "earth":
        return crtbp.earth_position(mu), 1.0 - mu
    raise ValueError(f"centre must be 'moon' or 'earth', not {centre!r}")


def keplerian_to_state_rotating(a_km, e, i_deg, raan_deg, argp_deg, true_anomaly_deg, mu=MU,
                                centre="moon", plane_rotation=None):
    """
    Rotating-frame barycentric state [x, y, z, vx, vy, vz] (LU, LU/TU)
    of a two-body orbit about the Moon or the Earth.

    a_km             : semi-major axis, kilometres
    e                : eccentricity (0 <= e < 1)
    i_deg            : inclination to the reference plane, degrees
    raan_deg         : right ascension of the ascending node from the
                       reference direction, degrees
    argp_deg         : argument of periapsis, degrees
    true_anomaly_deg : true anomaly, degrees
    centre           : "moon" or "earth"
    plane_rotation   : optional 3x3 matrix from the elements' reference
                       frame to the rotating frame; identity means the
                       Moon's orbit plane with +x as reference direction.

    The two-body position and velocity are inertial quantities relative
    to the central body.  Converting to the rotating frame subtracts the
    frame's own rotation, omega x r with omega the unit z vector, applied
    to the position relative to the body (the body itself is at rest in
    the rotating frame, so its own inertial velocity cancels).
    """
    body_position, gm = central_body(centre, mu)
    a = crtbp.length_to_nondim(a_km)
    i = np.radians(i_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    nu = np.radians(true_anomaly_deg)

    # Two-body position and velocity in the perifocal frame.
    semi_latus_rectum = a * (1.0 - e ** 2)
    radius = semi_latus_rectum / (1.0 + e * np.cos(nu))
    position_perifocal = np.array([radius * np.cos(nu), radius * np.sin(nu), 0.0])
    speed_factor = np.sqrt(gm / semi_latus_rectum)
    velocity_perifocal = np.array([-speed_factor * np.sin(nu), speed_factor * (e + np.cos(nu)), 0.0])

    rotation = rotation_perifocal_to_reference(i, raan, argp)
    if plane_rotation is not None:
        rotation = np.asarray(plane_rotation, dtype=float) @ rotation
    relative_position = rotation @ position_perifocal
    relative_velocity_inertial = rotation @ velocity_perifocal

    # Inertial to rotating: v_rot = v_inertial - omega x r, omega = z_hat.
    omega_cross_r = np.array([-relative_position[1], relative_position[0], 0.0])
    relative_velocity_rotating = relative_velocity_inertial - omega_cross_r

    position = body_position + relative_position
    return np.concatenate([position, relative_velocity_rotating])


def state_to_keplerian(state, mu=MU, centre="moon", plane_rotation=None):
    """
    Osculating elements of a rotating-frame state about the Moon or the
    Earth, in the same reference frame keplerian_to_state_rotating uses.

    Returns a dictionary with a_km, e, i_deg, raan_deg, argp_deg,
    true_anomaly_deg, periapsis_km, apoapsis_km, period_hours and
    specific_energy (LU^2/TU^2, negative when bound to the body).
    Hyperbolic states get a negative a_km and apoapsis_km = inf.
    """
    body_position, gm = central_body(centre, mu)
    state = np.asarray(state, dtype=float)
    relative_position = state[:3] - body_position
    omega_cross_r = np.array([-relative_position[1], relative_position[0], 0.0])
    relative_velocity = state[3:] + omega_cross_r

    # Express both vectors in the elements' reference frame.
    if plane_rotation is not None:
        to_reference = np.asarray(plane_rotation, dtype=float).T
        relative_position = to_reference @ relative_position
        relative_velocity = to_reference @ relative_velocity

    r = np.linalg.norm(relative_position)
    v_squared = np.dot(relative_velocity, relative_velocity)
    angular_momentum = np.cross(relative_position, relative_velocity)
    h = np.linalg.norm(angular_momentum)

    specific_energy = 0.5 * v_squared - gm / r
    a = -gm / (2.0 * specific_energy)

    eccentricity_vector = np.cross(relative_velocity, angular_momentum) / gm - relative_position / r
    e = np.linalg.norm(eccentricity_vector)

    inclination = np.arccos(np.clip(angular_momentum[2] / h, -1.0, 1.0))

    # Node line: z cross h.  For an equatorial orbit it vanishes and the
    # node is undefined; the +x axis is used so the angles stay finite.
    node_vector = np.array([-angular_momentum[1], angular_momentum[0], 0.0])
    node_length = np.linalg.norm(node_vector)
    if node_length < 1e-14:
        node_vector = np.array([1.0, 0.0, 0.0])
        node_length = 1.0
    raan = np.arctan2(node_vector[1], node_vector[0])

    if e < 1e-12:
        argp = 0.0
        true_anomaly = np.arctan2(np.dot(np.cross(node_vector, relative_position), angular_momentum) / h,
                                  np.dot(node_vector, relative_position))
    else:
        argp = np.arctan2(np.dot(np.cross(node_vector, eccentricity_vector), angular_momentum) / h,
                          np.dot(node_vector, eccentricity_vector))
        true_anomaly = np.arctan2(np.dot(np.cross(eccentricity_vector, relative_position), angular_momentum) / h,
                                  np.dot(eccentricity_vector, relative_position))

    periapsis = a * (1.0 - e)
    apoapsis = a * (1.0 + e) if e < 1.0 else np.inf
    period_hours = 2.0 * np.pi * np.sqrt(a ** 3 / gm) * crtbp.TIME_UNIT_S / 3600.0 if a > 0.0 else np.inf

    return {"a_km": crtbp.length_to_km(a),
            "e": e,
            "i_deg": np.degrees(inclination),
            "raan_deg": np.degrees(raan) % 360.0,
            "argp_deg": np.degrees(argp) % 360.0,
            "true_anomaly_deg": np.degrees(true_anomaly) % 360.0,
            "periapsis_km": crtbp.length_to_km(periapsis),
            "apoapsis_km": crtbp.length_to_km(apoapsis),
            "period_hours": period_hours,
            "specific_energy": specific_energy}


def semi_major_axis_for_period(period_hours, mu=MU, centre="moon"):
    """Semi-major axis in km of a two-body orbit about the body with this period."""
    _, gm = central_body(centre, mu)
    period_nd = crtbp.time_to_nondim(period_hours * 3600.0)
    return crtbp.length_to_km((gm * (period_nd / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0))


ELEMENT_KEYS = ["a_km", "e", "i_deg", "raan_deg", "argp_deg", "true_anomaly_deg",
                "periapsis_km", "apoapsis_km", "period_hours", "specific_energy"]


def elements_history(states, mu=MU, centre="moon", plane_rotation=None):
    """
    Osculating elements along an (n, 6) trajectory, as a dictionary of
    arrays with the same keys as state_to_keplerian.  Used to watch how
    an orbit drifts under the other body's pull.
    """
    history = {key: np.zeros(len(states)) for key in ELEMENT_KEYS}
    for index, state in enumerate(states):
        elements = state_to_keplerian(state, mu, centre, plane_rotation)
        for key in ELEMENT_KEYS:
            history[key][index] = elements[key]
    return history
