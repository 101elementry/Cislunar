"""
Ephemeris dynamics: a spacecraft pulled by the Earth, the Moon and the
Sun as point masses at their real positions, taken from the JPL
ephemeris at every instant.

This is the next rung up from the circular restricted three-body
problem.  The CRTBP puts the Earth and Moon on a fixed circle and leaves
the Sun out; here the Earth-Moon distance, the tilt and turning of the
Moon's orbit, and the Sun's pull are all whatever DE440 says they were
on the day.  Nothing here repeats exactly, so periodic orbits do not
exist in this model; a CRTBP orbit carried into it drifts, and how fast
it drifts is the measure of how good the CRTBP was.

Still left out: the Moon's and Earth's gravity fields beyond the point
mass (J2, mascons), solar radiation pressure, the other planets, and
relativity.  At NRHO distances the Sun and the true lunar orbit are the
largest effects the CRTBP misses; the rest are one to two orders of
magnitude smaller.

Frame and units
  * Moon-centred, ICRF axes, kilometres and seconds.  Centred on the
    Moon because an NRHO passes within 3,300 km of it, where its pull
    changes fastest.
  * Time is seconds from an epoch given as a TDB Julian date.
  * The Moon-centred frame is accelerating (the Moon falls toward the
    Earth and the Sun), so each third body contributes its pull on the
    spacecraft minus its pull on the Moon: the "indirect" term.
"""

import numpy as np
from scipy.integrate import solve_ivp

from engine import crtbp
from engine.interplanetary import MU_SUN_KM3_S2

SECONDS_PER_DAY = 86400.0

# Gravitational parameters (km^3/s^2).  The Earth and Moon values are
# the ones the CRTBP's units are built from, so the two models share
# their masses; DE440's own values differ by less than one part in 10^7.
GM_KM3_S2 = {"earth": crtbp.GM_EARTH_KM3_S2, "moon": crtbp.GM_MOON_KM3_S2, "sun": MU_SUN_KM3_S2}

# Relative tolerance as in the CRTBP.  The absolute tolerance is in km
# and km/s here: 1e-9 km is a micrometre.
INTEGRATOR_METHOD = "DOP853"
INTEGRATOR_RTOL = 1e-12
INTEGRATOR_ATOL = 1e-9


# --------------------------------------------------------------------------
# Where the bodies are
# --------------------------------------------------------------------------

def bodies_from_moon(ephemeris, jd_tdb):
    """
    Positions (km, ICRF) of the Earth and the Sun relative to the Moon's
    centre at the given TDB Julian date(s).  Returns (earth, sun), each
    (n, 3).

    DE440 gives the Earth and the Moon relative to the Earth-Moon
    barycentre, and the barycentre and the Sun relative to the
    solar-system barycentre, so each vector is a difference of segments.
    """
    moon_from_emb, _ = ephemeris.segments["moon"].position_velocity(jd_tdb)
    earth_from_emb, _ = ephemeris.segments["earth"].position_velocity(jd_tdb)
    emb_from_ssb, _ = ephemeris.segments["emb"].position_velocity(jd_tdb)
    sun_from_ssb, _ = ephemeris.segments["sun"].position_velocity(jd_tdb)
    earth = earth_from_emb - moon_from_emb
    sun = sun_from_ssb - (emb_from_ssb + moon_from_emb)
    return earth, sun


# --------------------------------------------------------------------------
# Equations of motion
# --------------------------------------------------------------------------

def acceleration(position_km, third_bodies_km):
    """
    Acceleration (km/s^2) of a spacecraft at position_km (3,) from the
    Moon's centre.  third_bodies_km is {name: (3,) position of that body
    from the Moon's centre}, for any of "earth" and "sun"; an empty dict
    leaves the Moon alone (two-body motion, used as a check).

        a = -GM_moon r/|r|^3
            + sum over bodies j of GM_j [ (d_j - r)/|d_j - r|^3 - d_j/|d_j|^3 ]

    The first term in the bracket is body j pulling the spacecraft; the
    second is body j pulling the Moon, subtracted because the frame
    rides with the Moon.
    """
    r = np.asarray(position_km, dtype=float)
    total = -GM_KM3_S2["moon"] * r / np.linalg.norm(r) ** 3
    for name, d in third_bodies_km.items():
        to_body = d - r
        total = total + GM_KM3_S2[name] * (to_body / np.linalg.norm(to_body) ** 3 - d / np.linalg.norm(d) ** 3)
    return total


def equations_of_motion(time_s, state, ephemeris, epoch_jd_tdb, bodies=("earth", "sun")):
    """
    Time derivative of [x, y, z, vx, vy, vz] (km, km/s, Moon-centred
    ICRF) at time_s seconds after epoch_jd_tdb.  bodies names the third
    bodies to include.
    """
    jd = epoch_jd_tdb + time_s / SECONDS_PER_DAY
    earth, sun = bodies_from_moon(ephemeris, np.array([jd]))
    positions = {"earth": earth[0], "sun": sun[0]}
    third_bodies = {name: positions[name] for name in bodies}
    return np.concatenate([state[3:], acceleration(state[:3], third_bodies)])


def propagate(state0_km, duration_s, ephemeris, epoch_jd_tdb, t_eval=None, bodies=("earth", "sun"),
              events=None, dense_output=False):
    """
    Integrate a Moon-centred ICRF state (km, km/s) for duration_s seconds
    (negative to go backward) from epoch_jd_tdb.  Returns the scipy
    solve_ivp result; times in .t are seconds from the epoch.
    """
    if not ephemeris.covers(epoch_jd_tdb + np.array([0.0, duration_s / SECONDS_PER_DAY])):
        raise ValueError("the ephemeris extract does not cover this time span")
    return solve_ivp(equations_of_motion, (0.0, duration_s), np.asarray(state0_km, dtype=float),
                     method=INTEGRATOR_METHOD, rtol=INTEGRATOR_RTOL, atol=INTEGRATOR_ATOL,
                     t_eval=t_eval, events=events, dense_output=dense_output,
                     args=(ephemeris, epoch_jd_tdb, tuple(bodies)))


# --------------------------------------------------------------------------
# Carrying a CRTBP state into the real frame, and back
# --------------------------------------------------------------------------

def instantaneous_frame(ephemeris, jd_tdb):
    """
    The real Earth-Moon rotating frame at one TDB Julian date.

    Returns (axes, distance_km, distance_rate_km_s, rate_rad_s):
      axes               (3, 3) rows x, y, z in ICRF: x from the Earth to
                         the Moon, z along the Moon's orbital angular
                         momentum, y completing the set (the same axes
                         Ephemeris.rotating_frame_axes gives)
      distance_km        the Earth-Moon distance on the day, which plays
                         the part of the CRTBP's fixed 384,400 km
      distance_rate_km_s how fast that distance is changing
      rate_rad_s         how fast the Earth-Moon line is turning, which
                         plays the part of the CRTBP's fixed mean motion
    """
    position, velocity = ephemeris.earth_to_moon(np.atleast_1d(jd_tdb))
    position, velocity = position[0], velocity[0]
    distance = np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    x_axis = position / distance
    z_axis = angular_momentum / np.linalg.norm(angular_momentum)
    y_axis = np.cross(z_axis, x_axis)
    rate = np.linalg.norm(angular_momentum) / distance ** 2
    return np.array([x_axis, y_axis, z_axis]), distance, np.dot(position, velocity) / distance, rate


def rotating_to_ephemeris(state_nd, ephemeris, jd_tdb):
    """
    Map a CRTBP rotating-frame state (non-dimensional, barycentric) to a
    Moon-centred ICRF state (km, km/s) using the real Earth-Moon geometry
    of the day.

    The CRTBP's two fixed numbers are replaced by that day's values: the
    Earth-Moon distance l instead of 384,400 km, and the time unit
    sqrt(l^3 / GM) that Kepler's third law gives for that distance
    instead of the fixed 4.34 days.  Measured from the Moon, a CRTBP
    offset rho (in Earth-Moon distances) becomes l rho km.
    Differentiating l(t) C(t)^T rho(t) in time gives the velocity: the
    distance changing (l_dot rho) plus the motion measured in the frame
    and the frame turning, both scaled by l / time unit.

    Why Kepler's time unit and not the real turning rate of the
    Earth-Moon line: the Moon's orbit is an ellipse, so near perigee the
    line turns up to about 13 % faster than average.  Scaling the
    velocity by that rate while the distance shrinks gives the spacecraft
    more energy than the Moon can hold at perilune, and it escapes on the
    first pass.  Kepler's unit scales speed as 1/sqrt(l), exactly as the
    Moon's pull scales with distance, so the energy balance of the CRTBP
    orbit is kept.  The real turning of the axes still enters, through
    the axes C of each day.

    This is the standard "pulsating" rotating frame.  The state it gives
    is the natural starting guess in the ephemeris model, not a periodic
    orbit there.
    """
    axes, distance, distance_rate, _ = instantaneous_frame(ephemeris, jd_tdb)
    time_unit_s = kepler_time_unit_s(distance)
    rho = np.asarray(state_nd[:3], dtype=float) - crtbp.moon_position()
    v = np.asarray(state_nd[3:], dtype=float)
    z_cross_rho = np.array([-rho[1], rho[0], 0.0])
    position_km = distance * axes.T @ rho
    velocity_km_s = axes.T @ (distance_rate * rho + distance / time_unit_s * (v + z_cross_rho))
    return np.concatenate([position_km, velocity_km_s])


def kepler_time_unit_s(distance_km):
    """
    The CRTBP time unit for a given Earth-Moon distance: sqrt(l^3 / GM)
    with GM the Earth's plus the Moon's.  At 384,400 km it is the fixed
    375,190 s of engine/crtbp.py.
    """
    return np.sqrt(distance_km ** 3 / (crtbp.GM_EARTH_KM3_S2 + crtbp.GM_MOON_KM3_S2))


def ephemeris_to_rotating(state_km, ephemeris, jd_tdb):
    """
    The inverse of rotating_to_ephemeris: a Moon-centred ICRF state (km,
    km/s) to non-dimensional barycentric rotating coordinates in that
    day's pulsating frame.  Used to compare an ephemeris trajectory with
    a CRTBP one in the frame where the CRTBP orbit stands still.
    """
    axes, distance, distance_rate, _ = instantaneous_frame(ephemeris, jd_tdb)
    time_unit_s = kepler_time_unit_s(distance)
    rho = axes @ np.asarray(state_km[:3], dtype=float) / distance
    velocity_in_frame = axes @ np.asarray(state_km[3:], dtype=float)
    z_cross_rho = np.array([-rho[1], rho[0], 0.0])
    v = (velocity_in_frame - distance_rate * rho) / (distance / time_unit_s) - z_cross_rho
    return np.concatenate([rho + crtbp.moon_position(), v])
