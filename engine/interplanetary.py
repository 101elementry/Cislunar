"""
Earth-Mars transfer design with patched conics.

The solar system is cut into three two-body problems.  Near the Earth
only the Earth pulls; on the way only the Sun; near Mars only Mars.
The pieces are joined at the edges of each planet's sphere of influence,
which is so small next to the distance between the planets that the
heliocentric leg is taken from planet centre to planet centre.

  1. Heliocentric leg.  The planets' positions at departure and arrival
     come from the JPL ephemeris.  Lambert's problem (engine/lambert.py)
     gives the Sun-centred orbit that joins them in the chosen time.
  2. Hyperbolic excess velocity.  The spacecraft's heliocentric velocity
     minus the planet's is the velocity it has left after climbing out
     of (or before falling into) the planet's gravity well:
         v_infinity = v_spacecraft - v_planet
     C3 = |v_infinity|^2 is the launch energy a launch vehicle is rated
     against.
  3. Burns.  Energy is conserved on the hyperbola about the planet, so
     at radius r the speed is sqrt(v_infinity^2 + 2 mu / r).  The burn
     is that speed minus the speed the vehicle already has there.

The last step is where the Oberth effect appears.  The same v_infinity
costs far less from an orbit that is already moving fast at a low
perigee.  A vehicle falling from the Moon's distance arrives at a
200 km perigee at almost escape speed, so it needs only the small
difference, where a vehicle in a circular low orbit must supply
3.6 km/s or so.  That is the case for staging a Mars vehicle in the
NRHO: the propellant is lifted there in separate, efficient pieces.

A porkchop plot is steps 1 and 2 over a grid of departure and arrival
dates.  Its minima repeat every 26 months, the synodic period, which is
the launch cycle any Mars campaign is scheduled around.

Units: km, km/s, seconds, Julian dates (TDB).  The ephemeris argument is
an engine.ephemeris.Ephemeris whose extract includes Mars.
"""

import numpy as np
from scipy.integrate import solve_ivp

from engine import lambert

MU_SUN_KM3_S2 = 1.32712440018e11
MU_EARTH_KM3_S2 = 398600.4418
MU_MARS_KM3_S2 = 42828.37
MU_VENUS_KM3_S2 = 324858.59
EARTH_RADIUS_KM = 6378.137
MARS_RADIUS_KM = 3396.2
VENUS_RADIUS_KM = 6051.8
MOON_DISTANCE_KM = 384400.0
SECONDS_PER_DAY = 86400.0
AU_KM = 149597870.7


def transfer(ephemeris, departure_jd, arrival_jd, origin="earth", destination="mars"):
    """
    One heliocentric transfer between two planets on given dates.

    Returns a dictionary
        r1, v1            : spacecraft heliocentric state at departure
        r2, v2            : at arrival
        v_infinity_depart : (3,) excess velocity leaving the origin, km/s
        v_infinity_arrive : (3,) excess velocity reaching the destination
        c3_km2_s2         : launch energy, |v_infinity_depart|^2
        time_of_flight_days
    Raises ValueError where Lambert's problem has no sensible solution.
    """
    time_of_flight_s = (arrival_jd - departure_jd) * SECONDS_PER_DAY
    if time_of_flight_s <= 0.0:
        raise ValueError("arrival must be after departure")
    origin_position, origin_velocity = ephemeris.heliocentric_state(origin, np.array([departure_jd]))
    target_position, target_velocity = ephemeris.heliocentric_state(destination, np.array([arrival_jd]))
    v1, v2 = lambert.solve(origin_position[0], target_position[0], time_of_flight_s, MU_SUN_KM3_S2)
    v_infinity_depart = v1 - origin_velocity[0]
    v_infinity_arrive = v2 - target_velocity[0]
    return {"r1": origin_position[0], "v1": v1, "r2": target_position[0], "v2": v2,
            "v_infinity_depart": v_infinity_depart, "v_infinity_arrive": v_infinity_arrive,
            "c3_km2_s2": float(np.dot(v_infinity_depart, v_infinity_depart)),
            "time_of_flight_days": arrival_jd - departure_jd}


def porkchop(ephemeris, departure_jds, arrival_jds, origin="earth", destination="mars",
             min_flight_days=60.0, max_flight_days=500.0):
    """
    Launch energy and arrival speed over a grid of dates.

    Returns (c3, v_infinity_arrive), each of shape
    (len(arrival_jds), len(departure_jds)) so that rows are arrival
    dates, as a porkchop plot is drawn.  Cells outside the flight time
    limits, or where Lambert fails, are NaN.
    """
    c3 = np.full((len(arrival_jds), len(departure_jds)), np.nan)
    v_infinity_arrive = np.full_like(c3, np.nan)
    for column, departure_jd in enumerate(departure_jds):
        for row, arrival_jd in enumerate(arrival_jds):
            flight_days = arrival_jd - departure_jd
            if flight_days < min_flight_days or flight_days > max_flight_days:
                continue
            try:
                result = transfer(ephemeris, departure_jd, arrival_jd, origin, destination)
            except ValueError:
                continue
            c3[row, column] = result["c3_km2_s2"]
            v_infinity_arrive[row, column] = np.linalg.norm(result["v_infinity_arrive"])
    return c3, v_infinity_arrive


def speed_on_hyperbola(v_infinity_km_s, radius_km, mu):
    """Speed at a given radius on the hyperbola with this excess speed: energy conservation."""
    return np.sqrt(v_infinity_km_s ** 2 + 2.0 * mu / radius_km)


def circular_speed(radius_km, mu):
    return np.sqrt(mu / radius_km)


def speed_at_periapsis(periapsis_radius_km, apoapsis_radius_km, mu):
    """Periapsis speed of an ellipse, from the vis-viva equation with a = (rp + ra) / 2."""
    return np.sqrt(2.0 * mu * (1.0 / periapsis_radius_km - 1.0 / (periapsis_radius_km + apoapsis_radius_km)))


def departure_burn_from_circular_orbit(v_infinity_km_s, altitude_km=400.0,
                                       mu=MU_EARTH_KM3_S2, body_radius_km=EARTH_RADIUS_KM):
    """Burn (km/s) from a circular parking orbit onto the departure hyperbola."""
    radius = body_radius_km + altitude_km
    return speed_on_hyperbola(v_infinity_km_s, radius, mu) - circular_speed(radius, mu)


def departure_burn_from_lunar_distance(v_infinity_km_s, perigee_altitude_km=200.0,
                                       apogee_radius_km=MOON_DISTANCE_KM,
                                       mu=MU_EARTH_KM3_S2, body_radius_km=EARTH_RADIUS_KM):
    """
    Burn (km/s) at the perigee of an ellipse that comes down from the
    Moon's distance, onto the departure hyperbola.  This is the powered
    Earth flyby a vehicle staged in the NRHO would fly.  The small burn
    that lowers the perigee from the NRHO in the first place is a
    three-body problem and is not included here.
    """
    radius = body_radius_km + perigee_altitude_km
    return speed_on_hyperbola(v_infinity_km_s, radius, mu) - speed_at_periapsis(radius, apogee_radius_km, mu)


def capture_burn(v_infinity_km_s, periapsis_altitude_km=250.0, apoapsis_altitude_km=33800.0,
                 mu=MU_MARS_KM3_S2, body_radius_km=MARS_RADIUS_KM):
    """
    Burn (km/s) at periapsis to capture from the arrival hyperbola into
    an ellipse about the destination.  The defaults are the one-sol
    (24.6 hour) Mars parking orbit used in NASA's reference studies.
    """
    periapsis = body_radius_km + periapsis_altitude_km
    apoapsis = body_radius_km + apoapsis_altitude_km
    return speed_on_hyperbola(v_infinity_km_s, periapsis, mu) - speed_at_periapsis(periapsis, apoapsis, mu)


def heliocentric_path(r1, v1, time_of_flight_days, n_points=300):
    """
    (n_points, 3) positions along the Sun-centred transfer orbit, by
    integrating the two-body equation r'' = -mu r / |r|^3 from the
    departure state.  The end point is a check on the Lambert solution.
    """
    def two_body(_, state):
        position = state[:3]
        return np.concatenate([state[3:], -MU_SUN_KM3_S2 * position / np.linalg.norm(position) ** 3])

    duration = time_of_flight_days * SECONDS_PER_DAY
    solution = solve_ivp(two_body, (0.0, duration), np.concatenate([r1, v1]),
                         t_eval=np.linspace(0.0, duration, n_points), method="DOP853", rtol=1e-11, atol=1e-6)
    return solution.y[:3].T


def best_in_window(ephemeris, departure_jds, arrival_jds, c3, v_infinity_arrive, max_flight_days=None):
    """
    The grid cell with the least total of departure and arrival excess
    speed, optionally among flights no longer than max_flight_days.
    Returns (departure_jd, arrival_jd, c3, v_infinity_arrive) or None.
    """
    total = np.sqrt(c3) + v_infinity_arrive
    if max_flight_days is not None:
        flight = arrival_jds[:, np.newaxis] - departure_jds[np.newaxis, :]
        total = np.where(flight <= max_flight_days, total, np.nan)
    if np.all(np.isnan(total)):
        return None
    row, column = np.unravel_index(np.nanargmin(total), total.shape)
    return departure_jds[column], arrival_jds[row], c3[row, column], v_infinity_arrive[row, column]



def gravity_assist(v_infinity_in, v_infinity_out, mu=MU_VENUS_KM3_S2, body_radius_km=VENUS_RADIUS_KM,
                   min_altitude_km=300.0):
    """
    What a planet must do to turn one excess velocity into another.

    A flyby cannot change the speed relative to the planet, only the
    direction: the planet's gravity swings the velocity round by a turn
    angle that is larger the closer and the slower the pass.  On a
    hyperbola with excess speed v and periapsis radius rp the asymptote
    makes the angle asin(1 / (1 + rp v^2 / mu)) with the axis, so the
    whole turn, inbound half plus outbound half, is

        turn = asin(1 / (1 + rp v_in^2 / mu)) + asin(1 / (1 + rp v_out^2 / mu))

    Given the turn the two heliocentric legs need, this is solved for
    rp by bisection (the turn falls steadily as rp grows).  If the two
    speeds differ, a burn at periapsis makes up the difference, where it
    is cheapest: the difference of the two periapsis speeds.

    v_infinity_in, v_infinity_out : (3,) excess velocities, km/s
    Returns (periapsis_altitude_km, burn_km_s, feasible).  feasible is
    False if the turn needs a pass below min_altitude_km; the altitude
    is then the limit and the turn is not achieved.
    """
    speed_in = np.linalg.norm(v_infinity_in)
    speed_out = np.linalg.norm(v_infinity_out)
    turn = np.arccos(np.clip(np.dot(v_infinity_in, v_infinity_out) / (speed_in * speed_out), -1.0, 1.0))

    def turn_at(periapsis_radius):
        return (np.arcsin(1.0 / (1.0 + periapsis_radius * speed_in ** 2 / mu))
                + np.arcsin(1.0 / (1.0 + periapsis_radius * speed_out ** 2 / mu)))

    lowest = body_radius_km + min_altitude_km
    feasible = turn_at(lowest) >= turn
    periapsis = lowest
    if feasible:
        low, high = lowest, 1.0e7
        for _ in range(80):
            middle = 0.5 * (low + high)
            if turn_at(middle) > turn:
                low = middle
            else:
                high = middle
        periapsis = 0.5 * (low + high)
    burn = abs(speed_on_hyperbola(speed_out, periapsis, mu) - speed_on_hyperbola(speed_in, periapsis, mu))
    return periapsis - body_radius_km, burn, bool(feasible)
