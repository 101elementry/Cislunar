"""
Two-burn transfers from a circular parking orbit to a moving target in
the three-body model: a lander climbing from low lunar orbit to the
NRHO, a crew vehicle leaving low Earth orbit for it.

The method has two stages, because Newton iteration in the three-body
problem only finds the solution nearest to its first guess and a poor
guess lands on an absurd one (leaving low lunar orbit for the NRHO with
no guess at all converges to a 3.6 km/s transfer; the real one is about
0.7).

  1. Seed.  Near its parking orbit the vehicle feels almost only the
     body it orbits, so Lambert's problem about that body (two-body,
     engine/lambert.py) gives a good first guess.  The departure point
     on the parking orbit is chosen so that the transfer leaves along
     the orbit's own direction of travel: the burn is then tangential,
     the cheapest way to raise an orbit, and the parking orbit plane is
     simply the transfer plane.  That point is found by sliding the
     departure position round the circle until the Lambert velocity
     there has no radial component.
  2. Correct.  The seed is refined with the full three-body equations
     by the same differential correction the station keeping uses
     (engine/stationkeeping.targeting_manoeuvre): Newton steps on the
     arrival position error with the state transition matrix.

The second burn matches the target's velocity on arrival.

Frames.  Lambert works in the body-centred inertial frame aligned with
the rotating axes at time zero (engine/frames.py).  A velocity change
is the same vector in both frames at that instant, because the frame
velocity z x r is the same for the vehicle before and after the burn;
so the seed burn carries over unchanged.

Units inside are non-dimensional (LU, TU); burns are reported in m/s.
"""

import numpy as np
from scipy.optimize import minimize

from engine import crtbp, frames, lambert, rendezvous, stationkeeping
from engine.crtbp import MU


def body_parameters(centre, mu=MU):
    """(gravitational parameter in LU^3/TU^2, rotating-frame position) of "moon" or "earth"."""
    if centre == "moon":
        return mu, crtbp.moon_position(mu)
    return 1.0 - mu, crtbp.earth_position(mu)


def inertial_to_rotating_at_zero(position_inertial, velocity_inertial, centre, mu=MU):
    """
    Rotating-frame state of a body-centred inertial state at time zero,
    when the two sets of axes coincide: add the body's position, and
    remove the velocity the rotating frame carries (z x r) after adding
    the body's own velocity (z x r_body).
    """
    _, body_position = body_parameters(centre, mu)
    z_hat = np.array([0.0, 0.0, 1.0])
    position = position_inertial + body_position
    velocity = velocity_inertial + np.cross(z_hat, body_position) - np.cross(z_hat, position)
    return np.concatenate([position, velocity])


def transfer_plane_normal(target_direction, plane_angle_deg):
    """
    Unit normal of a transfer plane that contains target_direction.  At
    plane_angle 0 the plane is the one closest to the Earth-Moon plane;
    the angle turns it about the target direction, and 90 degrees gives
    the plane through the z axis, which about the Moon is a polar orbit.
    """
    z_hat = np.array([0.0, 0.0, 1.0])
    flat_normal = z_hat - np.dot(z_hat, target_direction) * target_direction
    flat_normal = flat_normal / np.linalg.norm(flat_normal)
    sideways = np.cross(target_direction, flat_normal)
    angle = np.radians(plane_angle_deg)
    return np.cos(angle) * flat_normal + np.sin(angle) * sideways


def tangential_departure_seed(target_position_inertial, transfer_time, radius, mu_body, normal):
    """
    Two-body seed: the point on a circular orbit of the given radius, in
    the plane with this normal, from which the Lambert transfer to
    target_position_inertial leaves with no radial velocity.

    Returns (r1, v1, circular_velocity) in the body-centred inertial
    frame, or None if no such point exists for this transfer time.
    """
    e1 = target_position_inertial / np.linalg.norm(target_position_inertial)
    e2 = np.cross(normal, e1)

    def radial_speed(angle_behind_deg):
        angle = np.radians(angle_behind_deg)
        r1 = radius * (np.cos(angle) * e1 - np.sin(angle) * e2)
        v1, _ = lambert.solve(r1, target_position_inertial, transfer_time, mu_body, prograde=True, normal=normal)
        return np.dot(r1, v1) / radius, r1, v1

    # Slide the departure point round the circle and bracket the change
    # of sign of the radial speed.  The search starts almost opposite
    # the target, where the transfer is closest to a Hohmann half
    # ellipse, the cheapest kind, and works back from there.
    angles = np.linspace(179.5, 60.0, 80)
    previous = None
    for angle in angles:
        try:
            value, _, _ = radial_speed(angle)
        except ValueError:
            previous = None
            continue
        if previous is not None and previous[1] * value < 0.0:
            low, high = previous[0], angle
            low_value = previous[1]
            for _ in range(50):
                middle = 0.5 * (low + high)
                middle_value, r1, v1 = radial_speed(middle)
                if low_value * middle_value <= 0.0:
                    high = middle
                else:
                    low, low_value = middle, middle_value
            along = np.cross(normal, r1 / radius)
            return r1, v1, np.sqrt(mu_body / radius) * along
        previous = (angle, value)
    return None


def from_circular_orbit(target_state, transfer_time, centre="moon", altitude_km=100.0, plane_angle_deg=0.0,
                        arrival_offset_km=None, mu=MU, n_points=400):
    """
    Two-burn transfer from a circular parking orbit about the Moon or
    the Earth to a target, arriving after transfer_time (TU).

    target_state    : (6,) rotating-frame state of the target at time zero
    altitude_km     : height of the parking orbit above the body
    plane_angle_deg : which parking orbit plane, see transfer_plane_normal
    arrival_offset_km : optional (3,) radial, along-track, cross-track
                      offset from the target, in its LVLH frame about
                      the Moon.  The vehicle then arrives at that hold
                      point and stops relative to the target, instead of
                      flying straight to it.

    Returns None if there is no tangential seed or the correction does
    not converge, otherwise a dictionary
        start_state         : (6,) rotating-frame state on the parking
                              orbit just before the first burn
        delta_v1_m_s, delta_v2_m_s : (3,) burns, rotating frame
        burn1_m_s, burn2_m_s, total_delta_v_m_s : magnitudes
        transfer_states, transfer_times : the coast, (n, 6) and (n,) TU
        arrival_position_error_km
        parking_inclination_deg : tilt of the parking orbit to the
                              Earth-Moon plane
    """
    mu_body, _ = body_parameters(centre, mu)
    body_radius = crtbp.MOON_RADIUS_ND if centre == "moon" else frames.EARTH_RADIUS_ND
    radius = body_radius + crtbp.length_to_nondim(altitude_km)

    target_arrival = crtbp.propagate(np.asarray(target_state, dtype=float), transfer_time, mu).y[:, -1]
    if arrival_offset_km is not None:
        target_arrival = rendezvous.hold_point_state(target_arrival, arrival_offset_km, "moon", mu)
    arrival_inertial = frames.rotating_to_inertial_states(target_arrival[np.newaxis, :],
                                                          np.array([transfer_time]), centre, mu)[0]
    normal = transfer_plane_normal(arrival_inertial[:3] / np.linalg.norm(arrival_inertial[:3]), plane_angle_deg)
    seed = tangential_departure_seed(arrival_inertial[:3], transfer_time, radius, mu_body, normal)
    if seed is None:
        return None
    r1, v1, circular_velocity = seed

    start_state = inertial_to_rotating_at_zero(r1, circular_velocity, centre, mu)
    delta_v1 = stationkeeping.targeting_manoeuvre(start_state, target_arrival[:3], transfer_time, mu,
                                                  iterations=12, initial_delta_v=v1 - circular_velocity)
    departed = start_state.copy()
    departed[3:] = departed[3:] + delta_v1
    coast = crtbp.propagate(departed, transfer_time, mu, t_eval=np.linspace(0.0, transfer_time, n_points))
    arrival = coast.y[:, -1]
    miss_km = crtbp.length_to_km(np.linalg.norm(arrival[:3] - target_arrival[:3]))
    if not np.isfinite(miss_km) or miss_km > 1.0:
        return None
    delta_v2 = target_arrival[3:] - arrival[3:]

    def to_m_s(velocity):
        return crtbp.velocity_to_km_s(velocity) * 1000.0

    return {"start_state": start_state,
            "delta_v1_m_s": to_m_s(delta_v1), "delta_v2_m_s": to_m_s(delta_v2),
            "burn1_m_s": float(np.linalg.norm(to_m_s(delta_v1))),
            "burn2_m_s": float(np.linalg.norm(to_m_s(delta_v2))),
            "total_delta_v_m_s": float(np.linalg.norm(to_m_s(delta_v1)) + np.linalg.norm(to_m_s(delta_v2))),
            "transfer_states": coast.y.T, "transfer_times": coast.t,
            "arrival_position_error_km": miss_km,
            "parking_inclination_deg": float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))}


# --------------------------------------------------------------------------
# Closest approach to the Earth, and the powered lunar flyby
# --------------------------------------------------------------------------

def earth_perigees(state, duration, mu=MU):
    """
    Every closest approach to the Earth along a trajectory propagated
    for `duration` TU (negative to go back in time).

    The distance to the Earth stops changing when the position relative
    to the Earth is perpendicular to the velocity; the Earth is fixed in
    the rotating frame, so the rotating-frame velocity can be used.  The
    integrator is asked to stop at nothing but to report those instants,
    and the ones that are minima are kept.

    Returns a list of (time TU, radius LU, state (6,)), in the order met.
    """
    earth = crtbp.earth_position(mu)

    def range_rate(_, current, *unused):    # solve_ivp also passes mu, the argument of the dynamics
        return np.dot(current[:3] - earth, current[3:6])

    # Going forward in time a perigee is where the range rate rises
    # through zero; going backward the integrator sees it falling.
    range_rate.direction = 1.0 if duration > 0.0 else -1.0
    solution = crtbp.propagate(np.asarray(state, dtype=float), duration, mu, events=range_rate)
    found = []
    for time, event_state in zip(solution.t_events[0], solution.y_events[0]):
        found.append((float(time), float(np.linalg.norm(event_state[:3] - earth)), event_state.copy()))
    return found


def earth_inertial_velocity(state, mu=MU):
    """Velocity relative to the Earth in non-rotating axes: v + z x (r - r_earth)."""
    offset = state[:3] - crtbp.earth_position(mu)
    return state[3:6] + np.cross(np.array([0.0, 0.0, 1.0]), offset)


def flyby_from_earth(leg_to_target, parking_altitude_km=200.0, search_days=8.0, mu=MU):
    """
    The Earth-to-Moon leg and the powered lunar flyby that feed a given
    Moon-to-target leg, which together make the three-burn transfer
    crewed missions fly: injection from Earth orbit, a braking burn at
    the closest point to the Moon, insertion at the target.

    leg_to_target : a result of from_circular_orbit about the Moon.  Its
                    departure point is the flyby's perilune and its
                    departure velocity is what the vehicle must have
                    after the flyby burn.

    The unknown is the flyby burn itself, three components.  For any
    choice, the velocity before the burn is known, and the path can be
    followed back in time from perilune to its closest approach to the
    Earth.  The burn wanted is the smallest one whose path came from a
    perigee at the parking orbit's altitude: a minimisation with one
    equality condition, solved by sequential quadratic programming
    (scipy SLSQP).  It starts from the best purely braking burn, the one
    along the direction of travel, found by a scan.  At perigee the
    velocity is horizontal, so the injection burn is the perigee speed
    minus circular speed.

    Returns None if the optimiser cannot reach the perigee, else
        injection_m_s, flyby_m_s : burn magnitudes
        coast_days               : perigee to perilune
        perigee_state            : (6,) rotating-frame state just after
                                   the injection burn
        flyby_delta_v            : (3,) LU/TU rotating frame, the burn at
                                   perilune (a braking burn)
        parking_inclination_deg  : tilt of the Earth parking orbit to the
                                   Earth-Moon plane
    """
    moon = crtbp.moon_position(mu)
    z_hat = np.array([0.0, 0.0, 1.0])
    perilune = leg_to_target["start_state"].copy()
    after_burn = perilune[3:] + crtbp.velocity_to_nondim(leg_to_target["delta_v1_m_s"] / 1000.0)
    # Moon-centred non-rotating velocity after the flyby burn.
    after_inertial = after_burn + np.cross(z_hat, perilune[:3] - moon)
    target_radius = frames.EARTH_RADIUS_ND + crtbp.length_to_nondim(parking_altitude_km)
    duration = -crtbp.time_to_nondim(search_days * crtbp.SECONDS_PER_DAY)

    def perigee_for(extra_inertial):
        """Closest approach to the Earth of the path that arrives at perilune faster by extra_inertial."""
        before_inertial = after_inertial + extra_inertial
        state = np.concatenate([perilune[:3], before_inertial - np.cross(z_hat, perilune[:3] - moon)])
        # The perigee wanted is the last one before the flyby: the first
        # met going back, at least a day away so that a wobble near the
        # Moon is not mistaken for it.
        for time, radius, event_state in earth_perigees(state, duration, mu):
            if abs(time) > crtbp.time_to_nondim(crtbp.SECONDS_PER_DAY):
                return radius, time, event_state
        return None

    # Scan braking burns along the direction of travel for the lowest perigee.
    best_k = None
    for k in np.linspace(0.02, 0.3, 15):
        found = perigee_for(k * after_inertial)
        if found is not None and (best_k is None or found[0] < best_k[1]):
            best_k = (k, found[0])
    if best_k is None:
        return None

    # Work in m/s so that the three unknowns are of order a hundred, and
    # measure the perigee miss in thousands of kilometres.
    unit = crtbp.velocity_to_nondim(0.001)

    def miss_thousand_km(extra_m_s):
        found = perigee_for(np.asarray(extra_m_s) * unit)
        if found is None:
            return 400.0
        return crtbp.length_to_km(found[0] - target_radius) / 1000.0

    solution = minimize(lambda extra: float(np.linalg.norm(extra)), best_k[0] * after_inertial / unit,
                        method="SLSQP", constraints=[{"type": "eq", "fun": miss_thousand_km}],
                        options={"maxiter": 60, "ftol": 1e-4})
    extra = solution.x * unit
    found = perigee_for(extra)
    if found is None or abs(crtbp.length_to_km(found[0] - target_radius)) > 50.0:
        return None
    braking = extra
    radius, time, perigee_state = found
    perigee_velocity = earth_inertial_velocity(perigee_state, mu)
    circular = np.sqrt((1.0 - mu) / radius)
    momentum = np.cross(perigee_state[:3] - crtbp.earth_position(mu), perigee_velocity)

    def to_m_s(velocity):
        return crtbp.velocity_to_km_s(velocity) * 1000.0

    return {"injection_m_s": float(to_m_s(np.linalg.norm(perigee_velocity) - circular)),
            "flyby_m_s": float(to_m_s(np.linalg.norm(braking))),
            "coast_days": float(crtbp.time_to_days(abs(time))),
            "perigee_state": perigee_state, "flyby_delta_v": -braking,
            "perigee_altitude_km": float(crtbp.length_to_km(radius - frames.EARTH_RADIUS_ND)),
            "parking_inclination_deg": float(np.degrees(np.arccos(np.clip(
                momentum[2] / np.linalg.norm(momentum), -1.0, 1.0))))}
