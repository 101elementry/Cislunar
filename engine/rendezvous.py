"""
Relative motion and impulsive rendezvous between two spacecraft.

Relative motion is reported in the target's local-vertical
local-horizontal (LVLH) frame about a central body: radial (from the
body through the target), along-track (completing the right-handed set
in the orbit plane, close to the velocity direction) and cross-track
(along the target's orbital angular momentum).  This is the frame the
Clohessy-Wiltshire equations use, and the natural one for a chaser
approaching a target.

A two-impulse rendezvous is solved by the same differential correction
the station keeping uses: a first burn that makes the chaser's position
coincide with the target's after the transfer time (found by Newton
iteration on the position error with the STM), and a second burn that
matches velocities on arrival.  Nothing here assumes a two-body model
or small separations; the full CRTBP flow is used, so the same function
serves a GEO rendezvous, an NRHO rendezvous, or a transfer between two
orbit families.

All internal quantities are non-dimensional (LU, TU); the interface
takes and returns kilometres, metres per second and days where it
says so.
"""

import numpy as np

from engine import crtbp, frames, stationkeeping
from engine.crtbp import MU


def lvlh_basis(target_state_inertial):
    """
    Radial, along-track and cross-track unit vectors of the LVLH frame
    of a body-centred inertial state (position and velocity relative
    to the central body).
    """
    position = target_state_inertial[:3]
    velocity = target_state_inertial[3:]
    radial = position / np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    cross_track = angular_momentum / np.linalg.norm(angular_momentum)
    along_track = np.cross(cross_track, radial)
    return radial, along_track, cross_track


def relative_motion_lvlh(target_states, chaser_states, times_nondim, centre="earth", mu=MU):
    """
    Relative position (km) and velocity (m/s) of a chaser in the
    target's LVLH frame about a central body over a trajectory.

    Both trajectories are rotating-frame (n, 6) arrays on the same time
    grid.  They are first taken to the body-centred inertial frame,
    then the difference is projected on the LVLH axes.  The relative
    velocity is the rate of change of the relative position as seen in
    the LVLH frame, which itself turns with the target at the rate
    omega = h / r^2 about the cross-track axis: the inertial relative
    velocity minus omega x (relative position).  This is the quantity
    the Clohessy-Wiltshire equations describe and a rendezvous sensor
    reports; two spacecraft on the same circular orbit a few degrees
    apart have near-zero relative velocity in it even though their
    inertial velocities point in different directions.

    Returns (relative_position_km (n, 3), relative_velocity_m_s (n, 3))
    with columns radial, along-track, cross-track.
    """
    target = frames.rotating_to_inertial_states(target_states, times_nondim, centre, mu)
    chaser = frames.rotating_to_inertial_states(chaser_states, times_nondim, centre, mu)
    relative_position = np.zeros((len(target), 3))
    relative_velocity = np.zeros((len(target), 3))
    for index in range(len(target)):
        radial, along_track, cross_track = lvlh_basis(target[index])
        basis = np.vstack([radial, along_track, cross_track])
        rho = chaser[index, :3] - target[index, :3]
        rho_dot_inertial = chaser[index, 3:] - target[index, 3:]
        r = np.linalg.norm(target[index, :3])
        h = np.linalg.norm(np.cross(target[index, :3], target[index, 3:]))
        omega = (h / r ** 2) * cross_track
        relative_position[index] = basis @ rho
        relative_velocity[index] = basis @ (rho_dot_inertial - np.cross(omega, rho))
    return crtbp.length_to_km(relative_position), crtbp.velocity_to_km_s(relative_velocity) * 1000.0


def state_from_lvlh_offset(target_state, relative_position_km, relative_velocity_m_s, centre="moon", mu=MU):
    """
    Rotating-frame state (6,) of a chaser placed at a given offset from
    a target, the inverse of relative_motion_lvlh at one instant.

    target_state          : (6,) rotating-frame state of the target at
                            scenario time zero
    relative_position_km  : (3,) radial, along-track, cross-track offset
                            of the chaser from the target, km
    relative_velocity_m_s : (3,) velocity of the chaser as seen in the
                            LVLH frame, m/s (zero means the chaser holds
                            its place in that frame at this instant)
    centre                : "moon" or "earth", the body the LVLH frame
                            is defined about

    The steps mirror relative_motion_lvlh.  The target is taken to the
    body-centred inertial frame, which at time zero is aligned with the
    rotating axes.  The offset is rotated from LVLH to inertial axes.
    The LVLH frame turns at omega = h / r^2 about the cross-track axis,
    so the inertial relative velocity is the LVLH one plus
    omega x (relative position).  Adding these to the target gives the
    chaser's inertial state, which is then taken back to the rotating
    frame: add the body's position, and remove the velocity the rotating
    frame carries, z_hat x r, together with the body's own velocity.
    """
    target_state = np.asarray(target_state, dtype=float)
    target_inertial = frames.rotating_to_inertial_states(target_state[np.newaxis, :], np.array([0.0]), centre, mu)[0]
    radial, along_track, cross_track = lvlh_basis(target_inertial)
    basis = np.vstack([radial, along_track, cross_track])

    rho_lvlh = crtbp.length_to_nondim(np.asarray(relative_position_km, dtype=float))
    rho_dot_lvlh = crtbp.velocity_to_nondim(np.asarray(relative_velocity_m_s, dtype=float) / 1000.0)

    r = np.linalg.norm(target_inertial[:3])
    h = np.linalg.norm(np.cross(target_inertial[:3], target_inertial[3:]))
    omega = (h / r ** 2) * cross_track

    rho = basis.T @ rho_lvlh
    rho_dot = basis.T @ rho_dot_lvlh + np.cross(omega, rho)
    chaser_position_inertial = target_inertial[:3] + rho
    chaser_velocity_inertial = target_inertial[3:] + rho_dot

    # Back to the rotating frame at time zero.  The body sits at
    # (body_x, 0, 0) and moves with velocity z_hat x (body_x, 0, 0).
    body_x = crtbp.earth_position(mu)[0] if centre == "earth" else crtbp.moon_position(mu)[0]
    body_position = np.array([body_x, 0.0, 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])
    position = chaser_position_inertial + body_position
    velocity = chaser_velocity_inertial + np.cross(z_hat, body_position) - np.cross(z_hat, position)
    return np.concatenate([position, velocity])


def hold_point_state(target_state, offset_km, centre="moon", mu=MU):
    """
    Rotating-frame state of a hold point: a place fixed in the target's
    LVLH frame (radial, along-track, cross-track offset in km) with no
    velocity in that frame.  A chaser put there is momentarily at rest
    as the target sees it.  The CRTBP equations do not depend on time,
    so state_from_lvlh_offset, written for time zero, holds at any time.
    """
    return state_from_lvlh_offset(target_state, offset_km, [0.0, 0.0, 0.0], centre, mu)


def hop_to_hold_point(chaser_state, target_state, offset_km, transfer_time, centre="moon", mu=MU):
    """
    Two burns that take a chaser from where it is to a hold point beside
    the target, arriving after transfer_time (TU) and stopping there.

    chaser_state, target_state : (6,) rotating-frame states at the start
    Returns (delta_v1, delta_v2, chaser_state_after, target_state_after,
    miss_km): the burns in LU/TU (rotating frame), both states just
    after the second burn, and how closely the first burn hit the point.
    """
    target_after = crtbp.propagate(np.asarray(target_state, dtype=float), transfer_time, mu).y[:, -1]
    hold = hold_point_state(target_after, offset_km, centre, mu)
    delta_v1 = stationkeeping.targeting_manoeuvre(np.asarray(chaser_state, dtype=float), hold[:3],
                                                  transfer_time, mu, iterations=8)
    departed = np.asarray(chaser_state, dtype=float).copy()
    departed[3:] = departed[3:] + delta_v1
    arrived = crtbp.propagate(departed, transfer_time, mu).y[:, -1]
    delta_v2 = hold[3:] - arrived[3:]
    miss_km = crtbp.length_to_km(np.linalg.norm(arrived[:3] - hold[:3]))
    arrived[3:] = hold[3:]
    return delta_v1, delta_v2, arrived, target_after, miss_km


def approach_sequence(chaser_state, target_state, waypoints, centre="moon", mu=MU):
    """
    A stepped approach through hold points, the way crewed vehicles close
    on a station: stop outside the keep-out sphere, wait for a go, move
    to the next point, stop again.

    waypoints : list of (offset_km (3,), wait_before TU, transfer_time TU)
    The chaser drifts freely during each wait; the next hop starts from
    wherever that leaves it, as a real one would.

    Returns (burns, chaser_state, target_state, elapsed) where burns is
    a list of (time TU from the start, delta_v (3,) LU/TU).
    """
    chaser = np.asarray(chaser_state, dtype=float)
    target = np.asarray(target_state, dtype=float)
    elapsed = 0.0
    burns = []
    for offset_km, wait, transfer_time in waypoints:
        if wait > 0.0:
            chaser = crtbp.propagate(chaser, wait, mu).y[:, -1]
            target = crtbp.propagate(target, wait, mu).y[:, -1]
            elapsed += wait
        delta_v1, delta_v2, chaser, target, _ = hop_to_hold_point(chaser, target, offset_km, transfer_time, centre, mu)
        burns.append((elapsed, delta_v1))
        elapsed += transfer_time
        burns.append((elapsed, delta_v2))
    return burns, chaser, target, elapsed


def two_impulse_rendezvous(chaser_state, target_state, transfer_time, mu=MU, n_points=400):
    """
    Two-burn rendezvous from the chaser's state to the target's state
    after transfer_time (TU).

    The first burn is the targeting manoeuvre that puts the chaser at
    the target's future position; the target's future state comes from
    integrating it for the same time.  The second burn matches the
    arrival velocity to the target's.

    Returns a dictionary
        delta_v1_m_s, delta_v2_m_s : (3,) burns in the rotating frame
        total_delta_v_m_s          : sum of magnitudes
        transfer_states            : (n_points, 6) chaser trajectory
        transfer_times             : (n_points,) TU from departure
        target_states              : (n_points, 6) target trajectory
        arrival_position_error_km  : how closely the first burn hits
    """
    target_arrival = crtbp.propagate(target_state, transfer_time, mu).y[:, -1]
    delta_v1 = stationkeeping.targeting_manoeuvre(np.asarray(chaser_state, dtype=float),
                                                  target_arrival[:3], transfer_time, mu, iterations=8)

    departed = np.asarray(chaser_state, dtype=float).copy()
    departed[3:] = departed[3:] + delta_v1
    t_eval = np.linspace(0.0, transfer_time, n_points)
    transfer = crtbp.propagate(departed, transfer_time, mu, t_eval=t_eval)
    target_path = crtbp.propagate(target_state, transfer_time, mu, t_eval=t_eval)
    arrival = transfer.y[:, -1]
    delta_v2 = target_arrival[3:] - arrival[3:]

    to_m_s = lambda v: crtbp.velocity_to_km_s(v) * 1000.0
    return {"delta_v1_m_s": to_m_s(delta_v1),
            "delta_v2_m_s": to_m_s(delta_v2),
            "total_delta_v_m_s": float(np.linalg.norm(to_m_s(delta_v1)) + np.linalg.norm(to_m_s(delta_v2))),
            "transfer_states": transfer.y.T,
            "transfer_times": transfer.t,
            "target_states": target_path.y.T,
            "arrival_position_error_km": crtbp.length_to_km(np.linalg.norm(arrival[:3] - target_arrival[:3]))}


def transfer_time_sweep(chaser_state, target_state, transfer_times, mu=MU):
    """
    Total delta-v (m/s) of the two-impulse rendezvous for each transfer
    time in an array (TU).  The porkchop-style curve a mission designer
    reads the cheapest transfer from.  Transfers the corrector cannot
    solve get NaN.
    """
    costs = np.full(len(transfer_times), np.nan)
    for index, transfer_time in enumerate(transfer_times):
        try:
            solution = two_impulse_rendezvous(chaser_state, target_state, float(transfer_time), mu, n_points=2)
        except np.linalg.LinAlgError:
            continue
        if solution["arrival_position_error_km"] < 1.0:
            costs[index] = solution["total_delta_v_m_s"]
    return costs
