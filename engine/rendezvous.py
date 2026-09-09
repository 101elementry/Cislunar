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
