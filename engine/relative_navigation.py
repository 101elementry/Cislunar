"""
Angles-only relative navigation: a camera on a chaser spacecraft
measures the direction to a target near the 9:2 NRHO, and the target's
state is estimated from those angles alone.

The question this module serves: where on the NRHO does the
three-body motion make the range to the target observable from camera
angles alone, without burns?  Under linear relative dynamics it never
is.  If the relative state obeys delta(t) = Phi(t) delta(0), then
k delta(t) is also a solution for any k, and every such solution
points along the same line of sight at every time, so the angles
cannot tell a target 50 km away from one 100 km away on a trajectory
twice the size (Woffinden and Geller, 2009).  The true flow is not
linear: the relative motion carries second-order terms, largest where
the gravity gradient changes fastest (perilune), which bend the line of
sight differently for different ranges.  Any range information in the
angles comes from that bending.

Stated assumption: the chaser knows its own state perfectly.  Only the
target's rotating-frame state [x, y, z, vx, vy, vz] is estimated, so
the CRTBP state and STM propagation, the EKF, UKF, batch least squares
and the Fisher information of engine/estimation.py and
engine/observability.py are used unchanged; the chaser's state enters
only through the measurement function, which carries it at each
measurement time.  In practice the chaser's own navigation error (from
ground tracking, of order a kilometre) would add to the target error
and is not modelled here.

The measurement: two angles of the chaser-to-target line of sight in
the chaser's LVLH frame about the Moon (radial from the Moon through
the chaser, cross-track along the chaser's orbital angular momentum,
along-track completing the right-handed set):
    in-plane angle     = atan2(rho . along_track, rho . radial)
    out-of-plane angle = asin(rho . cross_track / |rho|)
in degrees.  A target straight ahead along-track reads (90, 0).

Units: states non-dimensional (LU, LU/TU); angles in degrees;
separations and uncertainties reported in km.
"""

import numpy as np

from engine import crtbp, estimation, rendezvous
from engine.crtbp import MU


# --------------------------------------------------------------------------
# The chaser's LVLH frame and the camera measurement
# --------------------------------------------------------------------------

def chaser_lvlh_basis(chaser_state, mu=MU):
    """
    Radial, along-track and cross-track unit vectors of the chaser's
    LVLH frame about the Moon, expressed on the rotating-frame axes.

    rendezvous.lvlh_basis needs the Moon-centred inertial position and
    velocity.  The inertial frame of frames.rotating_to_inertial_states
    is the rotating frame turned by Rz(t), so a Moon-centred inertial
    state is Rz(t) applied to
        position  r - r_moon
        velocity  v + z_hat x (r - r_moon)
    (the second term is the velocity the rotating frame carries,
    relative to the Moon, which the frame carries too).  Building the
    basis from these rotating-frame components gives the same unit
    vectors turned back by Rz(t), and since the line of sight is
    turned back by the same rotation, every dot product with it, and so
    every angle, is the same as in the inertial frame.  The result does
    not depend on time.
    """
    chaser_state = np.asarray(chaser_state, dtype=float)
    position_from_moon = chaser_state[:3] - crtbp.moon_position(mu)
    z_hat = np.array([0.0, 0.0, 1.0])
    velocity_from_moon = chaser_state[3:] + np.cross(z_hat, position_from_moon)
    return rendezvous.lvlh_basis(np.concatenate([position_from_moon, velocity_from_moon]))


def camera_angles(target_state, chaser_state, mu=MU):
    """
    In-plane and out-of-plane angles (degrees) of the target seen from
    the chaser, on the chaser's LVLH axes.  The in-plane angle is in
    [0, 360), measured from radial towards along-track; the
    out-of-plane angle is the elevation above the chaser's orbit plane.
    """
    radial, along_track, cross_track = chaser_lvlh_basis(chaser_state, mu)
    line_of_sight = np.asarray(target_state, dtype=float)[:3] - np.asarray(chaser_state, dtype=float)[:3]
    # estimation.angle_pair gives atan2(rho.b2, rho.b1) and
    # atan2(rho.b3, sqrt((rho.b1)^2 + (rho.b2)^2)); the second equals
    # asin(rho.b3 / |rho|).
    return estimation.angle_pair(line_of_sight, radial, along_track, cross_track)


def camera_angles_jacobian(target_state, chaser_state, mu=MU):
    """
    Analytic (2, 6) Jacobian of the two camera angles with respect to
    the target's rotating-frame state, degrees per LU.

    The chaser's state, and so its LVLH basis, is known and does not
    depend on the target, so only the line of sight rho = r_target -
    r_chaser changes, with d(rho)/d(r_target) = identity.  The partials
    are then those of estimation.angle_pair_jacobian: both rows are
    perpendicular to rho (an angle cannot see a change of range) and
    scale as one over the range.  The velocity columns are zero.
    """
    radial, along_track, cross_track = chaser_lvlh_basis(chaser_state, mu)
    line_of_sight = np.asarray(target_state, dtype=float)[:3] - np.asarray(chaser_state, dtype=float)[:3]
    jacobian = np.zeros((2, 6))
    jacobian[:, :3] = estimation.angle_pair_jacobian(line_of_sight, radial, along_track, cross_track)
    return jacobian


def make_camera_measurement(chaser_state, mu=MU):
    """
    A measurement function target_state -> (2,) camera angles for one
    instant, with the chaser's known state at that instant held inside
    it.  It carries the attributes estimation.make_measurement gives,
    so the EKF, UKF, batch least squares and Fisher information take it
    unchanged: `wraps_at_360` (the in-plane angle wraps), `jacobian`
    (analytic, above) and `kind`.
    """
    chaser_state = np.array(chaser_state, dtype=float)

    def function(target_state):
        return camera_angles(target_state, chaser_state, mu)

    function.wraps_at_360 = True
    function.jacobian = lambda target_state: camera_angles_jacobian(target_state, chaser_state, mu)
    function.kind = "camera"
    function.chaser_state = chaser_state
    return function


def simulate_camera_measurements(target_states, chaser_states, times_nondim, noise_sigma, mask=None, every=1,
                                 seed=0, mu=MU):
    """
    Synthetic camera angles of a true target trajectory from a chaser.

    target_states, chaser_states : (n, 6) rotating-frame truth on one grid
    times_nondim                 : (n,) TU
    noise_sigma                  : (2,) one-sigma noise per angle, degrees
    mask                         : optional (n,) boolean, e.g. the camera's
                                   access mask (Sun, Earth, Moon exclusion)
    every                        : take every k-th eligible sample
    Returns the measurement list extended_kalman_filter expects, the
    same shape as estimation.simulate_measurements.
    """
    rng = np.random.default_rng(seed)
    noise_sigma = np.asarray(noise_sigma, dtype=float)
    eligible = np.arange(len(times_nondim)) if mask is None else np.where(mask)[0]
    measurements = []
    for index in eligible[::every]:
        function = make_camera_measurement(chaser_states[index], mu)
        value = function(target_states[index]) + rng.normal(0.0, noise_sigma)
        value[0] = value[0] % 360.0
        measurements.append({"time_nondim": float(times_nondim[index]), "function": function,
                             "value": value, "noise_sigma": noise_sigma, "index": int(index)})
    return measurements


def lvlh_delta_v_to_rotating(chaser_state, delta_v_lvlh_m_s, mu=MU):
    """
    A chaser burn given on its LVLH axes (radial, along-track,
    cross-track, m/s) as a rotating-frame velocity change, LU/TU.  An
    impulse changes the velocity by the same vector in the rotating and
    the inertial frame (the frame's own velocity z_hat x r depends only
    on position), so only the axes change.
    """
    radial, along_track, cross_track = chaser_lvlh_basis(chaser_state, mu)
    basis = np.vstack([radial, along_track, cross_track])
    delta_v_km_s = np.asarray(delta_v_lvlh_m_s, dtype=float) / 1000.0
    return crtbp.velocity_to_nondim(basis.T @ delta_v_km_s)


# --------------------------------------------------------------------------
# The linear model of the relative motion
# --------------------------------------------------------------------------

def linearised_chaser_states(target_states, target_stms, initial_offset, burn_index=None, burn_delta_v=None):
    """
    Chaser trajectory (n, 6) under linear relative dynamics about the
    target: the comparison case in which range should be unobservable.

    target_states : (n, 6) true target trajectory on the grid
    target_stms   : (n, 6, 6) the target's STM Phi(t_k, 0) on the grid
    initial_offset: (6,) target state minus chaser state at time zero,
                    delta(0)
    burn_index    : optional grid index at which the chaser burns
    burn_delta_v  : (3,) the chaser's rotating-frame delta-v, LU/TU

    The relative state delta = x_target - x_chaser is taken to obey the
    target's variational equations, delta(t) = Phi(t, 0) delta(0), so
        x_chaser(t) = x_target(t) - Phi(t, 0) delta(0).
    A chaser burn at t_b lowers delta by (0, delta_v) there, which then
    moves with Phi(t, t_b) = Phi(t, 0) Phi(t_b, 0)^-1, so for t > t_b
        x_chaser(t) = x_target(t) - Phi(t, 0) delta(0)
                      + Phi(t, 0) Phi(t_b, 0)^-1 (0, delta_v).
    Without a burn every line of sight r_target - r_chaser is the
    position part of Phi(t, 0) delta(0), so scaling delta(0) scales
    every line of sight and leaves every angle alone: the scale of the
    relative motion, and with it the range, cannot be seen.  A known
    burn adds a displacement that does not scale with delta(0) and
    breaks the tie; this is the classical fix.
    """
    target_states = np.asarray(target_states, dtype=float)
    initial_offset = np.asarray(initial_offset, dtype=float)
    chaser = np.zeros_like(target_states)
    for index in range(len(target_states)):
        chaser[index] = target_states[index] - target_stms[index] @ initial_offset
    if burn_index is not None:
        kick = np.concatenate([np.zeros(3), np.asarray(burn_delta_v, dtype=float)])
        stm_at_burn_inverse = np.linalg.inv(target_stms[burn_index])
        for index in range(burn_index + 1, len(target_states)):
            chaser[index] = chaser[index] + target_stms[index] @ stm_at_burn_inverse @ kick
    return chaser


# --------------------------------------------------------------------------
# Range and cross-range: what the angles see and what they do not
# --------------------------------------------------------------------------

def line_of_sight_unit(target_state, chaser_state):
    """Unit vector (3,) from the chaser to the target, rotating-frame axes."""
    line_of_sight = np.asarray(target_state, dtype=float)[:3] - np.asarray(chaser_state, dtype=float)[:3]
    return line_of_sight / np.linalg.norm(line_of_sight)


def range_and_cross_range_sigma_km(covariance, target_state, chaser_state):
    """
    Split a target position covariance into the part along the line of
    sight (range) and the part across it.

    covariance : (6, 6) or (3, 3), LU^2 in the position block
    Returns (range_sigma_km, cross_range_sigma_km): the one-sigma
    uncertainty along the unit line of sight u, sqrt(u^T P u), and the
    root of the position variance across it, sqrt(trace(Pi P Pi)) with
    Pi = I - u u^T the projector onto the plane across the line of
    sight, which is the two cross-range directions together.  The
    projector is used rather than trace(P) - u^T P u because the range
    variance can be 1e12 times the cross-range variance, and the
    subtraction would leave only rounding.
    """
    position_covariance = np.asarray(covariance, dtype=float)[:3, :3]
    unit = line_of_sight_unit(target_state, chaser_state)
    along = float(unit @ position_covariance @ unit)
    projector = np.eye(3) - np.outer(unit, unit)
    across = float(np.trace(projector @ position_covariance @ projector))
    return crtbp.length_to_km(np.sqrt(max(along, 0.0))), crtbp.length_to_km(np.sqrt(max(across, 0.0)))


def range_and_cross_range_errors_km(estimated_state, true_target_state, chaser_state):
    """
    Split a target position error into range and cross-range parts, km.
    The range error is signed (positive means the estimate is too far
    away); the cross-range error is the length of the rest.
    """
    error = np.asarray(estimated_state, dtype=float)[:3] - np.asarray(true_target_state, dtype=float)[:3]
    unit = line_of_sight_unit(true_target_state, chaser_state)
    along = float(error @ unit)
    across = error - along * unit
    return crtbp.length_to_km(along), crtbp.length_to_km(np.linalg.norm(across))


def line_of_sight_departure_arcsec(true_target_states, true_chaser_states, linear_chaser_states):
    """
    Angle (arcsec) between the true line of sight and the one the linear
    relative model predicts, at each grid time, shape (n,).  This is the
    size of the nonlinear signal: range becomes observable only where it
    is not small against the camera noise.
    """
    true_line = np.asarray(true_target_states, dtype=float)[:, :3] - np.asarray(true_chaser_states, dtype=float)[:, :3]
    linear_line = np.asarray(true_target_states, dtype=float)[:, :3] - np.asarray(linear_chaser_states, dtype=float)[:, :3]
    true_unit = true_line / np.linalg.norm(true_line, axis=1)[:, np.newaxis]
    linear_unit = linear_line / np.linalg.norm(linear_line, axis=1)[:, np.newaxis]
    cosine = np.clip(np.sum(true_unit * linear_unit, axis=1), -1.0, 1.0)
    # arccos loses precision for tiny angles; the cross product keeps it.
    sine = np.linalg.norm(np.cross(true_unit, linear_unit), axis=1)
    return np.degrees(np.arctan2(sine, cosine)) * 3600.0


def scale_direction_basis(initial_offset):
    """
    An orthonormal (6, 6) basis whose first column is the unit vector
    along delta(0), the direction in which the target state moves when
    the whole relative motion is scaled up or down (a farther target on
    a proportionally larger relative trajectory).  In the linear model
    the angles carry no information along it at all.

    Built as a Householder reflection Q = I - 2 w w^T with
    w = (e1 - n) / |e1 - n|, which swaps the first axis e1 with the unit
    offset n and leaves everything perpendicular to both alone; Q is
    symmetric and orthogonal.  Pass it to
    observability.fisher_information(basis=...) so the weak direction is
    accumulated on its own coordinate (see that docstring).
    """
    direction = np.asarray(initial_offset, dtype=float)
    direction = direction / np.linalg.norm(direction)
    first_axis = np.zeros(6)
    first_axis[0] = 1.0
    difference = first_axis - direction
    if np.linalg.norm(difference) < 1e-12:
        return np.eye(6)
    w = difference / np.linalg.norm(difference)
    return np.eye(6) - 2.0 * np.outer(w, w)


def covariance_from_information(information_in_basis, basis):
    """
    Covariance (6, 6) of the epoch state on the state's own axes from an
    information matrix accumulated on the columns of an orthonormal
    basis Q: P = Q A^-1 Q^T.  With the weak direction on the first
    coordinate the inverse of A is well behaved even when the
    information spans sixteen orders of magnitude, because elimination
    meets the large entries first and the small one last.
    """
    covariance_in_basis = np.linalg.inv(information_in_basis)
    covariance = basis @ covariance_in_basis @ basis.T
    return 0.5 * (covariance + covariance.T)


def scale_information_ratio(data_information_in_basis, prior_information_in_basis):
    """
    Information the measurements alone carry about the scale of the
    relative motion (the first coordinate of scale_direction_basis),
    divided by what the prior carries about it: the (0, 0) entries of
    the data-only and the prior-only information on that basis.  A
    diagonal entry is the information with the other five coordinates
    held known, so it is an upper limit on what the range can gain,
    not the bound itself.  Zero to rounding in the linear model; well
    above one where the angles, not the prior, can fix the range.
    Dimensionless.
    """
    return float(data_information_in_basis[0, 0] / prior_information_in_basis[0, 0])
