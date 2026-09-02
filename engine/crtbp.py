"""
Circular restricted three-body problem (CRTBP) for the Earth-Moon system.

Everything in this module works in the standard non-dimensional rotating
frame unless a function name says otherwise:

  * length unit  LU = Earth-Moon distance
  * time unit    TU = 1 / (mean motion of the Moon), so one Moon
                 revolution about the barycentre takes 2*pi TU
  * mass unit    Earth mass + Moon mass

The frame rotates with the Moon, the origin is the Earth-Moon barycentre,
the x-axis points from Earth to Moon, z is along the angular momentum of
the Moon's orbit, and y completes the right-handed set.  Earth sits at
(-mu, 0, 0) and the Moon at (1 - mu, 0, 0).

State vectors are ordered [x, y, z, vx, vy, vz].
"""

import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------
# Mass parameter
# --------------------------------------------------------------------------

# mu = m_Moon / (m_Earth + m_Moon).  This value is consistent with the DE
# ephemeris GM values used below, so the unit conversions are self-consistent.
MU = 0.01215058560962404


# --------------------------------------------------------------------------
# Unit conversion (the one and only place dimensional numbers appear)
# --------------------------------------------------------------------------

# Gravitational parameters, km^3 / s^2 (DE430-era values).
GM_EARTH_KM3_S2 = 398600.4418
GM_MOON_KM3_S2 = 4902.800066

# Characteristic length: the mean Earth-Moon distance, km.
LENGTH_UNIT_KM = 384400.0

# Characteristic time.  In the CRTBP the mean motion of the primaries is
# n = sqrt(G (m1 + m2) / a^3), and the time unit is 1 / n so that n = 1 in
# non-dimensional units.  With the numbers above this is about 375190 s,
# roughly 4.34 days, and one full Moon revolution is 2*pi TU = 27.3 days.
TIME_UNIT_S = np.sqrt(LENGTH_UNIT_KM ** 3 / (GM_EARTH_KM3_S2 + GM_MOON_KM3_S2))

# Characteristic speed, km / s.
VELOCITY_UNIT_KM_S = LENGTH_UNIT_KM / TIME_UNIT_S

# Mean lunar radius, km, and the same in non-dimensional length units.
MOON_RADIUS_KM = 1737.0
MOON_RADIUS_ND = MOON_RADIUS_KM / LENGTH_UNIT_KM

SECONDS_PER_DAY = 86400.0


def length_to_km(length_nd):
    """Convert a non-dimensional length (LU) to kilometres."""
    return length_nd * LENGTH_UNIT_KM


def length_to_nondim(length_km):
    """Convert a length in kilometres to non-dimensional length units (LU)."""
    return length_km / LENGTH_UNIT_KM


def time_to_seconds(time_nd):
    """Convert a non-dimensional time (TU) to seconds."""
    return time_nd * TIME_UNIT_S


def time_to_days(time_nd):
    """Convert a non-dimensional time (TU) to days."""
    return time_nd * TIME_UNIT_S / SECONDS_PER_DAY


def time_to_nondim(time_s):
    """Convert a time in seconds to non-dimensional time units (TU)."""
    return time_s / TIME_UNIT_S


def velocity_to_km_s(velocity_nd):
    """Convert a non-dimensional speed (LU / TU) to km / s."""
    return velocity_nd * VELOCITY_UNIT_KM_S


def velocity_to_nondim(velocity_km_s):
    """Convert a speed in km / s to non-dimensional speed units (LU / TU)."""
    return velocity_km_s / VELOCITY_UNIT_KM_S


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def earth_position(mu=MU):
    """Position of the Earth in the rotating frame, non-dimensional."""
    return np.array([-mu, 0.0, 0.0])


def moon_position(mu=MU):
    """Position of the Moon in the rotating frame, non-dimensional."""
    return np.array([1.0 - mu, 0.0, 0.0])


def distances_to_primaries(position, mu=MU):
    """
    Distances from a point to the Earth (r1) and to the Moon (r2).

    position : array of shape (3,), non-dimensional rotating-frame position.
    Returns (r1, r2), both non-dimensional.
    """
    x = position[0]
    y = position[1]
    z = position[2]
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    return r1, r2


def distance_to_moon(states, mu=MU):
    """
    Distance to the Moon's centre for one state or an array of states.

    states : shape (6,) or (n, 6), non-dimensional.
    Returns a scalar or an array of length n, non-dimensional.
    """
    states = np.atleast_2d(states)
    dx = states[:, 0] - (1.0 - mu)
    dy = states[:, 1]
    dz = states[:, 2]
    r2 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if r2.size == 1:
        return r2[0]
    return r2


# --------------------------------------------------------------------------
# Pseudo-potential and its derivatives
# --------------------------------------------------------------------------

def pseudo_potential(state, mu=MU):
    """
    Effective (pseudo-) potential U of the rotating frame, non-dimensional.

        U = (x^2 + y^2) / 2 + (1 - mu) / r1 + mu / r2

    The first term is the centrifugal potential of the rotating frame, the
    other two are the gravitational potentials of Earth and Moon.  With this
    sign convention the equations of motion are
        x'' - 2 y' = dU/dx,  y'' + 2 x' = dU/dy,  z'' = dU/dz.
    """
    x = state[0]
    y = state[1]
    r1, r2 = distances_to_primaries(state[:3], mu)
    return 0.5 * (x ** 2 + y ** 2) + (1.0 - mu) / r1 + mu / r2


def pseudo_potential_gradient(state, mu=MU):
    """
    Gradient of the pseudo-potential, (dU/dx, dU/dy, dU/dz), non-dimensional.

    These are the accelerations before the Coriolis terms are added.
    """
    x = state[0]
    y = state[1]
    z = state[2]
    r1, r2 = distances_to_primaries(state[:3], mu)

    # Gravitational acceleration of each primary falls off as 1 / r^3 times
    # the displacement vector; the centrifugal term adds (x, y, 0).
    u_x = x - (1.0 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1.0 + mu) / r2 ** 3
    u_y = y - (1.0 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
    u_z = -(1.0 - mu) * z / r1 ** 3 - mu * z / r2 ** 3
    return np.array([u_x, u_y, u_z])


def pseudo_potential_hessian(state, mu=MU):
    """
    Hessian of the pseudo-potential, the 3x3 symmetric matrix of second
    partial derivatives d2U/dxi dxj, non-dimensional.

    Each gravitational term contributes  -GM (I / r^3 - 3 d d^T / r^5)
    where d is the displacement from the primary; the centrifugal term
    contributes diag(1, 1, 0).
    """
    x = state[0]
    y = state[1]
    z = state[2]
    r1, r2 = distances_to_primaries(state[:3], mu)

    # Displacements from Earth (d1) and from the Moon (d2).
    d1_x = x + mu
    d2_x = x - 1.0 + mu

    r1_cubed = r1 ** 3
    r2_cubed = r2 ** 3
    r1_fifth = r1 ** 5
    r2_fifth = r2 ** 5

    u_xx = (1.0
            - (1.0 - mu) / r1_cubed - mu / r2_cubed
            + 3.0 * (1.0 - mu) * d1_x ** 2 / r1_fifth
            + 3.0 * mu * d2_x ** 2 / r2_fifth)
    u_yy = (1.0
            - (1.0 - mu) / r1_cubed - mu / r2_cubed
            + 3.0 * (1.0 - mu) * y ** 2 / r1_fifth
            + 3.0 * mu * y ** 2 / r2_fifth)
    u_zz = (- (1.0 - mu) / r1_cubed - mu / r2_cubed
            + 3.0 * (1.0 - mu) * z ** 2 / r1_fifth
            + 3.0 * mu * z ** 2 / r2_fifth)
    u_xy = (3.0 * (1.0 - mu) * d1_x * y / r1_fifth
            + 3.0 * mu * d2_x * y / r2_fifth)
    u_xz = (3.0 * (1.0 - mu) * d1_x * z / r1_fifth
            + 3.0 * mu * d2_x * z / r2_fifth)
    u_yz = (3.0 * (1.0 - mu) * y * z / r1_fifth
            + 3.0 * mu * y * z / r2_fifth)

    return np.array([[u_xx, u_xy, u_xz],
                     [u_xy, u_yy, u_yz],
                     [u_xz, u_yz, u_zz]])


# --------------------------------------------------------------------------
# Equations of motion
# --------------------------------------------------------------------------

def equations_of_motion(t, state, mu=MU):
    """
    Time derivative of the six-element state in the rotating frame.

    state : [x, y, z, vx, vy, vz], non-dimensional.
    Returns d(state)/dt, non-dimensional.

    The accelerations are the pseudo-potential gradient plus the Coriolis
    terms 2 vy and -2 vx, which come from the frame rotating at unit rate
    about z.  The centrifugal part of the rotation is already inside U.
    """
    vx = state[3]
    vy = state[4]
    vz = state[5]
    u_x, u_y, u_z = pseudo_potential_gradient(state, mu)

    ax = 2.0 * vy + u_x
    ay = -2.0 * vx + u_y
    az = u_z
    return np.array([vx, vy, vz, ax, ay, az])


def jacobian(state, mu=MU):
    """
    Analytic Jacobian A = d(state')/d(state), a 6x6 matrix, non-dimensional.

    A = [[ 0,   I    ],
         [ Uxx, Omega]]

    where Uxx is the Hessian of the pseudo-potential and Omega is the
    Coriolis matrix [[0, 2, 0], [-2, 0, 0], [0, 0, 0]].
    """
    a_matrix = np.zeros((6, 6))
    a_matrix[0:3, 3:6] = np.eye(3)
    a_matrix[3:6, 0:3] = pseudo_potential_hessian(state, mu)
    a_matrix[3, 4] = 2.0
    a_matrix[4, 3] = -2.0
    return a_matrix


def equations_of_motion_with_stm(t, state_and_stm, mu=MU):
    """
    Time derivative of the 42-element vector [state (6), STM (36)].

    The state transition matrix Phi(t, 0) obeys Phi' = A(state) Phi with
    Phi(0) = identity, so the STM must be integrated alongside the state
    because A depends on the current position.  The 6x6 matrix is stored
    row-major in the last 36 entries.
    """
    state = state_and_stm[:6]
    phi = state_and_stm[6:].reshape((6, 6))

    state_derivative = equations_of_motion(t, state, mu)
    phi_derivative = jacobian(state, mu) @ phi

    return np.concatenate([state_derivative, phi_derivative.reshape(36)])


# --------------------------------------------------------------------------
# Integrals of motion
# --------------------------------------------------------------------------

def jacobi_constant(state, mu=MU):
    """
    Jacobi constant C = 2 U - v^2, non-dimensional.

    C is the only integral of the CRTBP.  Larger C means less energy
    (motion confined by the zero-velocity surfaces); the L1 and L2 values
    for Earth-Moon are about 3.188 and 3.172.
    """
    v_squared = state[3] ** 2 + state[4] ** 2 + state[5] ** 2
    return 2.0 * pseudo_potential(state, mu) - v_squared


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------

INTEGRATOR_METHOD = "DOP853"
INTEGRATOR_RTOL = 1e-12
INTEGRATOR_ATOL = 1e-12


def propagate(state0, t_final, mu=MU, events=None, dense_output=False, t_eval=None):
    """
    Propagate a six-element state from t = 0 to t = t_final.

    state0  : [x, y, z, vx, vy, vz], non-dimensional.
    t_final : non-dimensional time (may be negative).
    events  : optional solve_ivp event function(s) with signature (t, state).
    Returns the scipy OdeResult; sol.y has shape (6, n).
    """
    return solve_ivp(equations_of_motion,
                     (0.0, t_final),
                     np.asarray(state0, dtype=float),
                     method=INTEGRATOR_METHOD,
                     rtol=INTEGRATOR_RTOL,
                     atol=INTEGRATOR_ATOL,
                     args=(mu,),
                     events=events,
                     dense_output=dense_output,
                     t_eval=t_eval)


def propagate_with_stm(state0, t_final, mu=MU, events=None, dense_output=False, t_eval=None):
    """
    Propagate a six-element state together with the 6x6 state transition
    matrix from t = 0 (Phi = identity) to t = t_final.

    Returns the scipy OdeResult; sol.y has shape (42, n) with the state in
    rows 0-5 and the STM stored row-major in rows 6-41.
    """
    initial_vector = np.concatenate([np.asarray(state0, dtype=float),
                                     np.eye(6).reshape(36)])
    return solve_ivp(equations_of_motion_with_stm,
                     (0.0, t_final),
                     initial_vector,
                     method=INTEGRATOR_METHOD,
                     rtol=INTEGRATOR_RTOL,
                     atol=INTEGRATOR_ATOL,
                     args=(mu,),
                     events=events,
                     dense_output=dense_output,
                     t_eval=t_eval)


def split_state_and_stm(vector42):
    """Split a 42-element vector into the state (6,) and the STM (6, 6)."""
    return vector42[:6], vector42[6:].reshape((6, 6))


# --------------------------------------------------------------------------
# Collinear libration points
# --------------------------------------------------------------------------

def collinear_libration_points(mu=MU):
    """
    x-coordinates of L1, L2 and L3 on the Earth-Moon line, non-dimensional.

    The collinear points are where dU/dx = 0 on the x-axis.  Each is found
    by Newton iteration on the one-dimensional function starting from the
    usual (mu / 3)^(1/3) Hill-sphere estimates near the Moon and from the
    far side of the Earth for L3.
    """
    def u_x_on_axis(x):
        return pseudo_potential_gradient(np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0]), mu)[0]

    def u_xx_on_axis(x):
        return pseudo_potential_hessian(np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0]), mu)[0, 0]

    hill_radius = (mu / 3.0) ** (1.0 / 3.0)
    initial_guesses = {"L1": 1.0 - mu - hill_radius,
                       "L2": 1.0 - mu + hill_radius,
                       "L3": -1.0 - 5.0 * mu / 12.0}

    points = {}
    for name, x in initial_guesses.items():
        for _ in range(50):
            step = u_x_on_axis(x) / u_xx_on_axis(x)
            x = x - step
            if abs(step) < 1e-15:
                break
        points[name] = x
    return points
