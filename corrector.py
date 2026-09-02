"""
Differential correction and continuation of halo orbits in the Earth-Moon
circular restricted three-body problem.

A halo orbit is symmetric about the xz-plane.  If the spacecraft starts on
that plane (y = 0) with its velocity perpendicular to the plane
(vx = vz = 0), and the next crossing of the plane is also perpendicular,
then the mirror theorem guarantees the orbit is periodic with period equal
to twice the crossing time.  The corrector enforces exactly that.

All quantities are non-dimensional (LU, TU) unless stated otherwise.
Orbits are passed around as plain dictionaries with keys
    state0   : [x0, 0, z0, 0, vy0, 0] initial state on the xz-plane
    period   : orbital period, TU
    jacobi   : Jacobi constant
"""

import numpy as np
from scipy.optimize import minimize_scalar

import crtbp
from crtbp import MU


# --------------------------------------------------------------------------
# Crossing event
# --------------------------------------------------------------------------

def make_xz_plane_crossing_event(direction):
    """
    Build a terminal solve_ivp event that fires when y = 0 is crossed in
    the given direction (+1 for increasing y, -1 for decreasing y).

    The initial state already sits on y = 0, so the direction must be set
    opposite to the initial vy; otherwise solve_ivp would report a crossing
    at t = 0 on its very first step.
    """
    def xz_plane_crossing(t, state_and_stm, mu):
        return state_and_stm[1]
    xz_plane_crossing.terminal = True
    xz_plane_crossing.direction = direction
    return xz_plane_crossing


def integrate_to_next_crossing(state0, mu=MU, t_max=10.0):
    """
    Integrate the state and STM from a point on the xz-plane to the next
    crossing of that plane.

    state0 : [x0, 0, z0, 0, vy0, 0], non-dimensional.
    Returns (t_cross, state_cross, stm_cross): the crossing time in TU,
    the 6-element state there and the 6x6 STM Phi(t_cross, 0).
    Raises RuntimeError if no crossing is found before t_max.
    """
    direction = -np.sign(state0[4])
    event = make_xz_plane_crossing_event(direction)
    sol = crtbp.propagate_with_stm(state0, t_max, mu, events=event)

    if len(sol.t_events[0]) == 0:
        raise RuntimeError("no xz-plane crossing found before t_max")

    t_cross = sol.t_events[0][0]
    state_cross, stm_cross = crtbp.split_state_and_stm(sol.y_events[0][0])
    return t_cross, state_cross, stm_cross


# --------------------------------------------------------------------------
# Differential corrector
# --------------------------------------------------------------------------

def correct_halo(x0, z0, vy0, mu=MU, fixed="z0", tolerance=1e-11,
                 max_iterations=30, verbose=False):
    """
    Newton-correct a perpendicular xz-plane crossing into a periodic halo.

    x0, z0, vy0 : initial guess for the state [x0, 0, z0, 0, vy0, 0].
    fixed       : which of "x0", "z0", "vy0" to hold constant.  The other
                  two are adjusted to drive the crossing velocities
                  vx(T/2) and vz(T/2) to zero.

    The derivatives come from the STM at the crossing.  Because the crossing
    time itself moves when the initial state changes, each STM column is
    corrected by the term -(acceleration) * (d t_cross / d q), with
    d t_cross / d q = -Phi[y, q] / vy taken from y(t_cross) = 0.

    Returns an orbit dictionary (see module docstring) plus the keys
    "iterations", "residual" and "half_period".
    Raises RuntimeError if Newton does not converge.
    """
    free_columns = {"z0": (0, 4), "x0": (2, 4), "vy0": (0, 2)}[fixed]

    state0 = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])

    for iteration in range(max_iterations):
        t_cross, state_cross, stm = integrate_to_next_crossing(state0, mu)

        residual = np.array([state_cross[3], state_cross[5]])
        residual_norm = np.linalg.norm(residual)
        if verbose:
            print(f"  iter {iteration:2d}: |vx, vz| at crossing = {residual_norm:.3e}")

        if residual_norm < tolerance:
            period = 2.0 * t_cross
            return {"state0": state0.copy(),
                    "period": period,
                    "half_period": t_cross,
                    "jacobi": crtbp.jacobi_constant(state0, mu),
                    "iterations": iteration,
                    "residual": residual_norm}

        # Accelerations at the crossing, needed for the time-shift term.
        derivative_cross = crtbp.equations_of_motion(t_cross, state_cross, mu)
        ax_cross = derivative_cross[3]
        az_cross = derivative_cross[5]
        vy_cross = state_cross[4]

        # Two-by-two Jacobian of (vx, vz) at the crossing with respect to
        # the two free initial components.
        jac = np.zeros((2, 2))
        for k, column in enumerate(free_columns):
            time_shift = -stm[1, column] / vy_cross
            jac[0, k] = stm[3, column] + ax_cross * time_shift
            jac[1, k] = stm[5, column] + az_cross * time_shift

        correction = np.linalg.solve(jac, -residual)
        for k, column in enumerate(free_columns):
            state0[column] = state0[column] + correction[k]

    raise RuntimeError(f"corrector did not converge in {max_iterations} iterations "
                       f"(residual {residual_norm:.3e})")


# --------------------------------------------------------------------------
# Third-order analytic seed (Richardson 1980)
# --------------------------------------------------------------------------

def richardson_halo_guess(z_amplitude, mu=MU, point="L2", southern=True):
    """
    Third-order Lindstedt-Poincare approximation of a halo orbit about L1
    or L2, evaluated at the xz-plane crossing nearest the Moon.

    z_amplitude : out-of-plane amplitude Az, non-dimensional LU.
    southern    : True for the southern family (z < 0 at the near-Moon
                  crossing, so most of the orbit lies below the plane).

    Returns (x0, z0, vy0, period_estimate) in the barycentric rotating
    frame.  Only the state is needed to seed the corrector; the period is
    for sanity checking.

    The expansion is written in a frame centred at the libration point and
    scaled by its distance gamma from the Moon, which is why the result is
    rescaled at the end.  The coefficient formulas are Richardson's and are
    transcribed in full rather than simplified.
    """
    libration = crtbp.collinear_libration_points(mu)
    x_point = libration[point]
    gamma = abs(x_point - (1.0 - mu))

    # Legendre-type coefficients c_n of the expanded potential.
    def c_coefficient(n):
        if point == "L1":
            return (mu + (-1.0) ** n * (1.0 - mu) * gamma ** (n + 1)
                    / (1.0 - gamma) ** (n + 1)) / gamma ** 3
        return ((-1.0) ** n * mu + (-1.0) ** n * (1.0 - mu) * gamma ** (n + 1)
                / (1.0 + gamma) ** (n + 1)) / gamma ** 3

    c2 = c_coefficient(2)
    c3 = c_coefficient(3)
    c4 = c_coefficient(4)

    # In-plane linear frequency lambda from the characteristic quartic.
    lam_squared = 0.5 * (2.0 - c2 + np.sqrt((c2 - 2.0) ** 2 + 4.0 * (c2 - 1.0) * (1.0 + 2.0 * c2)))
    lam = np.sqrt(lam_squared)
    k = 2.0 * lam / (lam ** 2 + 1.0 - c2)
    delta = lam ** 2 - c2

    d1 = 3.0 * lam ** 2 / k * (k * (6.0 * lam ** 2 - 1.0) - 2.0 * lam)
    d2 = 8.0 * lam ** 2 / k * (k * (11.0 * lam ** 2 - 1.0) - 2.0 * lam)

    a21 = 3.0 * c3 * (k ** 2 - 2.0) / (4.0 * (1.0 + 2.0 * c2))
    a22 = 3.0 * c3 / (4.0 * (1.0 + 2.0 * c2))
    a23 = -3.0 * c3 * lam / (4.0 * k * d1) * (3.0 * k ** 3 * lam - 6.0 * k * (k - lam) + 4.0)
    a24 = -3.0 * c3 * lam / (4.0 * k * d1) * (2.0 + 3.0 * k * lam)
    b21 = -3.0 * c3 * lam / (2.0 * d1) * (3.0 * k * lam - 4.0)
    b22 = 3.0 * c3 * lam / d1
    d21 = -c3 / (2.0 * lam ** 2)

    a31 = (-9.0 * lam / (4.0 * d2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k ** 2))
           + (9.0 * lam ** 2 + 1.0 - c2) / (2.0 * d2)
           * (3.0 * c3 * (2.0 * a23 - k * b21) + c4 * (2.0 + 3.0 * k ** 2)))
    a32 = (-1.0 / d2 * (9.0 * lam / 4.0 * (4.0 * c3 * (k * a24 - b22) + k * c4)
                        + 1.5 * (9.0 * lam ** 2 + 1.0 - c2)
                        * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)))
    b31 = (3.0 / (8.0 * d2) * (8.0 * lam * (3.0 * c3 * (k * b21 - 2.0 * a23) - c4 * (2.0 + 3.0 * k ** 2))
                               + (9.0 * lam ** 2 + 1.0 + 2.0 * c2)
                               * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k ** 2))))
    b32 = (1.0 / d2 * (9.0 * lam * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
                       + 3.0 / 8.0 * (9.0 * lam ** 2 + 1.0 + 2.0 * c2)
                       * (4.0 * c3 * (k * a24 - b22) + k * c4)))
    d31 = 3.0 / (64.0 * lam ** 2) * (4.0 * c3 * a24 + c4)
    d32 = 3.0 / (64.0 * lam ** 2) * (4.0 * c3 * (a23 - d21) + c4 * (4.0 + k ** 2))

    s1 = (1.0 / (2.0 * lam * (lam * (1.0 + k ** 2) - 2.0 * k))
          * (1.5 * c3 * (2.0 * a21 * (k ** 2 - 2.0) - a23 * (k ** 2 + 2.0) - 2.0 * k * b21)
             - 3.0 / 8.0 * c4 * (3.0 * k ** 4 - 8.0 * k ** 2 + 8.0)))
    s2 = (1.0 / (2.0 * lam * (lam * (1.0 + k ** 2) - 2.0 * k))
          * (1.5 * c3 * (2.0 * a22 * (k ** 2 - 2.0) + a24 * (k ** 2 + 2.0) + 2.0 * k * b22 + 5.0 * d21)
             + 3.0 / 8.0 * c4 * (12.0 - k ** 2)))
    l1 = -1.5 * c3 * (2.0 * a21 + a23 + 5.0 * d21) - 3.0 / 8.0 * c4 * (12.0 - k ** 2) + 2.0 * lam ** 2 * s1
    l2 = 1.5 * c3 * (a24 - 2.0 * a22) + 9.0 / 8.0 * c4 + 2.0 * lam ** 2 * s2

    # Amplitudes in the gamma-scaled frame.  The amplitude constraint
    # l1 Ax^2 + l2 Az^2 + delta = 0 links the in-plane and out-of-plane
    # amplitudes: a halo only exists for Ax above a minimum size.
    az = z_amplitude / gamma
    ax = np.sqrt(-(l2 * az ** 2 + delta) / l1)

    # Frequency correction and period.
    omega = 1.0 + s1 * ax ** 2 + s2 * az ** 2
    period_estimate = 2.0 * np.pi / (lam * omega)

    # Sign of the out-of-plane part.  Richardson labels the families by
    # m = 1, 3 with delta_n = 2 - m.  With the phase choice used here
    # (tau1 = 0 at the crossing nearest the Moon) the family whose apolune
    # lies below the xy-plane, which is what "southern" means in the NRHO
    # literature, turns out to need z > 0 at this crossing.  The sign was
    # fixed by propagating the converged orbit and checking where apolune
    # ends up, not by trusting the label.
    delta_n = 1.0 if southern else -1.0

    # Evaluate the series at phase tau1 = 0: x is at its minimum (toward
    # the Moon), y = 0, and vx = vz = 0 by symmetry.
    x_local = (a21 * ax ** 2 + a22 * az ** 2 - ax
               + (a23 * ax ** 2 - a24 * az ** 2)
               + (a31 * ax ** 3 - a32 * ax * az ** 2))
    z_local = delta_n * (az + d21 * ax * az * (1.0 - 3.0)
                         + (d32 * az * ax ** 2 - d31 * az ** 3))
    vy_local = lam * omega * (k * ax
                              + 2.0 * (b21 * ax ** 2 - b22 * az ** 2)
                              + 3.0 * (b31 * ax ** 3 - b32 * ax * az ** 2))

    # Back to the barycentric rotating frame.  Local x points away from
    # the Moon for L2 (toward increasing x), which is the same direction
    # as the rotating-frame x-axis; for L1 it points toward the Earth.
    if point == "L2":
        x0 = x_point + gamma * x_local
    else:
        x0 = x_point - gamma * x_local
    z0 = gamma * z_local
    vy0 = gamma * vy_local
    if point == "L1":
        vy0 = -vy0

    return x0, z0, vy0, period_estimate


# --------------------------------------------------------------------------
# Orbit properties
# --------------------------------------------------------------------------

def propagate_orbit(orbit, mu=MU, n_points=2000):
    """
    Sample one full period of a periodic orbit at n_points equally spaced
    times.  Returns (t, states) with states of shape (n_points, 6).
    """
    t_eval = np.linspace(0.0, orbit["period"], n_points)
    sol = crtbp.propagate(orbit["state0"], orbit["period"], mu, t_eval=t_eval)
    return sol.t, sol.y.T


def monodromy_matrix(orbit, mu=MU):
    """
    Monodromy matrix Phi(T, 0) of a periodic orbit: the STM integrated over
    one full period from the orbit's initial state.

    Its eigenvalues come in reciprocal pairs (lambda, 1 / lambda) because
    the flow is Hamiltonian, and one pair equals unity: one for the
    periodicity itself and one for the existence of the Jacobi constant.
    """
    sol = crtbp.propagate_with_stm(orbit["state0"], orbit["period"], mu)
    _, phi = crtbp.split_state_and_stm(sol.y[:, -1])
    return phi


def stability_index(monodromy):
    """
    Stability index of a periodic orbit from its monodromy matrix.

    For each reciprocal eigenvalue pair the index is
        nu = (lambda + 1 / lambda) / 2,
    which is real for stable (unit-circle) and real-hyperbolic pairs.  The
    function returns the largest |nu| over the non-trivial pairs; |nu| <= 1
    means all pairs are on the unit circle (linearly stable), |nu| > 1 means
    the orbit has an unstable direction.  Values in the single digits are
    the NRHO regime; classical halos near L2 have |nu| in the hundreds.
    """
    eigenvalues = np.linalg.eigvals(monodromy)
    # The trivial unit pair is removed by dropping the two eigenvalues
    # closest to 1; the remaining four form two reciprocal pairs.
    order = np.argsort(np.abs(eigenvalues - 1.0))
    nontrivial = eigenvalues[order[2:]]
    indices = 0.5 * (nontrivial + 1.0 / nontrivial)
    return np.max(np.abs(indices.real)), eigenvalues


def closest_and_farthest_approach(orbit, mu=MU):
    """
    Perilune and apolune of a periodic orbit.

    Returns (perilune_radius, perilune_state, apolune_radius, apolune_state)
    with radii in LU.  A coarse sample over one period locates each
    extremum, then a bounded scalar minimisation on the dense solution
    refines it.
    """
    sol = crtbp.propagate(orbit["state0"], orbit["period"], mu, dense_output=True)
    t_coarse = np.linspace(0.0, orbit["period"], 4000)
    r_coarse = crtbp.distance_to_moon(sol.sol(t_coarse).T, mu)

    def radius_at(t):
        return crtbp.distance_to_moon(sol.sol(t), mu)

    def refine(index, sign):
        t_low = t_coarse[max(index - 1, 0)]
        t_high = t_coarse[min(index + 1, len(t_coarse) - 1)]
        result = minimize_scalar(lambda t: sign * radius_at(t), bounds=(t_low, t_high),
                                 method="bounded", options={"xatol": 1e-12})
        return sign * result.fun, sol.sol(result.x)

    perilune_radius, perilune_state = refine(int(np.argmin(r_coarse)), 1.0)
    apolune_radius, apolune_state = refine(int(np.argmax(r_coarse)), -1.0)
    return perilune_radius, perilune_state, apolune_radius, apolune_state


def perilune(orbit, mu=MU):
    """Closest approach to the Moon: (radius in LU, state at that instant)."""
    radius, state, _, _ = closest_and_farthest_approach(orbit, mu)
    return radius, state


def apolune(orbit, mu=MU):
    """Farthest point from the Moon: (radius in LU, state at that instant)."""
    _, _, radius, state = closest_and_farthest_approach(orbit, mu)
    return radius, state


# --------------------------------------------------------------------------
# Continuation
# --------------------------------------------------------------------------

def continue_family(first_orbit, mu=MU, initial_step=0.005, max_step=0.02,
                    min_step=1e-6, max_vy_change=0.05, max_members=400,
                    stop_perilune_radius=None, verbose=True):
    """
    Natural-parameter continuation along a halo family.

    Starting from one converged orbit, each step fixes one of the initial
    coordinates x0 or z0 at a new value and re-converges the other two
    unknowns.  The stepped coordinate is whichever changed most between
    the last two members, which keeps the walk moving through folds in
    either coordinate.  The other coordinates are predicted by linear
    extrapolation from the previous two members.

    Deep in the NRHO regime the initial point sits at perilune and vy0
    changes very quickly with z0, so the step is also capped so that the
    predicted change in vy0 stays below max_vy_change.  A converged
    solution that lands far from the predictor is treated as a jump onto a
    different family and rejected, which halves the step.

    The step is halved when the corrector fails or converges slowly and
    grown when it converges quickly.  The walk stops when max_members is
    reached, the perilune radius drops below stop_perilune_radius (LU),
    the step shrinks below min_step, or the initial state ends up inside
    the Moon.

    Returns a list of orbit dictionaries, each with the extra keys
    "perilune_radius", "apolune_radius", "stability_index" and
    "eigenvalues".
    """
    def annotate(orbit):
        radius_min, _, radius_max, _ = closest_and_farthest_approach(orbit, mu)
        nu, eigenvalues = stability_index(monodromy_matrix(orbit, mu))
        orbit["perilune_radius"] = radius_min
        orbit["apolune_radius"] = radius_max
        orbit["stability_index"] = nu
        orbit["eigenvalues"] = eigenvalues
        return orbit

    def looks_like_a_jump(guess, converged_state):
        change = np.abs(converged_state - guess)
        return change[0] > 0.02 or change[2] > 0.02 or change[4] > 0.2

    family = [annotate(dict(first_orbit))]

    # First step: grow the orbit, meaning push z0 away from zero.
    step = initial_step
    stepped = "z0"
    direction = np.sign(first_orbit["state0"][2])

    while len(family) < max_members:
        last = family[-1]["state0"]
        if len(family) >= 2:
            secant = family[-1]["state0"] - family[-2]["state0"]
            # Step whichever position coordinate is moving fastest.
            if abs(secant[0]) > abs(secant[2]):
                stepped = "x0"
            else:
                stepped = "z0"
            column = 0 if stepped == "x0" else 2
            direction = np.sign(secant[column])
            slope = secant / secant[column]
        else:
            column = 2
            slope = np.zeros(6)
            slope[column] = 1.0

        # Cap the step so the predicted vy0 change stays moderate.
        if abs(slope[4]) > 0.0:
            step = min(step, max_vy_change / abs(slope[4]))

        converged = None
        while step >= min_step:
            guess = last + slope * direction * step
            try:
                candidate = correct_halo(guess[0], guess[2], guess[4], mu, fixed=stepped)
            except (RuntimeError, np.linalg.LinAlgError):
                step = step * 0.5
                continue
            if looks_like_a_jump(guess, candidate["state0"]):
                step = step * 0.5
                continue
            converged = candidate
            break

        if converged is None:
            if verbose:
                print("continuation stopped: step size below minimum")
            break

        r_start = crtbp.distance_to_moon(converged["state0"], mu)
        if r_start < crtbp.MOON_RADIUS_ND:
            if verbose:
                print("continuation stopped: initial state inside the Moon")
            break

        family.append(annotate(converged))

        if verbose:
            member = family[-1]
            print(f"member {len(family):3d}: step {stepped} {direction * step:+.5f}  "
                  f"x0 = {member['state0'][0]:.6f}  z0 = {member['state0'][2]:.6f}  "
                  f"vy0 = {member['state0'][4]:.6f}  T = {member['period']:.5f}  "
                  f"C = {member['jacobi']:.6f}  r_p = {crtbp.length_to_km(member['perilune_radius']):8.1f} km  "
                  f"nu = {member['stability_index']:.3f}  ({member['iterations']} it)")

        if stop_perilune_radius is not None and family[-1]["perilune_radius"] < stop_perilune_radius:
            if verbose:
                print("continuation stopped: perilune radius below target")
            break

        if converged["iterations"] <= 3:
            step = min(step * 1.5, max_step)
        elif converged["iterations"] >= 6:
            step = step * 0.5

    return family


def family_to_arrays(family):
    """Collect the per-member scalars of a family into a dictionary of arrays."""
    return {"state0": np.array([orbit["state0"] for orbit in family]),
            "period": np.array([orbit["period"] for orbit in family]),
            "jacobi": np.array([orbit["jacobi"] for orbit in family]),
            "perilune_radius": np.array([orbit["perilune_radius"] for orbit in family]),
            "apolune_radius": np.array([orbit["apolune_radius"] for orbit in family]),
            "stability_index": np.array([orbit["stability_index"] for orbit in family]),
            "eigenvalues": np.array([orbit["eigenvalues"] for orbit in family])}


def arrays_to_family(arrays):
    """Inverse of family_to_arrays."""
    family = []
    for k in range(len(arrays["period"])):
        family.append({"state0": arrays["state0"][k],
                       "period": arrays["period"][k],
                       "jacobi": arrays["jacobi"][k],
                       "perilune_radius": arrays["perilune_radius"][k],
                       "apolune_radius": arrays["apolune_radius"][k],
                       "stability_index": arrays["stability_index"][k],
                       "eigenvalues": arrays["eigenvalues"][k]})
    return family


def build_l2_southern_family(mu=MU, seed_z_amplitude=0.03, verbose=True, **kwargs):
    """
    Seed an L2 southern halo from the Richardson approximation, converge it,
    then continue toward the NRHO regime.  Returns the family list.
    """
    x0, z0, vy0, period_estimate = richardson_halo_guess(seed_z_amplitude, mu, "L2", southern=True)
    if verbose:
        print(f"Richardson seed: x0 = {x0:.6f}, z0 = {z0:.6f}, vy0 = {vy0:.6f}, "
              f"T ~ {period_estimate:.4f}")
    first = correct_halo(x0, z0, vy0, mu, fixed="z0", verbose=verbose)
    if verbose:
        print(f"converged seed:  x0 = {first['state0'][0]:.6f}, z0 = {first['state0'][2]:.6f}, "
              f"vy0 = {first['state0'][4]:.6f}, T = {first['period']:.6f}, C = {first['jacobi']:.6f}")
    return continue_family(first, mu, verbose=verbose, **kwargs)
