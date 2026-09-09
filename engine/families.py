"""
Seeds and continuation for the periodic orbit families other than the
L2 southern halos, which corrector.build_l2_southern_family handles.

Each builder returns a list of orbit dictionaries with the same keys as
the halo family (state0, period, jacobi, perilune_radius,
apolune_radius, stability_index, eigenvalues), so every family can be
stored, displayed and used identically.

Families
  * L1 and L2 halos, northern and southern.  The CRTBP is symmetric
    under z -> -z, so a northern family is the exact mirror of the
    southern one; it is produced by flipping the sign of z0 rather than
    by running the continuation again.
  * Planar Lyapunov orbits about L1 and L2, seeded from the linearised
    in-plane motion about the point and continued outward in x0.
  * Distant retrograde orbits (DROs) about the Moon, seeded from a
    two-body retrograde circular orbit and continued outward.

All quantities are non-dimensional (LU, TU).
"""

import numpy as np

from engine import corrector, crtbp
from engine.crtbp import MU


# --------------------------------------------------------------------------
# Linearised in-plane motion about a collinear point
# --------------------------------------------------------------------------

def in_plane_linear_mode(point="L2", mu=MU):
    """
    Frequency lambda and amplitude ratio k of the periodic in-plane mode
    of the motion linearised about a collinear libration point.

    About the point the in-plane equations are
        x'' - 2 y' = (1 + 2 c2) x,   y'' + 2 x' = (1 - c2) y
    with c2 = Uxx at the point minus one (equivalently -Uyy plus one).
    The oscillatory solution is x = A cos(lambda t), y = -k A sin(lambda t),
    with lambda from the characteristic equation
        lambda^4 + (c2 - 2) lambda^2 - (c2 - 1)(1 + 2 c2) = 0
    and k = 2 lambda / (lambda^2 + 1 - c2).

    Returns (x_point, lambda, k, c2).
    """
    x_point = crtbp.collinear_libration_points(mu)[point]
    hessian = crtbp.pseudo_potential_hessian(np.array([x_point, 0.0, 0.0, 0.0, 0.0, 0.0]), mu)
    c2 = 0.5 * (hessian[0, 0] - 1.0)
    lam_squared = 0.5 * (2.0 - c2 + np.sqrt((c2 - 2.0) ** 2 + 4.0 * (c2 - 1.0) * (1.0 + 2.0 * c2)))
    lam = np.sqrt(lam_squared)
    k = 2.0 * lam / (lam ** 2 + 1.0 - c2)
    return x_point, lam, k, c2


def lyapunov_seed(amplitude, point="L2", mu=MU):
    """
    Initial state [x0, 0, 0, 0, vy0, 0] of a planar Lyapunov orbit of
    the given x-amplitude (LU) from the linearised motion.

    At t = 0 the linear solution has x = A and y' = -k lambda A, so the
    motion is clockwise (retrograde) in the rotating frame about both
    L1 and L2 whatever the sign of A.  A positive amplitude starts at
    the crossing on the +x side of the point, a negative one on the -x
    side; the families are seeded at the crossing farthest from the
    Moon (+x for L2, -x for L1) so the continuation can step that
    crossing away from the point while the other side grows toward the
    Moon.  The corrector removes the error of the linear model.
    """
    x_point, lam, k, _ = in_plane_linear_mode(point, mu)
    return x_point + amplitude, -k * lam * amplitude


# --------------------------------------------------------------------------
# Planar continuation
# --------------------------------------------------------------------------

def continue_planar_family(first_orbit, mu=MU, step=0.01, max_members=60, direction=+1.0,
                           min_step=1e-5, max_step=0.05, stop_radius_from_moon=None, verbose=True):
    """
    Natural-parameter continuation of a planar symmetric family: x0 is
    stepped and vy0 re-converged with the planar corrector.  vy0 is
    predicted by linear extrapolation from the last two members.

    direction        : +1 steps x0 upward, -1 downward
    stop_radius_from_moon : stop when the initial point comes within this
                       distance (LU) of the Moon's centre
    The step halves when the corrector fails and grows when it converges
    quickly.  Returns a list of annotated orbit dictionaries.
    """
    family = [corrector.annotate_orbit(dict(first_orbit), mu)]
    while len(family) < max_members:
        last = family[-1]["state0"]
        last_period = family[-1]["period"]
        if len(family) >= 2:
            secant = family[-1]["state0"] - family[-2]["state0"]
            slope_vy = secant[4] / secant[0] if secant[0] != 0.0 else 0.0
            slope_period = (family[-1]["period"] - family[-2]["period"]) / secant[0] if secant[0] != 0.0 else 0.0
        else:
            slope_vy = 0.0
            slope_period = 0.0

        def looks_like_a_jump(candidate, vy0_guess, period_guess):
            # A converged solution far from the predictor, one whose
            # period is nothing like the neighbour's, or one with vy0 = 0
            # (the degenerate rectilinear "orbit" along the x-axis) is a
            # different solution branch, not the next member.
            vy0 = candidate["state0"][4]
            return (abs(vy0 - vy0_guess) > 0.05
                    or abs(candidate["period"] - period_guess) > 0.25 * last_period
                    or abs(vy0) < 1e-6)

        converged = None
        while step >= min_step:
            x0_guess = last[0] + direction * step
            vy0_guess = last[4] + slope_vy * direction * step
            period_guess = last_period + slope_period * direction * step
            try:
                candidate = corrector.correct_planar(x0_guess, vy0_guess, mu, fixed="x0")
            except (RuntimeError, ZeroDivisionError, FloatingPointError):
                step = step * 0.5
                continue
            if looks_like_a_jump(candidate, vy0_guess, period_guess):
                step = step * 0.5
                continue
            converged = candidate
            break
        if converged is None:
            if verbose:
                print("continuation stopped: step size below minimum")
            break

        annotated = corrector.annotate_orbit(converged, mu)
        if annotated["perilune_radius"] < crtbp.MOON_RADIUS_ND or (
                stop_radius_from_moon is not None and annotated["perilune_radius"] < stop_radius_from_moon):
            if verbose:
                print("continuation stopped: orbit reaches the Moon's vicinity")
            break

        family.append(annotated)
        if verbose:
            member = family[-1]
            print(f"member {len(family):3d}: x0 = {member['state0'][0]:.6f}  vy0 = {member['state0'][4]:.6f}  "
                  f"T = {member['period']:.5f}  C = {member['jacobi']:.6f}  "
                  f"r_p = {crtbp.length_to_km(member['perilune_radius']):9.1f} km  "
                  f"nu = {member['stability_index']:.3f}  ({member['iterations']} it)")

        if converged["iterations"] <= 3:
            step = min(step * 1.5, max_step)
        elif converged["iterations"] >= 6:
            step = step * 0.5
    return family


# --------------------------------------------------------------------------
# Family builders
# --------------------------------------------------------------------------

def build_lyapunov_family(point="L2", mu=MU, seed_amplitude=0.005, max_members=40, verbose=True):
    """
    Planar Lyapunov orbits about L1 or L2, from a small linear seed
    outward.  Stops when the initial point reaches the Moon's vicinity
    or max_members is reached.
    """
    direction = +1.0 if point == "L2" else -1.0
    x0, vy0 = lyapunov_seed(direction * seed_amplitude, point, mu)
    first = corrector.correct_planar(x0, vy0, mu, fixed="x0", verbose=verbose)
    if verbose:
        _, lam, _, _ = in_plane_linear_mode(point, mu)
        print(f"{point} Lyapunov seed: x0 = {x0:.6f}, vy0 = {vy0:.6f}; converged T = {first['period']:.6f} "
              f"(linear 2 pi / lambda = {2.0 * np.pi / lam:.6f})")
    return continue_planar_family(first, mu, step=0.005, max_members=max_members, direction=direction,
                                  max_step=0.02, stop_radius_from_moon=2.0 * crtbp.MOON_RADIUS_ND,
                                  verbose=verbose)


def dro_seed(radius, mu=MU):
    """
    Initial state [x0, 0, 0, 0, vy0, 0] of a distant retrograde orbit of
    the given radius (LU) about the Moon, on the far side of the Moon
    from the Earth, from a two-body retrograde circular orbit.

    Retrograde means clockwise in the inertial frame; at the far-side
    crossing the inertial velocity relative to the Moon is -sqrt(mu / r)
    in y.  Subtracting the frame rotation omega x r = r in y gives the
    rotating-frame vy0 = -(sqrt(mu / r) + r).
    """
    x0 = 1.0 - mu + radius
    vy0 = -(np.sqrt(mu / radius) + radius)
    return x0, vy0


def build_dro_family(mu=MU, seed_radius=0.03, max_radius=0.3, max_members=50, verbose=True):
    """
    Distant retrograde orbits about the Moon from seed_radius outward to
    max_radius (both LU; 0.3 LU is about 115,000 km, beyond which the
    orbit is no longer usefully Moon-centred).  DROs are linearly stable
    for a wide range of sizes, which the stability index (1 or below)
    shows.
    """
    x0, vy0 = dro_seed(seed_radius, mu)
    first = corrector.correct_planar(x0, vy0, mu, fixed="x0", verbose=verbose)
    if verbose:
        print(f"DRO seed: r = {crtbp.length_to_km(seed_radius):.0f} km, x0 = {x0:.6f}, vy0 = {vy0:.6f}; "
              f"converged T = {first['period']:.6f} TU = {crtbp.time_to_days(first['period']):.3f} d")
    family = continue_planar_family(first, mu, step=0.01, max_members=max_members, direction=+1.0,
                                    max_step=0.02, verbose=verbose)
    return [orbit for orbit in family if orbit["state0"][0] - (1.0 - mu) <= max_radius]


def mirror_family(family):
    """
    The northern counterpart of a southern halo family (or the reverse):
    z0 -> -z0.  Exact because the equations of motion are unchanged by
    z -> -z, vz -> -vz.
    """
    mirrored = []
    for orbit in family:
        copy = dict(orbit)
        copy["state0"] = np.array(orbit["state0"], dtype=float).copy()
        copy["state0"][2] = -copy["state0"][2]
        copy["state0"][5] = -copy["state0"][5]
        mirrored.append(copy)
    return mirrored


def is_southern(orbit, mu=MU):
    """
    True if the orbit's apolune lies below the xy-plane, which is what
    "southern" means in the NRHO literature (apolune over the Moon's
    south pole).
    """
    _, _, _, apolune_state = corrector.closest_and_farthest_approach(orbit, mu)
    return apolune_state[2] < 0.0


def build_l1_halo_family(mu=MU, seed_z_amplitude=0.02, stop_perilune_km=1800.0, verbose=True, **kwargs):
    """
    L1 halo family from the Richardson seed, continued toward the Moon
    until the perilune radius reaches stop_perilune_km.  The seed's
    north/south label is not trusted: the converged family is classified
    by where its apolune lies, and the returned family is the southern
    one (mirror it for the northern).
    """
    x0, z0, vy0, period_estimate = corrector.richardson_halo_guess(seed_z_amplitude, mu, "L1", southern=True)

    # The Richardson evaluation places the L1 seed at the crossing on the
    # Moon side of L1, from which the corrector does not converge (the
    # third-order expansion is poorest there, closest to the Moon).  The
    # same halo crosses the plane on the Earth side of L1 at nearly the
    # mirror-image x with the opposite vy, and from that crossing Newton
    # converges in a handful of iterations; the seed is reflected about
    # L1 before correcting.  Found empirically, see build_families.py.
    x_point = crtbp.collinear_libration_points(mu)["L1"]
    x0 = 2.0 * x_point - x0
    vy0 = -vy0
    if verbose:
        print(f"L1 Richardson seed (reflected to the Earth-side crossing): x0 = {x0:.6f}, z0 = {z0:.6f}, "
              f"vy0 = {vy0:.6f}, T ~ {period_estimate:.4f}")
    first = corrector.correct_halo(x0, z0, vy0, mu, fixed="z0", verbose=verbose)
    options = dict(initial_step=0.004, max_step=0.01, max_vy_change=0.03,
                   stop_perilune_radius=crtbp.length_to_nondim(stop_perilune_km))
    options.update(kwargs)
    family = corrector.continue_family(first, mu, verbose=verbose, **options)
    if not is_southern(family[0], mu):
        if verbose:
            print("seed converged onto the northern family; mirroring to southern")
        family = mirror_family(family)
    return family
