"""
Validation checks for crtbp.py and corrector.py.  Run this file; it
prints every result and writes the halo family and two figures to the
output directory.

Checks:
  1. Jacobi constant drift over 10 TU of propagation.
  2. One column of the STM against a central finite difference.
  3. A converged L2 southern halo (the member closest to the 9:2 NRHO)
     with period and Jacobi constant for comparison with the JPL
     three-body periodic orbit catalogue.
  4. 3D plot of the family and stability index against perilune radius.
"""

import os

import numpy as np

import crtbp
import corrector
import plots
from crtbp import MU


def check_jacobi_drift(state0, duration=10.0, label=""):
    """Propagate for `duration` TU and report the largest |C(t) - C(0)|."""
    t_eval = np.linspace(0.0, duration, 2001)
    sol = crtbp.propagate(state0, duration, MU, t_eval=t_eval)
    jacobi = np.array([crtbp.jacobi_constant(state, MU) for state in sol.y.T])
    drift = np.max(np.abs(jacobi - jacobi[0]))
    verdict = "OK" if drift < 1e-10 else "FAIL"
    print(f"  {label:<28s} C0 = {jacobi[0]:.12f}   max |dC| = {drift:.3e}   "
          f"({verdict}, threshold 1e-10)")
    return drift


def check_stm_column(state0, t_final, column=0, step=1e-6):
    """
    Compare one STM column with a central finite difference of the flow.

    The column d(state(t)) / d(state0[column]) from the integrated STM is
    compared with [flow(state0 + h e) - flow(state0 - h e)] / (2 h).
    """
    sol = crtbp.propagate_with_stm(state0, t_final, MU)
    _, phi = crtbp.split_state_and_stm(sol.y[:, -1])

    perturbation = np.zeros(6)
    perturbation[column] = step
    forward = crtbp.propagate(state0 + perturbation, t_final, MU).y[:, -1]
    backward = crtbp.propagate(state0 - perturbation, t_final, MU).y[:, -1]
    finite_difference = (forward - backward) / (2.0 * step)

    analytic = phi[:, column]
    relative_error = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(analytic)
    names = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]
    print(f"  STM column d(state)/d({names[column]}) at t = {t_final:.4f} TU, "
          f"central difference h = {step:g}")
    print(f"    analytic : {np.array2string(analytic, precision=8)}")
    print(f"    finite   : {np.array2string(finite_difference, precision=8)}")
    print(f"    relative error (2-norm) = {relative_error:.3e}")
    return relative_error


def check_full_stm(state0, t_final, step=1e-6):
    """Largest relative column error over all six STM columns."""
    sol = crtbp.propagate_with_stm(state0, t_final, MU)
    _, phi = crtbp.split_state_and_stm(sol.y[:, -1])
    worst = 0.0
    for column in range(6):
        perturbation = np.zeros(6)
        perturbation[column] = step
        forward = crtbp.propagate(state0 + perturbation, t_final, MU).y[:, -1]
        backward = crtbp.propagate(state0 - perturbation, t_final, MU).y[:, -1]
        finite_difference = (forward - backward) / (2.0 * step)
        error = np.linalg.norm(phi[:, column] - finite_difference) / np.linalg.norm(phi[:, column])
        worst = max(worst, error)
    print(f"    worst relative column error over all six columns = {worst:.3e}")
    return worst


def report_orbit(orbit, title):
    """Print everything a catalogue comparison needs for one orbit."""
    state0 = orbit["state0"]
    r_peri, _, r_apo, _ = corrector.closest_and_farthest_approach(orbit, MU)
    nu, eigenvalues = corrector.stability_index(corrector.monodromy_matrix(orbit, MU))
    print(f"  {title}")
    print(f"    x0  = {state0[0]:.15f} LU")
    print(f"    z0  = {state0[2]:.15f} LU")
    print(f"    vy0 = {state0[4]:.15f} LU/TU   ({crtbp.velocity_to_km_s(state0[4]):.6f} km/s)")
    print(f"    period          = {orbit['period']:.12f} TU  =  {crtbp.time_to_days(orbit['period']):.6f} days")
    print(f"    Jacobi constant = {orbit['jacobi']:.12f}")
    print(f"    perilune radius = {crtbp.length_to_km(r_peri):10.1f} km   "
          f"(altitude {crtbp.length_to_km(r_peri) - crtbp.MOON_RADIUS_KM:8.1f} km)")
    print(f"    apolune radius  = {crtbp.length_to_km(r_apo):10.1f} km")
    print(f"    stability index = {nu:.6f}")
    print("    monodromy eigenvalues:")
    for value in eigenvalues:
        print(f"      {value.real:+.10e} {value.imag:+.10e}j   |lambda| = {abs(value):.10e}")
    product = np.prod(eigenvalues)
    print(f"    product of eigenvalues (should be 1) = {product.real:.12f} {product.imag:+.3e}j")


def main():
    os.makedirs(plots.OUTPUT_DIR, exist_ok=True)
    np.set_printoptions(linewidth=140)

    print("=" * 78)
    print("Units")
    print("=" * 78)
    print(f"  mu = {MU}")
    print(f"  1 LU = {crtbp.LENGTH_UNIT_KM:.1f} km")
    print(f"  1 TU = {crtbp.TIME_UNIT_S:.3f} s = {crtbp.TIME_UNIT_S / crtbp.SECONDS_PER_DAY:.6f} days")
    print(f"  1 LU/TU = {crtbp.VELOCITY_UNIT_KM_S:.6f} km/s")
    libration = crtbp.collinear_libration_points(MU)
    print(f"  L1 x = {libration['L1']:.12f}   L2 x = {libration['L2']:.12f}   L3 x = {libration['L3']:.12f}")
    print()

    print("=" * 78)
    print("1. Halo family: Richardson seed, correction, continuation to the NRHO regime")
    print("=" * 78)
    family = corrector.build_l2_southern_family(
        stop_perilune_radius=crtbp.length_to_nondim(1800.0),
        initial_step=0.004, max_step=0.01, max_vy_change=0.03, verbose=True)
    np.savez(plots.FAMILY_FILE, **corrector.family_to_arrays(family))
    print(f"  {len(family)} members saved to {plots.FAMILY_FILE}")
    print()

    large_halo = family[0]
    nrho = plots.pick_representative_nrho(family)

    print("=" * 78)
    print("2. Jacobi constant drift over 10 TU (DOP853, rtol = atol = 1e-12)")
    print("=" * 78)
    check_jacobi_drift(large_halo["state0"], 10.0, "large halo (first member)")
    check_jacobi_drift(nrho["state0"], 10.0, "9:2-like NRHO")
    print()

    print("=" * 78)
    print("3. State transition matrix against finite differences")
    print("=" * 78)
    check_stm_column(large_halo["state0"], large_halo["half_period"], column=0)
    check_full_stm(large_halo["state0"], large_halo["half_period"])
    print()

    print("=" * 78)
    print("4. Converged orbits for comparison with the JPL periodic orbit catalogue")
    print("=" * 78)
    report_orbit(large_halo, "First member (Richardson-seeded large halo)")
    print()
    report_orbit(nrho, f"Member closest to the 9:2 resonant period ({plots.PERIOD_9_2_TU:.5f} TU)")
    print()

    print("=" * 78)
    print("5. Family summary")
    print("=" * 78)
    print(f"  {'#':>3s} {'x0':>10s} {'z0':>10s} {'vy0':>10s} {'T [TU]':>9s} {'T [days]':>9s} "
          f"{'C':>10s} {'r_peri km':>10s} {'r_apo km':>10s} {'nu':>9s}")
    for k, orbit in enumerate(family):
        s = orbit["state0"]
        print(f"  {k + 1:3d} {s[0]:10.6f} {s[2]:10.6f} {s[4]:10.6f} {orbit['period']:9.5f} "
              f"{crtbp.time_to_days(orbit['period']):9.4f} {orbit['jacobi']:10.6f} "
              f"{crtbp.length_to_km(orbit['perilune_radius']):10.1f} "
              f"{crtbp.length_to_km(orbit['apolune_radius']):10.1f} {orbit['stability_index']:9.3f}")
    print()

    print("=" * 78)
    print("6. Figures")
    print("=" * 78)
    print("  saved", plots.plot_family_3d(family))
    print("  saved", plots.plot_stability_index(family))


if __name__ == "__main__":
    main()
