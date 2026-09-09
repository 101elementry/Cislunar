"""
Station keeping of an unstable periodic orbit by impulsive targeting.

A spacecraft placed on an unstable periodic orbit leaves it: any error
grows by the unstable eigenvalue of the monodromy matrix every
revolution.  Station keeping is a sequence of small manoeuvres that
keep the spacecraft near the reference orbit.  The scheme here is the
simplest defensible one:

  * The reference is the periodic orbit itself, sampled at control
    nodes spaced evenly in time.
  * At each node the estimated state (the true state plus navigation
    noise) is compared with the reference and a single impulsive
    manoeuvre is computed that brings the position back onto the
    reference at the next node.  The manoeuvre is found by Newton
    iteration on the position error using the STM, the same tool the
    differential corrector uses:
        r(t_next) is linear in dv through the upper-right 3x3 block of
        Phi(t_next, t_now), so dv = -Phi_rv^-1 (r_pred - r_ref), refined
        by re-integrating.
  * The manoeuvre is applied with an execution error and the true
    state is integrated to the next node.

The output is the delta-v history, from which a cost per year follows.
The cost rises with the stability index, which is the link between the
monodromy analysis and mission design.

All internal quantities are non-dimensional (LU, TU); the interface
takes and returns kilometres, metres per second and days.
"""

import numpy as np

from engine import crtbp
from engine.crtbp import MU


def targeting_manoeuvre(state, target_position, transfer_time, mu=MU, iterations=4):
    """
    Impulsive velocity change (3,) in LU/TU that makes a state arrive at
    target_position after transfer_time.  A few Newton steps on the
    position error using the STM's position-velocity block.
    """
    delta_v = np.zeros(3)
    for _ in range(iterations):
        departed = state.copy()
        departed[3:] = departed[3:] + delta_v
        sol = crtbp.propagate_with_stm(departed, transfer_time, mu)
        arrived, phi = crtbp.split_state_and_stm(sol.y[:, -1])
        position_error = arrived[:3] - target_position
        if np.linalg.norm(position_error) < 1e-12:
            break
        phi_rv = phi[0:3, 3:6]
        delta_v = delta_v - np.linalg.solve(phi_rv, position_error)
    return delta_v


def simulate(orbit, n_revolutions=10, nodes_per_revolution=1, node_offset=0.5,
             injection_error_km=1.0, injection_error_m_s=0.01,
             navigation_position_km=1.0, navigation_velocity_m_s=0.01,
             execution_error_fraction=0.01, seed=0, mu=MU):
    """
    Run the station-keeping loop for n_revolutions of a periodic orbit.

    nodes_per_revolution : manoeuvres per orbit period
    node_offset          : where the nodes sit, as a fraction of the
                           node interval: node k is at (k + offset) *
                           period / nodes_per_revolution.  With the
                           default 0.5 one node per revolution sits at
                           apolune for the halo families (whose initial
                           state is the perilune crossing) and several
                           nodes per revolution straddle perilune
                           without landing on it, where a manoeuvre is
                           expensive and ill-conditioned.
    injection_error_*    : one-sigma error of the initial state
    navigation_*         : one-sigma error of the estimated state at
                           each node
    execution_error_fraction : one-sigma proportional error applied to
                           each manoeuvre
    seed                 : random seed, so runs are repeatable

    Returns a dictionary
        node_times_days       : (n,) manoeuvre times
        delta_v_m_s           : (n, 3) manoeuvre vectors
        delta_v_magnitude_m_s : (n,)
        total_delta_v_m_s     : sum of magnitudes
        delta_v_per_year_m_s  : total scaled to 365.25 days
        position_error_km     : (n,) true position error at each node
                                before the manoeuvre
        trajectory            : (m, 6) true trajectory sampled through
                                the run
        trajectory_times_days : (m,)
    """
    rng = np.random.default_rng(seed)
    period = float(orbit["period"])
    interval = period / nodes_per_revolution
    n_nodes = n_revolutions * nodes_per_revolution

    # Reference orbit as a dense solution over one period.
    reference = crtbp.propagate(orbit["state0"], period, mu, dense_output=True)

    def reference_state(t):
        return reference.sol(np.mod(t, period))

    # Injection: start at the first node with an error.
    t_now = node_offset * interval
    true_state = reference_state(t_now).copy()
    true_state[:3] = true_state[:3] + rng.normal(0.0, injection_error_km / crtbp.LENGTH_UNIT_KM, 3)
    true_state[3:] = true_state[3:] + rng.normal(0.0, crtbp.velocity_to_nondim(injection_error_m_s / 1000.0), 3)

    node_times = []
    delta_vs = []
    position_errors = []
    trajectory = []
    trajectory_times = []

    for _ in range(n_nodes):
        t_next = t_now + interval
        estimated = true_state.copy()
        estimated[:3] = estimated[:3] + rng.normal(0.0, navigation_position_km / crtbp.LENGTH_UNIT_KM, 3)
        estimated[3:] = estimated[3:] + rng.normal(0.0, crtbp.velocity_to_nondim(navigation_velocity_m_s / 1000.0), 3)

        delta_v = targeting_manoeuvre(estimated, reference_state(t_next)[:3], interval, mu)
        executed = delta_v * (1.0 + rng.normal(0.0, execution_error_fraction, 3))

        position_errors.append(np.linalg.norm(true_state[:3] - reference_state(t_now)[:3]))
        node_times.append(t_now)
        delta_vs.append(executed)

        true_state = true_state.copy()
        true_state[3:] = true_state[3:] + executed
        t_eval = np.linspace(0.0, interval, 200)
        leg = crtbp.propagate(true_state, interval, mu, t_eval=t_eval)
        trajectory.append(leg.y.T)
        trajectory_times.append(t_now + leg.t)
        true_state = leg.y[:, -1].copy()
        t_now = t_next

    delta_vs = np.array(delta_vs)
    delta_v_m_s = crtbp.velocity_to_km_s(delta_vs) * 1000.0
    magnitudes = np.linalg.norm(delta_v_m_s, axis=1)
    elapsed_days = crtbp.time_to_days(n_nodes * interval)
    return {"node_times_days": crtbp.time_to_days(np.array(node_times)),
            "delta_v_m_s": delta_v_m_s,
            "delta_v_magnitude_m_s": magnitudes,
            "total_delta_v_m_s": float(magnitudes.sum()),
            "delta_v_per_year_m_s": float(magnitudes.sum() * 365.25 / elapsed_days),
            "position_error_km": crtbp.length_to_km(np.array(position_errors)),
            "trajectory": np.vstack(trajectory),
            "trajectory_times_days": crtbp.time_to_days(np.concatenate(trajectory_times))}
