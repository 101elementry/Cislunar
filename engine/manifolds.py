"""
Stable and unstable invariant manifolds of a periodic orbit.

A periodic orbit with a real hyperbolic pair of monodromy eigenvalues
(lambda, 1 / lambda) has a one-dimensional unstable direction and a
one-dimensional stable direction at every point along it.  Following
trajectories that leave the orbit along the unstable direction traces
the unstable manifold; following trajectories backward in time along
the stable direction traces the stable manifold, which is the set of
trajectories that arrive on the orbit.  These are the natural transfer
paths to and from the orbit and cost no propellant to ride.

Procedure (the standard one)
  1. Eigen-decompose the monodromy matrix; take the eigenvector of the
     largest-magnitude real eigenvalue as the unstable direction at the
     initial point, and that of its reciprocal as the stable one.
  2. Transport the direction to other points on the orbit with the STM:
     v(t) = Phi(t, 0) v(0).  This is exact for the linearised flow.
  3. Displace the point on the orbit by a small distance along the
     direction (both signs) and integrate: forward for the unstable
     manifold, backward for the stable one.

All quantities are non-dimensional (LU, TU).
"""

import numpy as np

from engine import corrector, crtbp
from engine.crtbp import MU


def hyperbolic_directions(monodromy):
    """
    (unstable_direction, stable_direction, unstable_eigenvalue) from a
    monodromy matrix, each direction a real unit 6-vector.

    The unstable direction belongs to the eigenvalue of largest modulus;
    the stable one to the eigenvalue of smallest modulus (its
    reciprocal).  Raises ValueError if the largest eigenvalue is not
    real and larger than one in modulus, meaning the orbit has no real
    hyperbolic pair (a linearly stable orbit such as a DRO).
    """
    eigenvalues, eigenvectors = np.linalg.eig(monodromy)
    unstable_index = int(np.argmax(np.abs(eigenvalues)))
    stable_index = int(np.argmin(np.abs(eigenvalues)))
    unstable_eigenvalue = eigenvalues[unstable_index]
    if abs(unstable_eigenvalue.imag) > 1e-6 * abs(unstable_eigenvalue) or abs(unstable_eigenvalue) < 1.0 + 1e-6:
        raise ValueError("the orbit has no real hyperbolic eigenvalue pair, so no manifolds")

    def real_unit(vector):
        # The eigenvector of a real eigenvalue is real up to a complex
        # phase; rotate the phase away and normalise.
        phase = np.exp(-1j * np.angle(vector[np.argmax(np.abs(vector))]))
        real_vector = (vector * phase).real
        return real_vector / np.linalg.norm(real_vector)

    return (real_unit(eigenvectors[:, unstable_index]),
            real_unit(eigenvectors[:, stable_index]),
            unstable_eigenvalue.real)


def make_impact_events(mu=MU):
    """
    Terminal solve_ivp events that stop a manifold trajectory when it
    hits the Moon or the Earth, so a branch that crashes is drawn only
    to the surface.
    """
    moon = crtbp.moon_position(mu)
    earth = crtbp.earth_position(mu)
    earth_radius = 6371.0 / crtbp.LENGTH_UNIT_KM

    def moon_impact(t, state, mu_):
        return np.linalg.norm(state[:3] - moon) - crtbp.MOON_RADIUS_ND

    def earth_impact(t, state, mu_):
        return np.linalg.norm(state[:3] - earth) - earth_radius

    for event in (moon_impact, earth_impact):
        event.terminal = True
        event.direction = -1
    return [moon_impact, earth_impact]


def manifold_branches(orbit, kind="unstable", n_branches=8, duration=5.0, displacement_km=50.0,
                      n_points=400, mu=MU):
    """
    Trajectories on the stable or unstable manifold of a periodic orbit.

    orbit           : orbit dictionary with state0 and period
    kind            : "unstable" (integrated forward) or "stable"
                      (integrated backward)
    n_branches      : number of departure points spread evenly in time
                      around the orbit
    duration        : how long each trajectory is followed, TU
    displacement_km : size of the initial step off the orbit along the
                      manifold direction, applied to the position part;
                      small enough for the linearisation to hold, large
                      enough to leave the orbit in reasonable time
    n_points        : samples per trajectory

    Returns a list of dictionaries, two per departure point (one per
    sign of the displacement), each with
        departure_time : TU along the orbit
        sign           : +1 or -1
        times          : (m,) TU relative to departure, negative for stable
        states         : (m, 6) rotating-frame states
        impact         : "moon", "earth" or None
    """
    if kind not in ("unstable", "stable"):
        raise ValueError("kind must be 'unstable' or 'stable'")

    period = float(orbit["period"])
    state0 = np.asarray(orbit["state0"], dtype=float)
    unstable_direction, stable_direction, _ = hyperbolic_directions(corrector.monodromy_matrix(orbit, mu))
    direction0 = unstable_direction if kind == "unstable" else stable_direction
    time_sign = 1.0 if kind == "unstable" else -1.0

    # State and STM at every departure point in one integration.
    departure_times = np.linspace(0.0, period, n_branches, endpoint=False)
    along = crtbp.propagate_with_stm(state0, period, mu, t_eval=departure_times)
    events = make_impact_events(mu)
    displacement = displacement_km / crtbp.LENGTH_UNIT_KM

    branches = []
    for index, t_depart in enumerate(departure_times):
        state_on_orbit, phi = crtbp.split_state_and_stm(along.y[:, index])
        direction = phi @ direction0
        # Scale so the position part of the step has the requested size.
        direction = direction / np.linalg.norm(direction[:3])
        for sign in (+1.0, -1.0):
            perturbed = state_on_orbit + sign * displacement * direction
            t_eval = np.linspace(0.0, time_sign * duration, n_points)
            sol = crtbp.propagate(perturbed, time_sign * duration, mu, events=events, t_eval=t_eval)
            states = sol.y.T
            times = sol.t
            impact = None
            if sol.status == 1:
                if len(sol.t_events[0]) > 0:
                    impact = "moon"
                    end_state = sol.y_events[0][0]
                    end_time = sol.t_events[0][0]
                else:
                    impact = "earth"
                    end_state = sol.y_events[1][0]
                    end_time = sol.t_events[1][0]
                states = np.vstack([states, end_state])
                times = np.append(times, end_time)
            branches.append({"departure_time": float(t_depart), "sign": sign,
                             "times": times, "states": states, "impact": impact})
    return branches


def closest_approach_to_body(branches, body_position):
    """
    For each branch, the smallest distance (LU) to a body and the time
    at which it occurs.  Returns two arrays of length len(branches).
    Used to rank manifold trajectories as transfer candidates.
    """
    body_position = np.asarray(body_position, dtype=float)
    distances = np.zeros(len(branches))
    times = np.zeros(len(branches))
    for index, branch in enumerate(branches):
        separation = np.linalg.norm(branch["states"][:, :3] - body_position, axis=1)
        nearest = int(np.argmin(separation))
        distances[index] = separation[nearest]
        times[index] = branch["times"][nearest]
    return distances, times
