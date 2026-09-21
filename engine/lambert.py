"""
Lambert's problem: the two-body orbit that joins two positions in a
given time.

Given where a spacecraft is now (r1), where it must be later (r2) and
the time allowed (the time of flight), there is one conic about the
central body that does it without going round more than once in a given
direction.  Its velocity at each end is what a mission designer needs:
subtract the planet's own velocity and what is left is the hyperbolic
excess velocity the launch vehicle must supply, or the capture burn
must remove.

The method is the universal-variable formulation (Bate, Mueller and
White, section 5.3; Curtis, algorithm 5.2).  One variable z stands for
the square of the change in eccentric anomaly, positive for an ellipse,
zero for a parabola, negative for a hyperbola, so a single equation
covers all three.  With the Stumpff functions C(z) and S(z),

    y(z)   = |r1| + |r2| + A (z S(z) - 1) / sqrt(C(z))
    t(z)   = [ (y / C)^(3/2) S + A sqrt(y) ] / sqrt(mu)

where A = sin(dtheta) sqrt(|r1| |r2| / (1 - cos(dtheta))) depends only
on the geometry.  t(z) rises monotonically with z on the single
revolution branch, so z is found by bisection, which cannot diverge.
The end velocities then follow from the Lagrange coefficients

    f = 1 - y / |r1|,   g = A sqrt(y / mu),   g_dot = 1 - y / |r2|
    v1 = (r2 - f r1) / g,   v2 = (g_dot r2 - r1) / g

Units: whatever consistent set mu is given in; this project uses km,
km/s and seconds for interplanetary work.
"""

import numpy as np


def stumpff_c(z):
    """Stumpff function C(z) = (1 - cos sqrt(z)) / z, with its hyperbolic and series forms."""
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    if z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    return 1.0 / 2.0 - z / 24.0 + z ** 2 / 720.0


def stumpff_s(z):
    """Stumpff function S(z) = (sqrt(z) - sin sqrt(z)) / z^(3/2), with its hyperbolic and series forms."""
    if z > 1e-6:
        root = np.sqrt(z)
        return (root - np.sin(root)) / root ** 3
    if z < -1e-6:
        root = np.sqrt(-z)
        return (np.sinh(root) - root) / root ** 3
    return 1.0 / 6.0 - z / 120.0 + z ** 2 / 5040.0


def solve(r1, r2, time_of_flight, mu, prograde=True, tolerance=1e-10, max_iterations=200):
    """
    Velocities (v1, v2) at the two ends of the single-revolution transfer
    from position r1 to position r2 in time_of_flight.

    r1, r2         : (3,) positions relative to the central body
    time_of_flight : positive time between them
    mu             : gravitational parameter of the central body
    prograde       : True for a transfer that goes round in the same
                     sense as the planets (angular momentum along +z),
                     False for the other way round

    Raises ValueError if the geometry is degenerate (the two positions
    and the central body in one line, where the orbit plane is
    undefined) or no solution is bracketed.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)

    # Transfer angle between the two positions, taken the prograde or
    # the retrograde way round according to the sign of (r1 x r2)_z.
    cos_dtheta = np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0)
    dtheta = np.arccos(cos_dtheta)
    normal_z = np.cross(r1, r2)[2]
    if (prograde and normal_z < 0.0) or (not prograde and normal_z >= 0.0):
        dtheta = 2.0 * np.pi - dtheta

    if abs(1.0 - np.cos(dtheta)) < 1e-12:
        raise ValueError("Lambert: the two positions are in the same direction; the transfer is undefined")
    a_constant = np.sin(dtheta) * np.sqrt(r1_norm * r2_norm / (1.0 - np.cos(dtheta)))
    if abs(a_constant) < 1e-12:
        raise ValueError("Lambert: the positions are opposite; the orbit plane is undefined")

    def y_of(z):
        return r1_norm + r2_norm + a_constant * (z * stumpff_s(z) - 1.0) / np.sqrt(stumpff_c(z))

    def time_of(z):
        y = y_of(z)
        return ((y / stumpff_c(z)) ** 1.5 * stumpff_s(z) + a_constant * np.sqrt(y)) / np.sqrt(mu)

    # Bracket the root.  Above (2 pi)^2 the transfer would need more than
    # one revolution.  The lower end is pushed down until the time there
    # is shorter than the one asked for; y must stay positive.
    z_upper = (2.0 * np.pi) ** 2 - 1e-6
    z_lower = -4.0 * np.pi ** 2
    while y_of(z_lower) <= 0.0:
        z_lower = 0.5 * z_lower + 0.05
        if z_lower > z_upper:
            raise ValueError("Lambert: no valid bracket")
    expansions = 0
    while time_of(z_lower) > time_of_flight and y_of(2.0 * z_lower) > 0.0 and expansions < 40:
        z_lower = 2.0 * z_lower
        expansions += 1
    if time_of(z_lower) > time_of_flight:
        raise ValueError("Lambert: time of flight too short for this geometry")

    for _ in range(max_iterations):
        z = 0.5 * (z_lower + z_upper)
        if y_of(z) <= 0.0:
            z_lower = z
            continue
        if time_of(z) < time_of_flight:
            z_lower = z
        else:
            z_upper = z
        if (z_upper - z_lower) < tolerance:
            break

    y = y_of(z)
    f = 1.0 - y / r1_norm
    g = a_constant * np.sqrt(y / mu)
    g_dot = 1.0 - y / r2_norm
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v1, v2
