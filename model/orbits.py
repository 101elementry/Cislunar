"""
Turning a Spacecraft description into an initial state, and correcting
a typed state into a periodic orbit.

This is the second place (after runner.py) that understands both the
model objects and the engine.  It exists so the interface can offer
"correct this state to a periodic orbit" and "describe this orbit"
without doing any physics in a callback.
"""

import numpy as np

from engine import corrector, crtbp, frames, kepler


def plane_rotation_for(spacecraft, epoch_jd):
    """
    Rotation from the frame the spacecraft's elements are quoted in to
    the rotating frame, or None for the Moon's orbit plane (identity).
    """
    if spacecraft.reference_plane == "earth equator":
        return frames.equatorial_to_rotating_matrix(epoch_jd)
    return None


def initial_state(spacecraft, families, epoch_jd):
    """
    Rotating-frame initial state (6,) of a Spacecraft at scenario time
    zero, whatever its source.

    families : {name: list of orbit dictionaries}, needed for "family"
    epoch_jd : Julian date of the epoch, needed for elements quoted in
               the Earth-equatorial frame
    """
    if spacecraft.source == "family":
        if families is None or spacecraft.family_name not in families:
            raise ValueError(f"spacecraft {spacecraft.name!r} needs the family {spacecraft.family_name!r}")
        return np.array(families[spacecraft.family_name][int(spacecraft.family_index)]["state0"], dtype=float)
    if spacecraft.source == "elements":
        elements = spacecraft.elements
        return kepler.keplerian_to_state_rotating(
            float(elements["a_km"]), float(elements["e"]), float(elements["i_deg"]),
            float(elements["raan_deg"]), float(elements["argp_deg"]), float(elements["true_anomaly_deg"]),
            centre=spacecraft.centre, plane_rotation=plane_rotation_for(spacecraft, epoch_jd))
    return np.array(spacecraft.initial_state, dtype=float)


def period(spacecraft, families):
    """Period in TU of a periodic spacecraft, or None if it has none."""
    if spacecraft.source == "family":
        return float(families[spacecraft.family_name][int(spacecraft.family_index)]["period"])
    if spacecraft.source == "state" and spacecraft.period_tu > 0.0:
        return float(spacecraft.period_tu)
    return None


def correct_to_periodic(spacecraft, families, epoch_jd, fixed=None, period_guess_days=None):
    """
    Correct the spacecraft's current initial state into a periodic orbit
    and store the result on the spacecraft: source becomes "state", the
    state is replaced by the converged one, period_tu is set and the
    propagation switched to "periodic".

    fixed             : component held fixed by the symmetric correctors
                        ("x0", "z0", "vy0"), or None for the default
    period_guess_days : needed only when the state is not a perpendicular
                        xz-plane crossing (general corrector)

    Returns a dictionary describing the converged orbit for display.
    Raises ValueError or RuntimeError with a readable message when the
    correction cannot be done.
    """
    state0 = initial_state(spacecraft, families, epoch_jd)
    period_guess = None
    if period_guess_days is not None and period_guess_days > 0.0:
        period_guess = crtbp.time_to_nondim(period_guess_days * crtbp.SECONDS_PER_DAY)
    elif not corrector.is_perpendicular_crossing(state0):
        existing = period(spacecraft, families)
        if existing is not None:
            period_guess = existing

    orbit, corrector_name = corrector.correct_any(state0, fixed=fixed, period_guess=period_guess)
    corrector.annotate_orbit(orbit)

    spacecraft.source = "state"
    spacecraft.initial_state = [float(value) for value in orbit["state0"]]
    spacecraft.period_tu = float(orbit["period"])
    spacecraft.propagation = "periodic"
    return {"corrector": corrector_name,
            "iterations": int(orbit["iterations"]),
            "residual": float(orbit["residual"]),
            "period_days": crtbp.time_to_days(orbit["period"]),
            "jacobi": float(orbit["jacobi"]),
            "perilune_km": crtbp.length_to_km(orbit["perilune_radius"]),
            "apolune_km": crtbp.length_to_km(orbit["apolune_radius"]),
            "stability_index": float(orbit["stability_index"])}


def describe(spacecraft, families, epoch_jd):
    """
    Plain-text lines summarising the orbit a spacecraft starts on, for
    the interface: period and Jacobi constant for periodic orbits,
    osculating elements for everything.
    """
    try:
        state0 = initial_state(spacecraft, families, epoch_jd)
    except (ValueError, KeyError) as error:
        return [f"cannot build the initial state: {error}"]

    lines = []
    period_tu = period(spacecraft, families)
    if period_tu is not None:
        lines.append(f"periodic orbit, period {crtbp.time_to_days(period_tu):.4f} days ({period_tu:.5f} TU)")
    lines.append(f"Jacobi constant C = {crtbp.jacobi_constant(state0):.6f}")
    lines.append(f"distance to the Moon {crtbp.length_to_km(crtbp.distance_to_moon(state0)):,.0f} km")

    centre = spacecraft.centre if spacecraft.source == "elements" else "moon"
    elements = kepler.state_to_keplerian(state0, centre=centre, plane_rotation=plane_rotation_for(spacecraft, epoch_jd))
    if elements["specific_energy"] < 0.0:
        lines.append(f"osculating about the {centre}: a = {elements['a_km']:,.0f} km, e = {elements['e']:.4f}, "
                     f"i = {elements['i_deg']:.2f} deg, periapsis {elements['periapsis_km']:,.0f} km, "
                     f"apoapsis {elements['apoapsis_km']:,.0f} km, two-body period {elements['period_hours']:.2f} h")
    else:
        lines.append(f"not bound to the {centre} in the two-body sense (energy > 0)")
    return lines
