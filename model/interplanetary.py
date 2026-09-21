"""
An Earth-Mars transfer laid out on a daily time grid for display: the
paths of the Earth, Mars and the vehicle about the Sun.  Maps a choice
of dates onto engine.interplanetary; knows nothing about display.
"""

import numpy as np

from engine import interplanetary

# Obliquity of the ecliptic at J2000.  The ephemeris is equatorial
# (ICRF); turning about x by this angle lays the planets' orbits flat in
# the x-y plane, which is how a view of the solar system is expected
# to look.
OBLIQUITY_RAD = np.radians(23.439291)


def equatorial_to_ecliptic(positions):
    """Rotate (n, 3) ICRF positions about the x axis into ecliptic axes."""
    cosine = np.cos(OBLIQUITY_RAD)
    sine = np.sin(OBLIQUITY_RAD)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]])
    return positions @ rotation.T


def transfer_scene(ephemeris, departure_jd, flight_days, margin_days=40.0):
    """
    Daily positions, in AU and ecliptic axes, of the Earth, Mars and a
    vehicle flying the Lambert transfer between them.

    The grid runs from margin_days before departure to margin_days
    after arrival.  The vehicle rides with the Earth before departure
    and with Mars after arrival, so all three paths share the grid.

    Returns {"jd": (n,), "paths_au": {"Earth", "Mars", "Vehicle": (n, 3)},
             "transfer": the engine.interplanetary.transfer dictionary,
             "arrival_miss_km": Lambert end point against Mars, a check}
    """
    result = interplanetary.transfer(ephemeris, departure_jd, departure_jd + flight_days)
    jd = np.arange(departure_jd - margin_days, departure_jd + flight_days + margin_days + 0.5, 1.0)
    earth, _ = ephemeris.heliocentric_state("earth", jd)
    mars, _ = ephemeris.heliocentric_state("mars", jd)

    in_flight = (jd >= departure_jd) & (jd <= departure_jd + flight_days)
    arc = interplanetary.heliocentric_path(result["r1"], result["v1"], flight_days, n_points=int(flight_days) + 1)
    vehicle = np.where((jd < departure_jd)[:, np.newaxis], earth, mars)
    vehicle[in_flight] = arc[:int(np.sum(in_flight))]

    paths = {"Earth": earth, "Mars": mars, "Vehicle": vehicle}
    return {"jd": jd,
            "paths_au": {name: equatorial_to_ecliptic(path) / interplanetary.AU_KM for name, path in paths.items()},
            "transfer": result,
            "arrival_miss_km": float(np.linalg.norm(arc[-1] - result["r2"]))}
