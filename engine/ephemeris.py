"""
Evaluation of a JPL planetary ephemeris from its Chebyshev coefficients.

A JPL Development Ephemeris stores each body's position as a sequence of
fixed-length time intervals, each holding one Chebyshev polynomial per
coordinate.  Evaluating the ephemeris is therefore: find the interval
the date falls in, map the date onto [-1, 1], and sum the polynomial.
The velocity is the derivative of the same polynomial, taken exactly
with the Chebyshev recurrence, not by finite differences.

The coefficient arrays come from model/ephemeris.py, which reads the
file scripts/fetch_ephemeris.py extracts from DE440s; nothing here
touches a file.

Conventions
  * Times are TDB Julian dates.  UTC differs from TDB by TT - UTC plus
    a periodic term below two milliseconds; utc_to_tdb applies the
    69.184 s offset valid since 2017 (37 leap seconds plus 32.184 s).
  * Positions are ICRF (J2000 equatorial) kilometres, velocities km/s.
  * Segment names: "emb" (solar-system barycentre to Earth-Moon
    barycentre), "sun", "earth" and "moon" (barycentre to body).
"""

import numpy as np

SECONDS_PER_DAY = 86400.0
TT_MINUS_UTC_S = 69.184


def utc_to_tdb(jd_utc):
    """TDB Julian date from a UTC Julian date (TT - UTC = 69.184 s since 2017; TDB ~ TT)."""
    return np.asarray(jd_utc, dtype=float) + TT_MINUS_UTC_S / SECONDS_PER_DAY


class ChebyshevSegment:
    """
    One body's coefficients: `coefficients` of shape (3, n_intervals,
    n_terms), the TDB Julian date `init` where the first interval
    starts, and the interval length in days.
    """

    def __init__(self, init, interval_days, coefficients):
        self.init = float(init)
        self.interval_days = float(interval_days)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.n_intervals = self.coefficients.shape[1]
        self.n_terms = self.coefficients.shape[2]

    def covers(self, jd_tdb):
        jd_tdb = np.asarray(jd_tdb, dtype=float)
        return bool(np.all(jd_tdb >= self.init) and np.all(jd_tdb < self.init + self.n_intervals * self.interval_days))

    def position_velocity(self, jd_tdb):
        """
        Position (n, 3) km and velocity (n, 3) km/s at TDB Julian dates.

        With s the date mapped onto [-1, 1] within its interval, the
        position is sum_k c_k T_k(s) and the velocity is
        sum_k c_k T_k'(s) * ds/dt, where ds/dt = 2 / (interval in
        seconds).  T_k and T_k' come from the recurrences
            T_0 = 1, T_1 = s, T_k = 2 s T_{k-1} - T_{k-2}
            T_0' = 0, T_1' = 1, T_k' = 2 T_{k-1} + 2 s T_{k-1}' - T_{k-2}'
        """
        jd_tdb = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
        if not self.covers(jd_tdb):
            raise ValueError("date outside the extracted ephemeris span")
        index = np.floor((jd_tdb - self.init) / self.interval_days).astype(int)
        index = np.clip(index, 0, self.n_intervals - 1)
        interval_start = self.init + index * self.interval_days
        s = 2.0 * (jd_tdb - interval_start) / self.interval_days - 1.0

        n = len(jd_tdb)
        t_values = np.zeros((self.n_terms, n))
        t_derivatives = np.zeros((self.n_terms, n))
        t_values[0] = 1.0
        if self.n_terms > 1:
            t_values[1] = s
            t_derivatives[1] = 1.0
        for k in range(2, self.n_terms):
            t_values[k] = 2.0 * s * t_values[k - 1] - t_values[k - 2]
            t_derivatives[k] = 2.0 * t_values[k - 1] + 2.0 * s * t_derivatives[k - 1] - t_derivatives[k - 2]

        # coefficients[axis, interval, term] picked per date.
        picked = self.coefficients[:, index, :]                      # (3, n, n_terms)
        position = np.einsum("ank,kn->na", picked, t_values)
        ds_dt = 2.0 / (self.interval_days * SECONDS_PER_DAY)
        velocity = np.einsum("ank,kn->na", picked, t_derivatives) * ds_dt
        return position, velocity


class Ephemeris:
    """
    The four segments needed for cislunar work and the derived vectors.
    `segments` is {name: ChebyshevSegment}.
    """

    def __init__(self, segments):
        self.segments = segments
        for name in ("emb", "sun", "earth", "moon"):
            if name not in segments:
                raise ValueError(f"ephemeris is missing the {name!r} segment")

    def covers(self, jd_tdb):
        return all(segment.covers(jd_tdb) for segment in self.segments.values())

    def earth_to_moon(self, jd_tdb):
        """Moon position (n, 3) km and velocity (n, 3) km/s relative to the Earth's centre."""
        moon_position, moon_velocity = self.segments["moon"].position_velocity(jd_tdb)
        earth_position, earth_velocity = self.segments["earth"].position_velocity(jd_tdb)
        return moon_position - earth_position, moon_velocity - earth_velocity

    def barycentre_to_sun(self, jd_tdb):
        """Sun position (n, 3) km relative to the Earth-Moon barycentre."""
        sun_position, _ = self.segments["sun"].position_velocity(jd_tdb)
        emb_position, _ = self.segments["emb"].position_velocity(jd_tdb)
        return sun_position - emb_position

    def earth_to_barycentre(self, jd_tdb):
        """Earth-Moon barycentre position (n, 3) km relative to the Earth's centre."""
        earth_position, _ = self.segments["earth"].position_velocity(jd_tdb)
        return -earth_position

    def rotating_frame_axes(self, jd_tdb):
        """
        Unit vectors of the CRTBP rotating frame in ICRF at each date:
        x along the Earth-to-Moon line, z along the Earth-Moon orbital
        angular momentum, y completing the right-handed set.  Returns
        an (n, 3, 3) array whose rows are x, y, z, so that multiplying
        an ICRF vector by it gives rotating-frame components.
        """
        position, velocity = self.earth_to_moon(jd_tdb)
        x_axis = position / np.linalg.norm(position, axis=1)[:, np.newaxis]
        angular_momentum = np.cross(position, velocity)
        z_axis = angular_momentum / np.linalg.norm(angular_momentum, axis=1)[:, np.newaxis]
        y_axis = np.cross(z_axis, x_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=1)

    def earth_moon_distance_km(self, jd_tdb):
        """Real Earth-Moon distance (n,) km, for comparison with the fixed length unit."""
        position, _ = self.earth_to_moon(jd_tdb)
        return np.linalg.norm(position, axis=1)
