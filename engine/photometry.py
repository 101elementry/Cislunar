"""
Reflected-light brightness of a spacecraft modelled as a diffuse sphere.
"""

import numpy as np

# Apparent visual magnitude of the Sun, the reference for reflected light.
SUN_APPARENT_MAGNITUDE = -26.74


def diffuse_sphere_phase_function(phase_angle_deg):
    """
    Fraction of the full-phase brightness a Lambertian sphere shows at a
    given Sun-target-observer phase angle:
        [(pi - alpha) cos(alpha) + sin(alpha)] / pi.
    Equals 1 at alpha = 0 (fully lit) and 0 at alpha = 180 (backlit).
    """
    alpha = np.radians(phase_angle_deg)
    return ((np.pi - alpha) * np.cos(alpha) + np.sin(alpha)) / np.pi


def apparent_magnitude(range_km, diameter_m, albedo, phase_angle_deg):
    """
    Apparent visual magnitude of a diffuse sphere lit by the Sun.

    At zero phase the ratio of reflected flux at the observer to direct
    sunlight is (2/3) * albedo * (radius / range)^2; the phase function
    scales that for other geometries.  The (radius / range)^2 factor is
    the inverse-square falloff.  The Sun is taken to be 1 au from the
    target, which is true to a quarter of a percent in cislunar space.
    """
    radius_m = 0.5 * diameter_m
    range_m = np.asarray(range_km, dtype=float) * 1000.0
    flux_ratio = ((2.0 / 3.0) * albedo * (radius_m / range_m) ** 2
                  * diffuse_sphere_phase_function(phase_angle_deg))
    return SUN_APPARENT_MAGNITUDE - 2.5 * np.log10(flux_ratio)
