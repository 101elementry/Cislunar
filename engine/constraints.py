"""
Pluggable access constraints.

Every constraint is a plain function with the signature

    constraint(step: geometry.StepGeometry) -> bool

returning True when the geometry at that time step is acceptable.  The
factories below bake the threshold into a closure so the access code
can call each constraint with the geometry alone and never needs to
know what it tests.  Each closure carries two attributes for reporting:
`kind`, a stable identifier of the constraint type, and `name`, a
description including the threshold.

To add a constraint, write another factory in the same shape and put
its result in the list handed to access.access_mask.
"""


def elevation_cutoff(min_elevation_deg):
    """The target must be at least this high above the local horizon."""
    def constraint(step):
        return step.elevation_deg >= min_elevation_deg
    constraint.kind = "elevation_cutoff"
    constraint.name = f"elevation >= {min_elevation_deg:g} deg"
    return constraint


def station_darkness(max_sun_elevation_deg):
    """
    The Sun must be below this elevation at the station: -6 civil, -12
    nautical, -18 astronomical twilight.
    """
    def constraint(step):
        return step.sun_elevation_deg <= max_sun_elevation_deg
    constraint.kind = "station_darkness"
    constraint.name = f"sun elevation <= {max_sun_elevation_deg:g} deg"
    return constraint


def target_illumination():
    """The target must be in sunlight (outside the Earth and Moon shadows)."""
    def constraint(step):
        return not step.in_shadow
    constraint.kind = "target_illumination"
    constraint.name = "target sunlit"
    return constraint


def limiting_magnitude(magnitude_limit):
    """The target must be at least as bright as the sensor's limit."""
    def constraint(step):
        return step.apparent_magnitude <= magnitude_limit
    constraint.kind = "limiting_magnitude"
    constraint.name = f"magnitude <= {magnitude_limit:g}"
    return constraint


def lunar_exclusion(min_separation_deg):
    """The line of sight must stay this far from the Moon."""
    def constraint(step):
        return step.lunar_separation_deg >= min_separation_deg
    constraint.kind = "lunar_exclusion"
    constraint.name = f"lunar separation >= {min_separation_deg:g} deg"
    return constraint
