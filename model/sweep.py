"""
Parameter sweeps: run a scenario repeatedly with one setting changed,
and collect one summary row per run.  This is the batch runner behind
scripts/ and the interface's sweep panel; both call sweep() and get the
same rows.

A sweep is described by the name of the object to change (or the
scenario itself), the attribute, and the list of values.  Every
observer-spacecraft pair contributes a row per value with its duty
cycle, window count and total and longest window, plus the per-
constraint pass fractions, so the rows can be written straight to CSV
or shown in a table.
"""

import copy
import csv

from model import runner


# Attributes the interface offers for sweeping, per object kind, with a
# label and the unit.  Anything numeric on an object can be swept from
# a script; this list only limits the interface.
SWEEPABLE = {
    "scenario": [("duration_days", "duration [days]"), ("time_step_s", "step [s]")],
    "spacecraft": [("family_index", "family member"), ("diameter_m", "diameter [m]"), ("albedo", "albedo")],
    "ground_station": [("min_elevation_deg", "min elevation [deg]"),
                       ("max_sun_elevation_deg", "max Sun elevation [deg]"),
                       ("latitude_deg", "latitude [deg]"), ("longitude_deg", "longitude [deg]")],
    "optical_sensor": [("limiting_magnitude", "limiting magnitude"), ("lunar_exclusion_deg", "lunar exclusion [deg]"),
                       ("max_range_km", "max range [km]"), ("max_slew_rate_deg_s", "max slew rate [deg/s]")],
}


def summary_rows(scenario, results, label, value):
    """One row per observer-spacecraft pair from a run's results."""
    rows = []
    for (observer, spacecraft), windows in results["windows"].items():
        observation = results["observations"][(observer, spacecraft)]
        total_h = sum(stop - start for start, stop in windows) / 3600.0
        longest_h = max([(stop - start) for start, stop in windows], default=0.0) / 3600.0
        row = {label: value,
               "observer": observer,
               "spacecraft": spacecraft,
               "duty_cycle": results["duty_cycle"][(observer, spacecraft)],
               "n_windows": len(windows),
               "total_hours": total_h,
               "longest_hours": longest_h}
        for kind, column in zip(observation["constraint_kinds"], observation["constraint_masks"].T):
            row[f"pass_fraction_{kind}"] = float(column.mean())
        rows.append(row)
    return rows


def sweep(scenario, target_name, attribute, values, families=None, progress=None):
    """
    Run the scenario once per value with `attribute` of the named object
    set to it.  target_name "scenario" changes the scenario's own
    attribute.  The scenario passed in is not modified.

    progress : optional callable(index, n_values) called before each run
    Returns a list of row dictionaries (see summary_rows); the swept
    attribute is the first column.
    """
    rows = []
    for index, value in enumerate(values):
        if progress is not None:
            progress(index, len(values))
        trial = copy.deepcopy(scenario)
        target = trial if target_name == "scenario" else trial.find(target_name)
        if target is None:
            raise ValueError(f"no object named {target_name!r}")
        current = getattr(target, attribute)
        setattr(target, attribute, int(value) if isinstance(current, int) and not isinstance(current, bool) else float(value))
        results = runner.run_scenario(trial, families)
        rows.extend(summary_rows(trial, results, attribute, value))
    return rows


def write_csv(rows, path):
    """Write sweep rows to a CSV file (columns from the first row)."""
    if len(rows) == 0:
        raise ValueError("no rows to write")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
