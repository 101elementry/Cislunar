"""
Access computation.  Evaluates a list of constraints over a geometry
series without knowing what any constraint tests, then turns the
resulting mask into windows and a duty cycle.
"""

import numpy as np


def evaluate_constraints(series, constraints):
    """
    Evaluate every constraint at every time step.

    series      : geometry.GeometrySeries
    constraints : list of callables step -> bool (see constraints.py)
    Returns a boolean array of shape (n, k): True where step n passes
    constraint k.
    """
    n_steps = len(series)
    results = np.zeros((n_steps, len(constraints)), dtype=bool)
    for index in range(n_steps):
        step = series.at(index)
        for column, constraint in enumerate(constraints):
            results[index, column] = bool(constraint(step))
    return results


def access_mask(series, constraints):
    """True at the steps where every constraint passes, shape (n,)."""
    if len(constraints) == 0:
        return np.ones(len(series), dtype=bool)
    return np.all(evaluate_constraints(series, constraints), axis=1)


def windows_from_mask(times_s, mask):
    """
    Contiguous runs where `mask` is True, as a list of (start, stop) in
    seconds.  Edges are resolved to the grid: a window starts at its
    first True sample and ends at its last, so an isolated True sample
    has zero duration.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0] - 1
    return [(float(times_s[start]), float(times_s[stop])) for start, stop in zip(starts, stops)]


def duty_cycle(windows, span_start_s, span_stop_s):
    """Fraction of [span_start, span_stop] covered by the windows."""
    span = span_stop_s - span_start_s
    if span <= 0.0:
        return 0.0
    covered = 0.0
    for start, stop in windows:
        overlap = min(stop, span_stop_s) - max(start, span_start_s)
        if overlap > 0.0:
            covered = covered + overlap
    return covered / span
