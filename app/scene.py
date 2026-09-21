"""
The 3D scene of one run in one display frame.  Shared by the Dash
interface (app/main.py) and the static showcase (app/showcase.py), so
both draw exactly the same figure.  The frame conversions are
engine.frames calls; nothing here computes physics.
"""

import numpy as np

from engine import crtbp, frames, propagation
from app import figures

FIXED_POINTS = propagation.fixed_points()

FRAME_LABELS = {"rotating": "rotating frame", "moon_rotating": "rotating frame, Moon-centred",
                "moon_inertial": "inertial, Moon-centred", "earth_inertial": "inertial, Earth-centred",
                "inertial": "inertial, barycentric"}
BODY_RADII = {"moon": crtbp.MOON_RADIUS_ND, "earth": frames.EARTH_RADIUS_ND}


def displayed_frame(results, frame):
    """
    Trajectories, manifolds, stations and bodies converted from the
    rotating frame to the displayed frame (engine.frames does the
    conversion).  Returns (trajectories, manifolds, stations, bodies,
    points) in the shapes figures.trajectory_figure expects.
    """
    times = results["times_nondim"]
    trajectories = results["trajectories"]
    manifold_branches = results["manifolds"]
    stations = results["stations"]

    if frame in ("rotating", "moon_rotating"):
        offset = crtbp.moon_position() if frame == "moon_rotating" else np.zeros(3)
        shift = np.concatenate([offset, np.zeros(3)])
        trajectories = {name: states - shift for name, states in trajectories.items()}
        manifold_branches = {name: {kind: [dict(branch, states=branch["states"] - shift) for branch in branches]
                                    for kind, branches in kinds.items()}
                             for name, kinds in manifold_branches.items()}
        stations = {name: positions - offset for name, positions in stations.items()}
        bodies = {"earth": FIXED_POINTS["earth"] - offset, "moon": FIXED_POINTS["moon"] - offset}
        points = {"L1": FIXED_POINTS["L1"] - offset, "L2": FIXED_POINTS["L2"] - offset}
        return trajectories, manifold_branches, stations, bodies, points

    centre = {"moon_inertial": "moon", "earth_inertial": "earth", "inertial": "barycentre"}[frame]
    trajectories = {name: frames.rotating_to_inertial_states(states, times, centre)
                    for name, states in trajectories.items()}
    converted_manifolds = {}
    for name, kinds in manifold_branches.items():
        converted_manifolds[name] = {}
        for kind, branches in kinds.items():
            converted = []
            for branch in branches:
                # A branch leaves the orbit at departure_time and runs for
                # branch["times"] after it (negative for stable branches).
                branch_times = branch["departure_time"] + branch["times"]
                converted.append(dict(branch, states=frames.rotating_to_inertial_states(branch["states"], branch_times, centre)))
            converted_manifolds[name][kind] = converted
    station_states = {name: np.hstack([positions, np.zeros_like(positions)]) for name, positions in stations.items()}
    stations = {name: frames.rotating_to_inertial_states(states, times, centre)[:, :3]
                for name, states in station_states.items()}
    bodies = frames.body_positions_inertial(times, centre)
    return trajectories, converted_manifolds, stations, bodies, None


def focus_point_for(focus, trajectories, bodies, points, index):
    """The displayed-frame point a focus choice refers to, or None."""
    kind, _, name = (focus or "none").partition(":")
    if kind == "body" and name in bodies:
        positions = np.asarray(bodies[name])
        return positions if positions.ndim == 1 else positions[index]
    if kind == "point" and points and name in points:
        return points[name]
    if kind == "spacecraft" and name in trajectories:
        return trajectories[name][index, :3]
    return None


def scene_figure(scenario, results, frame, view, focus, index):
    """The 3D figure of one frame at one time index."""
    trajectories, manifold_branches, stations, bodies, points = displayed_frame(results, frame)
    markers = {name: states[index] for name, states in trajectories.items()}
    trail = max(2, len(results["times_s"]) // 12)
    clock = {"epoch_utc": scenario.epoch_utc, "time_step_s": float(scenario.time_step_s),
             "n_samples": int(len(results["times_s"]))}
    return figures.trajectory_figure(trajectories, bodies, BODY_RADII, index=index, points=points,
                                     station_positions=stations, marker_states=markers,
                                     manifolds=manifold_branches, view=view,
                                     frame_label=FRAME_LABELS[frame],
                                     focus_point=focus_point_for(focus, trajectories, bodies, points, index),
                                     focus_key=focus, trail_samples=trail, clock=clock)
