"""
Plotly figure builders.  Pure functions of analysis results: they take
arrays and return figures, so the interface callbacks stay thin and the
same figures can be produced from a script.
"""

import numpy as np
import plotly.graph_objects as go

import crtbp
from mission import frames

# Number of points drawn per trajectory.  The analysis grid can have tens
# of thousands of samples; the browser only needs a few thousand.
MAX_PLOT_POINTS = 4000


def subsample(array, max_points=MAX_PLOT_POINTS):
    """Every k-th row so that at most max_points remain."""
    stride = max(1, int(np.ceil(len(array) / max_points)))
    return array[::stride]


def sphere_surface(centre, radius, name, color, resolution=30):
    """A Plotly surface for a sphere, used for the Moon and the Earth."""
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(x=x, y=y, z=z, name=name, showscale=False, hoverinfo="name",
                      colorscale=[[0, color], [1, color]], opacity=1.0)


def view_window(trajectories, geometry, view):
    """
    Axis ranges for the 3D view.

    view = "moon"   : bounding box of the trajectories, the Moon and L1/L2.
    view = "system" : the same box extended to include the Earth.
    Returns (lower, upper) arrays of shape (3,) in LU.
    """
    points = [geometry["moon"], geometry["L1"], geometry["L2"]]
    for states in trajectories.values():
        points.append(states[:, :3].min(axis=0))
        points.append(states[:, :3].max(axis=0))
    if view == "system":
        points.append(geometry["earth"] - frames.EARTH_RADIUS_ND)
        points.append(geometry["earth"] + frames.EARTH_RADIUS_ND)
    points = np.array(points)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    margin = 0.06 * (upper - lower).max()
    return lower - margin, upper + margin


def rotating_frame_figure(trajectories, geometry, station_positions=None, marker_states=None,
                          view="moon"):
    """
    3D view of the rotating frame.

    trajectories      : {name: states (n, 6)} in LU.
    geometry          : dictionary from analysis.fixed_geometry().
    station_positions : optional {name: positions (n, 3)} to draw station
                        tracks on the Earth.
    marker_states     : optional {name: state (6,)} for the current-time
                        markers.
    view              : "moon" or "system", see view_window.
    """
    figure = go.Figure()

    for name, states in trajectories.items():
        shown = subsample(states)
        figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
                                      mode="lines", name=name, line=dict(width=3)))

    figure.add_trace(sphere_surface(geometry["moon"], crtbp.MOON_RADIUS_ND, "Moon", "#8a8a8a"))
    figure.add_trace(sphere_surface(geometry["earth"], frames.EARTH_RADIUS_ND, "Earth", "#3b6fd6"))

    for point in ("L1", "L2"):
        figure.add_trace(go.Scatter3d(x=[geometry[point][0]], y=[geometry[point][1]], z=[geometry[point][2]],
                                      mode="markers+text", name=point, text=[point],
                                      textposition="top center",
                                      marker=dict(size=4, color="red")))

    if station_positions:
        for name, positions in station_positions.items():
            shown = subsample(positions)
            figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
                                          mode="lines", name=name, line=dict(width=2, color="orange")))

    if marker_states:
        for name, state in marker_states.items():
            figure.add_trace(go.Scatter3d(x=[state[0]], y=[state[1]], z=[state[2]],
                                          mode="markers", name=f"{name} (now)",
                                          marker=dict(size=6, color="black", symbol="diamond")))

    # Equal scale on all three axes.  Plotly's default stretches each axis
    # to fill the box, which turns an NRHO into a fat ellipse.  With an
    # explicit window the aspect ratio has to be set by hand from the
    # range of each axis.
    lower, upper = view_window(trajectories, geometry, view)
    span = upper - lower
    ratio = span / span.max()

    km = f"1 LU = {crtbp.LENGTH_UNIT_KM:,.0f} km"
    figure.update_layout(
        scene=dict(xaxis=dict(title=f"x [LU]  ({km})", range=[lower[0], upper[0]]),
                   yaxis=dict(title="y [LU]", range=[lower[1], upper[1]]),
                   zaxis=dict(title="z [LU]", range=[lower[2], upper[2]]),
                   aspectmode="manual",
                   aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
                   camera=dict(eye=dict(x=1.0, y=-1.3, z=0.6))),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        uirevision=view)
    return figure
