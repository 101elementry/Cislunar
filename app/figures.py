"""
Plotly figure builders for the interface.  They take engine results
(arrays and geometry series) and return figures; no analysis happens
here.
"""

import numpy as np
import plotly.graph_objects as go

from engine import crtbp, frames

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
    geometry          : dictionary from engine.propagation.fixed_points().
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
                   camera=dict(eye=dict(x=0.75, y=-1.0, z=0.45))),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        uirevision=view)
    return figure


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
TEXT_SECONDARY = "#52514e"
GRID = "#ebebe7"

PLOT_LAYOUT = dict(template="plotly_white",
                   font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Inter, Roboto, Arial, sans-serif",
                             size=12, color="#0b0b0b"),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def time_series_figure(series, thresholds, windows, current_time_s=None):
    """
    Elevation, apparent magnitude and lunar separation against time, with
    constraint thresholds as dashed lines, access windows shaded, and an
    optional vertical marker at the current time.

    series     : engine.geometry.GeometrySeries
    thresholds : {"elevation_deg": value or None,
                  "apparent_magnitude": value or None,
                  "lunar_separation_deg": value or None}
    windows    : list of (start_s, stop_s)
    """
    from plotly.subplots import make_subplots

    days = np.asarray(series.time_s) / 86400.0
    stride = max(1, int(np.ceil(len(days) / MAX_PLOT_POINTS)))
    shown_days = days[::stride]

    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                           subplot_titles=("Elevation above horizon [deg]",
                                           "Apparent magnitude (brighter is up)",
                                           "Angular separation from the Moon [deg]"))

    panels = [("elevation_deg", series.elevation_deg, SERIES_BLUE, "min elevation"),
              ("apparent_magnitude", series.apparent_magnitude, SERIES_ORANGE, "limiting magnitude"),
              ("lunar_separation_deg", series.lunar_separation_deg, SERIES_AQUA, "lunar exclusion")]

    for row, (key, values, color, threshold_name) in enumerate(panels, start=1):
        figure.add_trace(go.Scatter(x=shown_days, y=values[::stride], mode="lines", name=key,
                                    line=dict(color=color, width=2), showlegend=False,
                                    hovertemplate="day %{x:.2f}<br>%{y:.2f}<extra></extra>"),
                         row=row, col=1)
        threshold = thresholds.get(key)
        if threshold is not None:
            figure.add_hline(y=threshold, line=dict(color=TEXT_SECONDARY, width=1, dash="dash"),
                             annotation_text=threshold_name, annotation_position="top right",
                             annotation_font=dict(size=10, color=TEXT_SECONDARY), row=row, col=1)

    for start, stop in windows:
        figure.add_vrect(x0=start / 86400.0, x1=stop / 86400.0, fillcolor=SERIES_AQUA, opacity=0.12,
                         line_width=0)

    if current_time_s is not None:
        figure.add_vline(x=current_time_s / 86400.0, line=dict(color="#0b0b0b", width=1.2))

    figure.update_yaxes(autorange="reversed", row=2, col=1)
    figure.update_xaxes(title_text="days from epoch", row=3, col=1)
    figure.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure.update_layout(height=360, margin=dict(l=50, r=20, t=30, b=40), hovermode="x unified",
                         **PLOT_LAYOUT)
    for annotation in figure.layout.annotations[:3]:
        annotation.font.size = 12
        annotation.font.color = TEXT_SECONDARY
        annotation.x = 0.0
        annotation.xanchor = "left"
    return figure


def empty_time_series_figure(message="Run the analysis to see time series"):
    """Placeholder shown before the first run."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font=dict(size=13, color=TEXT_SECONDARY))
    figure.update_layout(height=360, margin=dict(l=50, r=20, t=30, b=40),
                         xaxis=dict(visible=False), yaxis=dict(visible=False), **PLOT_LAYOUT)
    return figure
