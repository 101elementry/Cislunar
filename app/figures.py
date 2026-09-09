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


# Palette shared with assets/style.css.
GROUND = "#0b0e14"
PANEL = "#121722"
GRID = "#232b3a"
TEXT = "#e7eaf0"
TEXT_SECONDARY = "#9aa4b5"
TEXT_MUTED = "#66718a"
EARTHSHINE = "#6fb1ff"
SUN = "#f5c451"
REGOLITH = "#d9d3c4"
UNSTABLE = "#ff6b57"
STABLE = "#43d19a"

# One colour per spacecraft, earthshine first, then hues that stay apart
# from the manifold red and green and the station gold.
SPACECRAFT_COLORS = ["#6fb1ff", "#c792ea", "#f78fb3", "#7fdbff", "#ffd166", "#b8e986", "#ff9f6b", "#9ab0ff"]

BODY_SHADING = {"moon": [[0.0, "#3a3a38"], [0.55, "#8f8d86"], [1.0, "#e8e4d8"]],
                "earth": [[0.0, "#0c2a5e"], [0.6, "#2d6fd0"], [1.0, "#9fd0ff"]]}


def sphere_surface(centre, radius, name, body, resolution=40):
    """
    A Plotly surface for a sphere, shaded as if lit from the +x side of
    the rotating frame, roughly where the Sun sits at new Moon; the
    shading is cosmetic and only there to make the body read as a
    sphere.  `body` picks the Moon or Earth colours.
    """
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    unit_x = np.outer(np.cos(u), np.sin(v))
    unit_y = np.outer(np.sin(u), np.sin(v))
    unit_z = np.outer(np.ones_like(u), np.cos(v))
    light = np.array([0.55, -0.6, 0.58])
    light = light / np.linalg.norm(light)
    shade = 0.5 + 0.5 * (unit_x * light[0] + unit_y * light[1] + unit_z * light[2])
    return go.Surface(x=centre[0] + radius * unit_x, y=centre[1] + radius * unit_y, z=centre[2] + radius * unit_z,
                      surfacecolor=shade, cmin=0.0, cmax=1.0, name=name, showscale=False, hoverinfo="name",
                      colorscale=BODY_SHADING[body],
                      lighting=dict(ambient=0.75, diffuse=0.5, specular=0.08, roughness=0.9, fresnel=0.1),
                      opacity=1.0)


SCENE_AXIS = dict(backgroundcolor=GROUND, gridcolor=GRID, zerolinecolor=GRID, showbackground=True,
                  color=TEXT_MUTED, tickfont=dict(size=10, family="IBM Plex Mono, Menlo, monospace"),
                  title_font=dict(size=11, color=TEXT_SECONDARY))


MANIFOLD_COLORS = {"unstable": "rgba(255, 107, 87, 0.5)", "stable": "rgba(67, 209, 154, 0.5)"}


def view_window(points, margin_fraction=0.08):
    """
    Axis ranges (lower, upper) in LU enclosing a list of (3,) or (n, 3)
    point arrays with a margin, used to give the 3D view equal scale.
    """
    stacked = np.vstack([np.atleast_2d(p) for p in points])
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    margin = margin_fraction * (upper - lower).max()
    return lower - margin, upper + margin


def camera_for(focus_point, lower, upper, ratio, zoom=1.3):
    """
    A Plotly 3D camera centred on a data point.  Plotly places the data
    box so that each axis spans [-aspectratio / 2, +aspectratio / 2]
    about the origin of the camera's coordinates (measured from the
    scene's model matrix, see assets/zoom_to_cursor.js), so a point at
    fraction f along an axis sits at (f - 1/2) times that axis's aspect
    ratio.  The eye is put on the default viewing direction at a
    distance of `zoom` times the largest aspect ratio, so scroll-zooming
    afterwards goes into the focused point.
    """
    fraction = (np.asarray(focus_point, dtype=float) - lower) / (upper - lower)
    center = (fraction - 0.5) * ratio
    direction = np.array([0.75, -1.0, 0.45])
    direction = direction / np.linalg.norm(direction)
    eye = center + zoom * ratio.max() * direction
    return dict(center=dict(x=center[0], y=center[1], z=center[2]),
                eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                up=dict(x=0, y=0, z=1))


def trajectory_figure(trajectories, bodies, body_radii, index=0, points=None, station_positions=None,
                      marker_states=None, manifolds=None, view="moon", frame_label="rotating frame",
                      focus_point=None, focus_key="none", zoom=1.3, trail_samples=0):
    """
    3D view of trajectories with the Earth and Moon drawn to scale.

    trajectories      : {name: states (n, 6)} in the displayed frame, LU
    bodies            : {"earth": positions, "moon": positions}, each (3,)
                        for a fixed body or (n, 3) for a moving one
    body_radii        : {"earth": r, "moon": r} in LU
    index             : time index at which moving bodies and markers
                        are drawn
    points            : optional {name: (3,)} fixed points to label (L1, L2)
    station_positions : optional {name: (n, 3)} ground-station tracks
    marker_states     : optional {name: state (6,)} current-time markers
    manifolds         : optional {spacecraft: {"unstable": [branch, ...],
                        "stable": [...]}} from engine.manifolds, in the
                        displayed frame
    view              : "moon" frames the trajectories and the Moon,
                        "system" also includes the Earth
    frame_label       : text for the axis titles
    focus_point       : optional (3,) point the camera is centred on, so
                        scroll zoom goes into it; focus_key names it for
                        Plotly's uirevision so a change of focus resets
                        the camera while slider moves keep it
    zoom              : eye distance as a fraction of the box when focusing
    trail_samples     : if > 0, draw a brighter fading tail of this many
                        samples behind each current-time marker
    """
    figure = go.Figure()

    def body_now(name):
        positions = np.asarray(bodies[name])
        if positions.ndim == 1:
            return positions
        return positions[min(index, len(positions) - 1)]

    if manifolds:
        legend_shown = set()
        for spacecraft, kinds in manifolds.items():
            for kind, branches in kinds.items():
                for branch in branches:
                    shown = subsample(branch["states"], 600)
                    label = f"{spacecraft} {kind} manifold"
                    figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode="lines",
                                                  name=label, legendgroup=label, showlegend=label not in legend_shown,
                                                  line=dict(width=1.5, color=MANIFOLD_COLORS[kind]),
                                                  hoverinfo="name"))
                    legend_shown.add(label)

    colors = {}
    for k, (name, states) in enumerate(trajectories.items()):
        colors[name] = SPACECRAFT_COLORS[k % len(SPACECRAFT_COLORS)]
        shown = subsample(states)
        full_width = 2.0 if trail_samples > 0 else 3.5
        figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
                                      mode="lines", name=name, line=dict(width=full_width, color=colors[name]),
                                      opacity=0.55 if trail_samples > 0 else 1.0,
                                      hovertemplate=f"{name}<br>x %{{x:.4f}}<br>y %{{y:.4f}}<br>z %{{z:.4f}} LU<extra></extra>"))
        if trail_samples > 0 and index > 1:
            # A comet tail: the last trail_samples up to now, fading from
            # transparent to the spacecraft colour.
            start = max(0, index - trail_samples)
            tail = states[start:index + 1, :3]
            fade = np.linspace(0.0, 1.0, len(tail))
            figure.add_trace(go.Scatter3d(x=tail[:, 0], y=tail[:, 1], z=tail[:, 2], mode="lines",
                                          showlegend=False, hoverinfo="skip",
                                          line=dict(width=6, color=fade,
                                                    colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, colors[name]]],
                                                    cmin=0.0, cmax=1.0)))

    body_paths = {"moon": REGOLITH, "earth": EARTHSHINE}
    for name in ("moon", "earth"):
        positions = np.asarray(bodies[name])
        if positions.ndim == 2:
            shown = subsample(positions)
            figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode="lines",
                                          name=f"{name.capitalize()} path", showlegend=False,
                                          line=dict(width=1, color=body_paths[name], dash="dot"), hoverinfo="skip"))
        figure.add_trace(sphere_surface(body_now(name), body_radii[name], name.capitalize(), name))

    if points:
        for name, position in points.items():
            figure.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
                                          mode="markers+text", name=name, text=[name], textposition="top center",
                                          textfont=dict(color=TEXT_SECONDARY, size=11),
                                          marker=dict(size=3.5, color=TEXT_SECONDARY, symbol="cross")))

    if station_positions:
        for name, positions in station_positions.items():
            shown = subsample(positions)
            figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
                                          mode="lines", name=name, line=dict(width=2, color=SUN)))

    if marker_states:
        for name, state in marker_states.items():
            color = colors.get(name, TEXT)
            # A soft halo behind a bright core reads as "now" without a
            # symbol that clashes with the trajectory colour.
            figure.add_trace(go.Scatter3d(x=[state[0]], y=[state[1]], z=[state[2]], mode="markers",
                                          showlegend=False, hoverinfo="skip",
                                          marker=dict(size=14, color=color, opacity=0.25)))
            figure.add_trace(go.Scatter3d(x=[state[0]], y=[state[1]], z=[state[2]], mode="markers",
                                          name=f"{name} (now)",
                                          marker=dict(size=6, color="#ffffff", line=dict(color=color, width=2))))

    # Equal scale on all three axes.  Plotly's default stretches each axis
    # to fill the box, which turns an NRHO into a fat ellipse.  With an
    # explicit window the aspect ratio has to be set by hand from the
    # range of each axis.
    window_points = [states[:, :3] for states in trajectories.values()]
    window_points.append(body_now("moon") - body_radii["moon"])
    window_points.append(body_now("moon") + body_radii["moon"])
    if points:
        window_points.extend(points.values())
    if view == "system":
        window_points.append(body_now("earth") - body_radii["earth"])
        window_points.append(body_now("earth") + body_radii["earth"])
        window_points.extend(np.atleast_2d(np.asarray(bodies[name])) for name in ("earth", "moon"))
    lower, upper = view_window(window_points)
    span = upper - lower
    ratio = span / span.max()
    if focus_point is not None:
        camera = camera_for(focus_point, lower, upper, ratio, zoom)
    else:
        camera = dict(eye=dict(x=0.75, y=-1.0, z=0.45), up=dict(x=0, y=0, z=1))

    km = f"1 LU = {crtbp.LENGTH_UNIT_KM:,.0f} km"
    figure.update_layout(
        scene=dict(xaxis=dict(title=f"x [LU], {frame_label}  ({km})", range=[lower[0], upper[0]], **SCENE_AXIS),
                   yaxis=dict(title="y [LU]", range=[lower[1], upper[1]], **SCENE_AXIS),
                   zaxis=dict(title="z [LU]", range=[lower[2], upper[2]], **SCENE_AXIS),
                   aspectmode="manual",
                   aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
                   dragmode="turntable",
                   camera=camera),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="IBM Plex Sans, -apple-system, Segoe UI, sans-serif", color=TEXT_SECONDARY, size=12),
        margin=dict(l=0, r=0, t=34, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=TEXT_SECONDARY)),
        hoverlabel=dict(bgcolor="#171d2a", bordercolor=GRID, font=dict(family="IBM Plex Mono, Menlo, monospace",
                                                                         color=TEXT, size=11)),
        uirevision=f"{view}-{frame_label}-{focus_key}")
    return figure


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------

SERIES_BLUE = EARTHSHINE
SERIES_ORANGE = SUN
SERIES_AQUA = STABLE

PLOT_LAYOUT = dict(template="plotly_dark",
                   font=dict(family="IBM Plex Sans, -apple-system, Segoe UI, sans-serif", size=12, color=TEXT_SECONDARY),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   hoverlabel=dict(bgcolor="#171d2a", bordercolor=GRID,
                                   font=dict(family="IBM Plex Mono, Menlo, monospace", color=TEXT, size=11)))


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

    # Three small multiples side by side: the strip stays short so the
    # 3D scene keeps the height.
    figure = make_subplots(rows=1, cols=3, horizontal_spacing=0.05,
                           subplot_titles=("Elevation above horizon [deg]",
                                           "Apparent magnitude (brighter is up)",
                                           "Angular separation from the Moon [deg]"))

    panels = [("elevation_deg", series.elevation_deg, SERIES_BLUE, "min elevation"),
              ("apparent_magnitude", series.apparent_magnitude, SERIES_ORANGE, "limiting magnitude"),
              ("lunar_separation_deg", series.lunar_separation_deg, SERIES_AQUA, "lunar exclusion")]

    for col, (key, values, color, threshold_name) in enumerate(panels, start=1):
        figure.add_trace(go.Scatter(x=shown_days, y=values[::stride], mode="lines", name=key,
                                    line=dict(color=color, width=1.8), showlegend=False,
                                    hovertemplate="day %{x:.2f}<br>%{y:.2f}<extra></extra>"),
                         row=1, col=col)
        threshold = thresholds.get(key)
        if threshold is not None:
            figure.add_hline(y=threshold, line=dict(color=TEXT_MUTED, width=1, dash="dash"),
                             annotation_text=threshold_name, annotation_position="top right",
                             annotation_font=dict(size=10, color=TEXT_MUTED), row=1, col=col)
        for start, stop in windows:
            figure.add_vrect(x0=start / 86400.0, x1=stop / 86400.0, fillcolor=SERIES_AQUA, opacity=0.14,
                             line_width=0, row=1, col=col)
        if current_time_s is not None:
            figure.add_vline(x=current_time_s / 86400.0, line=dict(color=REGOLITH, width=1.2), row=1, col=col)
        figure.update_xaxes(title_text="days", title_font=dict(size=10), title_standoff=4, row=1, col=col)

    figure.update_yaxes(autorange="reversed", row=1, col=2)
    figure.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    figure.update_layout(height=210, margin=dict(l=40, r=16, t=28, b=34), hovermode="x",
                         **PLOT_LAYOUT)
    for annotation in figure.layout.annotations[:3]:
        annotation.font.size = 11
        annotation.font.color = TEXT_SECONDARY
    return figure


def empty_time_series_figure(message="Run the analysis to see time series"):
    """Placeholder shown before the first run."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font=dict(size=13, color=TEXT_SECONDARY))
    figure.update_layout(height=210, margin=dict(l=40, r=16, t=28, b=34),
                         xaxis=dict(visible=False), yaxis=dict(visible=False), **PLOT_LAYOUT)
    return figure
