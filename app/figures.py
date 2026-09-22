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
GROUND = "#000000"
PANEL = "#070707"
GRID = "#1c1c1c"
TEXT = "#f2f2f2"
TEXT_SECONDARY = "#a3a3a3"
TEXT_MUTED = "#6e6e6e"
EARTHSHINE = "#7fb2e5"
SUN = "#e8b84a"
REGOLITH = "#d6d2c8"
UNSTABLE = "#e5604d"
STABLE = "#3ecf8e"

# One colour per spacecraft, earthshine first, then hues that stay apart
# from the manifold red and green and the station gold.
SPACECRAFT_COLORS = ["#7fb2e5", "#b79ad6", "#de8fa8", "#7cc7c9", "#e3c16f", "#a9cf85", "#e39a6b", "#93a3d9"]

BODY_SHADING = {"moon": [[0.0, "#3a3a38"], [0.55, "#8f8d86"], [1.0, "#e8e4d8"]],
                "earth": [[0.0, "#0c2a5e"], [0.6, "#2d6fd0"], [1.0, "#9fd0ff"]]}


def sphere_surface(centre, radius, name, body, resolution=28):
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


# No axis walls: the scene is a black void with a faint grid, like the
# orbit viewers operators use, so the trajectories carry the picture.
SCENE_AXIS = dict(backgroundcolor=GROUND, gridcolor=GRID, zerolinecolor="#2c2c2c", showbackground=False,
                  color=TEXT_MUTED, tickfont=dict(size=9, family="IBM Plex Mono, Menlo, monospace"),
                  title_font=dict(size=10, color=TEXT_MUTED))


MANIFOLD_COLORS = {"unstable": "rgba(229, 96, 77, 0.5)", "stable": "rgba(62, 207, 142, 0.5)"}


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
                      focus_point=None, focus_key="none", zoom=1.3, trail_samples=0, clock=None):
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
    clock             : optional {"epoch_utc", "time_step_s", "n_samples"}
                        stored in the layout so the browser-side playback
                        (assets/playback.js) can run the clock itself

    Every trace carries a `meta` role (path, trail, halo, marker, body,
    bodypath, manifold) so the playback script can find and move the
    right ones.
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
                    # A branch is a trajectory, not a static curve, so it is
                    # drawn only as far as a spacecraft on it has flown by
                    # the current time.  An unstable branch leaves the orbit
                    # at departure_time and grows away from it; a stable one
                    # has been falling toward the orbit for the whole flight
                    # and lands on it at departure_time.  The branch is
                    # integrated on a uniform time grid, so a fraction of the
                    # flight time is the same fraction of the drawn points;
                    # only the last point of a branch that ends on a surface
                    # breaks that, and by less than one step.
                    figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode="lines",
                                                  name=label, legendgroup=label, showlegend=label not in legend_shown,
                                                  line=dict(width=1.5, color=MANIFOLD_COLORS[kind]),
                                                  hoverinfo="name",
                                                  meta={"role": "manifold", "kind": kind,
                                                        "departure_s": float(crtbp.time_to_seconds(branch["departure_time"])),
                                                        "flight_s": float(abs(crtbp.time_to_seconds(branch["times"][-1])))}))
                    legend_shown.add(label)

    colors = {}
    for k, (name, states) in enumerate(trajectories.items()):
        colors[name] = SPACECRAFT_COLORS[k % len(SPACECRAFT_COLORS)]
        shown = subsample(states)
        full_width = 2.0 if trail_samples > 0 else 3.5
        stride = max(1, int(np.ceil(len(states) / MAX_PLOT_POINTS)))
        figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
                                      mode="lines", name=name, line=dict(width=full_width, color=colors[name]),
                                      opacity=0.55 if trail_samples > 0 else 1.0,
                                      meta={"role": "path", "spacecraft": name, "stride": stride},
                                      hovertemplate=f"{name}<br>x %{{x:.4f}}<br>y %{{y:.4f}}<br>z %{{z:.4f}} LU<extra></extra>"))
        if trail_samples > 0:
            # A comet tail: the last trail_samples up to now, fading from
            # transparent to the spacecraft colour.  Always the same number
            # of points (the start is clamped) so the playback script can
            # replace the coordinates without touching the fade.
            picks = np.clip(np.arange(index - trail_samples, index + 1), 0, len(states) - 1)
            tail = states[picks, :3]
            fade = np.linspace(0.0, 1.0, len(tail))
            figure.add_trace(go.Scatter3d(x=tail[:, 0], y=tail[:, 1], z=tail[:, 2], mode="lines",
                                          showlegend=False, hoverinfo="skip",
                                          meta={"role": "trail", "spacecraft": name, "samples": int(trail_samples)},
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
                                          meta={"role": "bodypath", "body": name,
                                                "stride": max(1, int(np.ceil(len(positions) / MAX_PLOT_POINTS)))},
                                          line=dict(width=1, color=body_paths[name], dash="dot"), hoverinfo="skip"))
        sphere = sphere_surface(body_now(name), body_radii[name], name.capitalize(), name)
        sphere.meta = {"role": "body", "body": name}
        figure.add_trace(sphere)

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
                                          meta={"role": "halo", "spacecraft": name},
                                          marker=dict(size=14, color=color, opacity=0.25)))
            figure.add_trace(go.Scatter3d(x=[state[0]], y=[state[1]], z=[state[2]], mode="markers",
                                          name=f"{name} (now)",
                                          meta={"role": "marker", "spacecraft": name},
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
    if manifolds:
        # Manifold branches travel far from their orbit; without them in
        # the window the tubes spill out of the axes box.
        for kinds in manifolds.values():
            for branches in kinds.values():
                window_points.extend(branch["states"][:, :3] for branch in branches)
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
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=TEXT_SECONDARY, size=12),
        margin=dict(l=0, r=0, t=34, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=TEXT_SECONDARY)),
        hoverlabel=dict(bgcolor="#111111", bordercolor="#303030", font=dict(family="IBM Plex Mono, Menlo, monospace",
                                                                            color=TEXT, size=11)),
        uirevision=f"{view}-{frame_label}-{focus_key}",
        meta=dict(clock or {}, index=int(index)))
    return figure


def wire_sphere(radius, color, name, n_circles=3, n_points=72):
    """Three great circles of a sphere about the origin as one line trace: a light keep-out marker."""
    angle = np.linspace(0.0, 2.0 * np.pi, n_points)
    zero = np.zeros_like(angle)
    gap = np.array([np.nan])
    circles = [(np.cos(angle), np.sin(angle), zero), (np.cos(angle), zero, np.sin(angle)),
               (zero, np.cos(angle), np.sin(angle))][:n_circles]
    x = np.concatenate([np.concatenate([radius * c[0], gap]) for c in circles])
    y = np.concatenate([np.concatenate([radius * c[1], gap]) for c in circles])
    z = np.concatenate([np.concatenate([radius * c[2], gap]) for c in circles])
    return go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, hoverinfo="name",
                        line=dict(width=1.5, color=color, dash="dot"))


def relative_figure(relative_paths_km, target_name, index=0, keep_out_radius_km=0.0, centre="moon",
                    trail_samples=0, clock=None):
    """
    Relative-motion view: every other spacecraft in the LVLH frame of a
    target, which sits at the origin.  This is the picture rendezvous is
    flown in.

    relative_paths_km  : {name: (n, 3)} radial, along-track, cross-track
                         position of each spacecraft relative to the
                         target, km (engine.rendezvous.relative_motion_lvlh)
    target_name        : name shown at the origin
    index              : time index of the current-time markers
    keep_out_radius_km : radius of the keep-out sphere around the target
    centre             : body the LVLH frame is defined about, for labels
    trail_samples, clock : as in trajectory_figure

    Plot axes: x is along-track (V-bar), y is cross-track (H-bar) and z
    is radial (R-bar, positive away from the body), so the orbit plane
    is the x-z plane and "behind the target" is negative x.  Traces carry
    the same meta roles as trajectory_figure, so the playback script
    moves them without knowing which kind of view it is.
    """
    figure = go.Figure()

    def plot_axes(path_km):
        """Columns radial, along, cross reordered to plot x, y, z."""
        return path_km[:, 1], path_km[:, 2], path_km[:, 0]

    window_points = []
    for k, (name, path_km) in enumerate(relative_paths_km.items()):
        color = SPACECRAFT_COLORS[(k + 1) % len(SPACECRAFT_COLORS)]
        stride = max(1, int(np.ceil(len(path_km) / MAX_PLOT_POINTS)))
        x, y, z = plot_axes(path_km[::stride])
        window_points.append(np.column_stack(plot_axes(path_km)))
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name,
                                      line=dict(width=2.0, color=color), opacity=0.6,
                                      meta={"role": "path", "spacecraft": name, "stride": stride},
                                      hovertemplate=(f"{name}<br>along-track %{{x:.1f}} km<br>cross-track %{{y:.1f}} km"
                                                     "<br>radial %{z:.1f} km<extra></extra>")))
        if trail_samples > 0:
            picks = np.clip(np.arange(index - trail_samples, index + 1), 0, len(path_km) - 1)
            tx, ty, tz = plot_axes(path_km[picks])
            fade = np.linspace(0.0, 1.0, len(picks))
            figure.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines", showlegend=False, hoverinfo="skip",
                                          meta={"role": "trail", "spacecraft": name, "samples": int(trail_samples)},
                                          line=dict(width=6, color=fade, cmin=0.0, cmax=1.0,
                                                    colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, color]])))
        now = path_km[min(index, len(path_km) - 1)]
        nx, ny, nz = now[1], now[2], now[0]
        figure.add_trace(go.Scatter3d(x=[nx], y=[ny], z=[nz], mode="markers", showlegend=False, hoverinfo="skip",
                                      meta={"role": "halo", "spacecraft": name},
                                      marker=dict(size=14, color=color, opacity=0.25)))
        figure.add_trace(go.Scatter3d(x=[nx], y=[ny], z=[nz], mode="markers", showlegend=False,
                                      name=f"{name} (now)", meta={"role": "marker", "spacecraft": name},
                                      marker=dict(size=6, color="#ffffff", line=dict(color=color, width=2))))

    # The target at the origin, and the axes through it: V-bar and R-bar
    # are the lines approaches are flown along.
    figure.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="markers+text", name=target_name,
                                  text=[target_name], textposition="top center",
                                  textfont=dict(color=TEXT_SECONDARY, size=11),
                                  marker=dict(size=5, color=SPACECRAFT_COLORS[0], symbol="diamond")))
    if keep_out_radius_km > 0.0:
        figure.add_trace(wire_sphere(keep_out_radius_km, UNSTABLE, f"keep-out {keep_out_radius_km:g} km"))
        window_points.append(np.array([[keep_out_radius_km] * 3, [-keep_out_radius_km] * 3]))

    window_points.append(np.zeros((1, 3)))
    lower, upper = view_window(window_points, margin_fraction=0.12)
    # Keep the box from collapsing when the motion is nearly a straight line.
    span = upper - lower
    floor = 0.25 * span.max()
    for axis in range(3):
        if span[axis] < floor:
            middle = 0.5 * (lower[axis] + upper[axis])
            lower[axis] = middle - 0.5 * floor
            upper[axis] = middle + 0.5 * floor
    span = upper - lower
    ratio = span / span.max()

    for axis, label in ((0, "V-bar"), (2, "R-bar")):
        ends = np.zeros((2, 3))
        ends[0, axis] = lower[axis]
        ends[1, axis] = upper[axis]
        figure.add_trace(go.Scatter3d(x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="lines+text", name=label,
                                      text=["", label], textposition="middle right", showlegend=False,
                                      textfont=dict(color=TEXT_MUTED, size=10), hoverinfo="skip",
                                      line=dict(width=1, color="#3a3a3a")))

    figure.update_layout(
        scene=dict(xaxis=dict(title=f"along-track [km], LVLH of {target_name} about the {centre.capitalize()}",
                              range=[lower[0], upper[0]], **SCENE_AXIS),
                   yaxis=dict(title="cross-track [km]", range=[lower[1], upper[1]], **SCENE_AXIS),
                   zaxis=dict(title="radial [km]", range=[lower[2], upper[2]], **SCENE_AXIS),
                   aspectmode="manual", aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
                   dragmode="turntable",
                   camera=dict(eye=dict(x=0.35, y=-1.45, z=0.45), up=dict(x=0, y=0, z=1))),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=TEXT_SECONDARY, size=12),
        margin=dict(l=0, r=0, t=34, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=TEXT_SECONDARY)),
        hoverlabel=dict(bgcolor="#111111", bordercolor="#303030",
                        font=dict(family="IBM Plex Mono, Menlo, monospace", color=TEXT, size=11)),
        uirevision=f"lvlh-{target_name}",
        meta=dict(clock or {}, index=int(index)))
    return figure


def heliocentric_figure(paths_au, index=0, trail_samples=0, clock=None):
    """
    Sun-centred view of an interplanetary transfer.

    paths_au : {name: (n, 3)} daily positions in AU, ecliptic axes, from
               model.interplanetary.transfer_scene; "Earth" and "Mars"
               get the planet colours, anything else a spacecraft colour
    Traces carry the same meta roles as trajectory_figure so the playback
    script moves the markers and trails.
    """
    body_colors = {"Earth": EARTHSHINE, "Mars": "#d9775a"}
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="markers+text", name="Sun", text=["Sun"],
                                  textposition="top center", textfont=dict(color=TEXT_SECONDARY, size=11),
                                  marker=dict(size=7, color=SUN), hoverinfo="name"))
    for name, path in paths_au.items():
        color = body_colors.get(name, "#f2f2f2")
        figure.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", name=name,
                                      line=dict(width=2.0, color=color), opacity=0.55,
                                      meta={"role": "path", "spacecraft": name, "stride": 1},
                                      hovertemplate=f"{name}<br>%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}} AU<extra></extra>"))
        if trail_samples > 0:
            picks = np.clip(np.arange(index - trail_samples, index + 1), 0, len(path) - 1)
            fade = np.linspace(0.0, 1.0, len(picks))
            figure.add_trace(go.Scatter3d(x=path[picks, 0], y=path[picks, 1], z=path[picks, 2], mode="lines",
                                          showlegend=False, hoverinfo="skip",
                                          meta={"role": "trail", "spacecraft": name, "samples": int(trail_samples)},
                                          line=dict(width=6, color=fade, cmin=0.0, cmax=1.0,
                                                    colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, color]])))
        now = path[min(index, len(path) - 1)]
        figure.add_trace(go.Scatter3d(x=[now[0]], y=[now[1]], z=[now[2]], mode="markers", showlegend=False,
                                      hoverinfo="skip", meta={"role": "halo", "spacecraft": name},
                                      marker=dict(size=14, color=color, opacity=0.25)))
        figure.add_trace(go.Scatter3d(x=[now[0]], y=[now[1]], z=[now[2]], mode="markers", showlegend=False,
                                      name=f"{name} (now)", meta={"role": "marker", "spacecraft": name},
                                      marker=dict(size=6, color="#ffffff", line=dict(color=color, width=2))))

    extent = 1.08 * max(np.abs(path[:, :2]).max() for path in paths_au.values())
    figure.update_layout(
        scene=dict(xaxis=dict(title="x [AU], ecliptic, Sun-centred", range=[-extent, extent], **SCENE_AXIS),
                   yaxis=dict(title="y [AU]", range=[-extent, extent], **SCENE_AXIS),
                   zaxis=dict(title="z [AU]", range=[-0.25 * extent, 0.25 * extent], **SCENE_AXIS),
                   aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=0.25), dragmode="turntable",
                   camera=dict(eye=dict(x=0.0, y=-0.95, z=0.85), up=dict(x=0, y=0, z=1))),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=TEXT_SECONDARY, size=12),
        margin=dict(l=0, r=0, t=34, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=TEXT_SECONDARY)),
        hoverlabel=dict(bgcolor="#111111", bordercolor="#303030",
                        font=dict(family="IBM Plex Mono, Menlo, monospace", color=TEXT, size=11)),
        uirevision="heliocentric", meta=dict(clock or {}, index=int(index)))
    return figure


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------

SERIES_BLUE = EARTHSHINE
SERIES_ORANGE = SUN
SERIES_AQUA = STABLE

PLOT_LAYOUT = dict(template="plotly_dark",
                   font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color=TEXT_SECONDARY),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   hoverlabel=dict(bgcolor="#111111", bordercolor="#303030",
                                   font=dict(family="IBM Plex Mono, Menlo, monospace", color=TEXT, size=11)))


GROUND_PANELS = [("elevation_deg", "Elevation above horizon [deg]", "min elevation"),
                 ("apparent_magnitude", "Apparent magnitude (brighter is up)", "limiting magnitude"),
                 ("lunar_separation_deg", "Angular separation from the Moon [deg]", "lunar exclusion")]
SPACE_PANELS = [("range_km", "Range to target [km]", "maximum range"),
                ("apparent_magnitude", "Apparent magnitude (brighter is up)", "limiting magnitude"),
                ("sun_separation_deg", "Angle between line of sight and Sun [deg]", "sun exclusion")]


def time_series_figure(series, thresholds, windows, current_time_s=None, panels=None):
    """
    Three quantities against time, with constraint thresholds as dashed
    lines, access windows shaded, and an optional vertical marker at the
    current time.

    series     : engine.geometry.GeometrySeries
    thresholds : {attribute name: value or None}
    windows    : list of (start_s, stop_s)
    panels     : three (attribute of the series, title, threshold label);
                 GROUND_PANELS (the default) suits a ground telescope,
                 SPACE_PANELS a camera on a spacecraft
    """
    from plotly.subplots import make_subplots

    days = np.asarray(series.time_s) / 86400.0
    stride = max(1, int(np.ceil(len(days) / MAX_PLOT_POINTS)))
    shown_days = days[::stride]

    # Three small multiples side by side: the strip stays short so the
    # 3D scene keeps the height.
    panels = panels or GROUND_PANELS
    figure = make_subplots(rows=1, cols=3, horizontal_spacing=0.05,
                           subplot_titles=tuple(title for _, title, _ in panels))
    colors = [SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA]

    for col, (key, _, threshold_name) in enumerate(panels, start=1):
        values = getattr(series, key)
        color = colors[col - 1]
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

    for col, (key, _, _) in enumerate(panels, start=1):
        if key == "apparent_magnitude":
            figure.update_yaxes(autorange="reversed", row=1, col=col)
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
