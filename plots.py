"""
Figures for the Earth-Moon L2 southern halo family.

Every function takes converged orbits (dictionaries from corrector.py),
draws one figure, saves it as a PNG in the output directory and returns
the path.  Positions are plotted in non-dimensional units (LU) with the
kilometre scale given in the axis labels; the Moon is drawn to scale.

Run as a script to produce all figures.  The family is loaded from
output/halo_family.npz if validate.py has already produced it, otherwise
it is computed here.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from engine import crtbp, corrector
from engine.crtbp import MU
from model.family import FAMILY_FILE, load_family

OUTPUT_DIR = "output"

# Approximate period of a 9:2 lunar-synodic resonant NRHO (the Gateway
# orbit): two revolutions per nine synodic months, in TU.  Used to pick a
# representative NRHO out of the family.
SYNODIC_MONTH_DAYS = 29.530589
PERIOD_9_2_TU = crtbp.time_to_nondim(2.0 / 9.0 * SYNODIC_MONTH_DAYS * crtbp.SECONDS_PER_DAY)

# Stability index bound used to shade the NRHO region.
NRHO_STABILITY_BOUND = 2.0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def axis_label(name):
    """Axis label with both the non-dimensional and the kilometre scale."""
    return f"{name} [LU]   (1 LU = {crtbp.LENGTH_UNIT_KM:,.0f} km)"


def short_axis_label(name):
    """Compact axis label for 3D axes, where the long form collides with the ticks."""
    return f"{name} [LU]"


SCALE_NOTE = f"1 LU = {crtbp.LENGTH_UNIT_KM:,.0f} km"


def masked_segments(points, keep):
    """
    Copy of `points` with rows outside `keep` set to NaN, so matplotlib
    breaks the line there instead of joining the exit and re-entry points.
    """
    masked = points.astype(float).copy()
    masked[~keep] = np.nan
    return masked


def set_equal_aspect_3d(ax, points, padding=0.05):
    """
    Give a 3D axis equal scale on all three directions.

    Matplotlib stretches 3D axes to fill the box by default, which turns a
    near-rectilinear orbit into something that looks like a fat ellipse.
    The limits are set to the bounding box of the points plus a margin and
    the box aspect is set to the same proportions.
    """
    points = np.asarray(points)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = upper - lower
    margin = padding * span.max()
    lower = lower - margin
    upper = upper + margin
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower)


def draw_moon_3d(ax, mu=MU, color="0.5", resolution=40):
    """Draw the Moon as a sphere of its true radius at (1 - mu, 0, 0)."""
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    radius = crtbp.MOON_RADIUS_ND
    x = 1.0 - mu + radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.9, linewidth=0, shade=True)


def draw_moon_2d(ax, horizontal_axis, vertical_axis, mu=MU):
    """Draw the Moon as a disc in a 2D projection (axes given as 0, 1, 2)."""
    centre = crtbp.moon_position(mu)
    disc = plt.Circle((centre[horizontal_axis], centre[vertical_axis]),
                      crtbp.MOON_RADIUS_ND, color="0.5", zorder=3)
    ax.add_patch(disc)


def add_km_axes(ax):
    """Attach secondary axes in km to a 2D plot."""
    ax.secondary_xaxis("top", functions=(crtbp.length_to_km, crtbp.length_to_nondim)).set_xlabel("km")
    ax.secondary_yaxis("right", functions=(crtbp.length_to_km, crtbp.length_to_nondim)).set_ylabel("km")


def coloured_line_3d(ax, points, values, cmap, norm, linewidth=1.5):
    """Draw a 3D polyline whose colour varies along its length."""
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    collection.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection3d(collection)
    return collection


def pick_representative_nrho(family):
    """The family member whose period is closest to the 9:2 resonance."""
    periods = np.array([orbit["period"] for orbit in family])
    return family[int(np.argmin(np.abs(periods - PERIOD_9_2_TU)))]


def nrho_mask(family):
    """
    Boolean mask of the members counted as NRHOs: the contiguous run of
    the family with stability index below NRHO_STABILITY_BOUND that
    contains the smallest perilune radii.
    """
    nu = np.array([orbit["stability_index"] for orbit in family])
    mask = nu <= NRHO_STABILITY_BOUND
    # Walk back from the end (smallest perilune) until the index exceeds
    # the bound; everything from there on is the NRHO region.
    last_outside = np.where(~mask)[0]
    if len(last_outside) == 0:
        return mask
    region = np.zeros_like(mask)
    region[last_outside[-1] + 1:] = True
    return region


def altitude_km(radius_nd):
    """Height above the lunar surface, km, from a radius in LU."""
    return crtbp.length_to_km(radius_nd) - crtbp.MOON_RADIUS_KM


# --------------------------------------------------------------------------
# Figure 1: the family in 3D coloured by Jacobi constant
# --------------------------------------------------------------------------

def plot_family_3d(family, path=os.path.join(OUTPUT_DIR, "fig1_halo_family_3d.png"), mu=MU):
    jacobi = np.array([orbit["jacobi"] for orbit in family])
    norm = plt.Normalize(jacobi.min(), jacobi.max())
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    for orbit in family:
        _, states = corrector.propagate_orbit(orbit, mu, n_points=600)
        points = states[:, :3]
        all_points.append(points)
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
                color=cmap(norm(orbit["jacobi"])), linewidth=0.8)

    draw_moon_3d(ax, mu)
    libration = crtbp.collinear_libration_points(mu)
    for name in ("L1", "L2"):
        ax.scatter([libration[name]], [0.0], [0.0], color="red", s=30, depthshade=False)
        ax.text(libration[name], 0.0, 0.01, name, color="red")

    all_points = np.vstack(all_points + [[[libration["L1"], 0, 0], [libration["L2"], 0, 0]]])
    set_equal_aspect_3d(ax, all_points)
    ax.set_xlabel(short_axis_label("x"))
    ax.set_ylabel(short_axis_label("y"))
    ax.set_zlabel(short_axis_label("z"))
    ax.set_title(f"Earth-Moon L2 southern halo family, rotating frame   ({SCALE_NOTE})")
    ax.view_init(elev=22, azim=-55)

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(jacobi)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Jacobi constant C")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 2: one NRHO from three viewpoints
# --------------------------------------------------------------------------

def plot_nrho_three_views(orbit, path=os.path.join(OUTPUT_DIR, "fig2_nrho_three_views.png"), mu=MU):
    """
    xy and xz projections drawn as true 2D plots with equal aspect, plus
    an oblique 3D view.  Perilune and apolune are marked with altitudes.
    """
    _, states = corrector.propagate_orbit(orbit, mu, n_points=3000)
    points = states[:, :3]
    r_peri, peri_state, r_apo, apo_state = corrector.closest_and_farthest_approach(orbit, mu)
    peri_text = f"perilune, {altitude_km(r_peri):,.0f} km altitude"
    apo_text = f"apolune, {altitude_km(r_apo):,.0f} km altitude"

    fig = plt.figure(figsize=(17, 6.5))

    projections = [("xy view (from +z)", 0, 1, "x", "y"),
                   ("xz view (from -y)", 0, 2, "x", "z")]
    for k, (title, h, v, h_name, v_name) in enumerate(projections):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.plot(points[:, h], points[:, v], color="tab:blue", linewidth=1.2)
        draw_moon_2d(ax, h, v, mu)
        ax.scatter(peri_state[h], peri_state[v], color="red", zorder=4)
        ax.scatter(apo_state[h], apo_state[v], color="darkorange", zorder=4)
        ax.annotate(peri_text, xy=(peri_state[h], peri_state[v]), xytext=(8, 8),
                    textcoords="offset points", color="red", fontsize=9)
        ax.annotate(apo_text, xy=(apo_state[h], apo_state[v]), xytext=(8, -14),
                    textcoords="offset points", color="darkorange", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{h_name} [LU]")
        ax.set_ylabel(f"{v_name} [LU]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        add_km_axes(ax)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue", linewidth=1.2)
    draw_moon_3d(ax, mu)
    ax.scatter(*peri_state[:3], color="red", s=30, depthshade=False)
    ax.scatter(*apo_state[:3], color="darkorange", s=30, depthshade=False)
    ax.text(*peri_state[:3], "  " + peri_text, color="red", fontsize=8)
    ax.text(*apo_state[:3], "  " + apo_text, color="darkorange", fontsize=8)
    set_equal_aspect_3d(ax, points)
    ax.set_xlabel(short_axis_label("x"))
    ax.set_ylabel(short_axis_label("y"))
    ax.set_zlabel(short_axis_label("z"))
    ax.set_title("oblique view")
    ax.view_init(elev=25, azim=-60)

    fig.suptitle(f"L2 southern NRHO, period {crtbp.time_to_days(orbit['period']):.2f} days, "
                 f"C = {orbit['jacobi']:.4f}   ({SCALE_NOTE})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 3: the same NRHO coloured by speed
# --------------------------------------------------------------------------

def plot_nrho_speed(orbit, path=os.path.join(OUTPUT_DIR, "fig3_nrho_speed.png"), mu=MU):
    _, states = corrector.propagate_orbit(orbit, mu, n_points=4000)
    points = states[:, :3]
    speed_km_s = crtbp.velocity_to_km_s(np.linalg.norm(states[:, 3:], axis=1))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(speed_km_s.min(), speed_km_s.max())
    collection = coloured_line_3d(ax, points, speed_km_s, cmap, norm, linewidth=2.0)
    draw_moon_3d(ax, mu)
    set_equal_aspect_3d(ax, points)
    ax.set_xlabel(short_axis_label("x"))
    ax.set_ylabel(short_axis_label("y"))
    ax.set_zlabel(short_axis_label("z"))
    ax.set_title(f"NRHO coloured by rotating-frame speed   ({SCALE_NOTE})")
    ax.view_init(elev=20, azim=-50)
    fig.colorbar(collection, ax=ax, shrink=0.6, pad=0.1, label="speed [km/s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 4: zoom on the perilune passage
# --------------------------------------------------------------------------

def plot_perilune_zoom(orbit, path=os.path.join(OUTPUT_DIR, "fig4_perilune_zoom.png"), mu=MU,
                       window=0.03):
    """
    Three views of the perilune passage inside a box of half-width
    `window` LU around the Moon: the yz projection (the plane the pass
    mostly lies in, since perilune sits over the pole), a 3D zoom, and the
    distance to the lunar surface against time.
    """
    t, states = corrector.propagate_orbit(orbit, mu, n_points=8000)
    r_peri, peri_state, _, _ = corrector.closest_and_farthest_approach(orbit, mu)
    moon = crtbp.moon_position(mu)

    near = np.all(np.abs(states[:, :3] - moon) < window, axis=1)
    points = masked_segments(states[:, :3], near)
    radial = (peri_state[:3] - moon) / r_peri
    surface_point = moon + radial * crtbp.MOON_RADIUS_ND
    altitude_text = (f"perilune altitude {altitude_km(r_peri):,.0f} km\n"
                     f"(radius {crtbp.length_to_km(r_peri):,.0f} km)")

    fig = plt.figure(figsize=(18, 6))

    # Panel 1: yz projection.
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(points[:, 1], points[:, 2], color="tab:blue", linewidth=1.5)
    draw_moon_2d(ax, 1, 2, mu)
    ax.plot([surface_point[1], peri_state[1]], [surface_point[2], peri_state[2]], color="red", linewidth=2)
    ax.scatter(peri_state[1], peri_state[2], color="red", zorder=4)
    ax.annotate(altitude_text, xy=(peri_state[1], peri_state[2]), xytext=(15, -25),
                textcoords="offset points", color="red")
    ax.set_xlim(moon[1] - window, moon[1] + window)
    ax.set_ylim(moon[2] - window, moon[2] + window)
    ax.set_aspect("equal")
    ax.set_xlabel("y [LU]")
    ax.set_ylabel("z [LU]")
    ax.set_title("yz projection")
    ax.grid(True, alpha=0.3)
    add_km_axes(ax)

    # Panel 2: 3D zoom with the Moon as a sphere.
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue", linewidth=1.5)
    draw_moon_3d(ax, mu, resolution=60)
    ax.plot([surface_point[0], peri_state[0]], [surface_point[1], peri_state[1]],
            [surface_point[2], peri_state[2]], color="red", linewidth=2)
    ax.scatter(*peri_state[:3], color="red", s=30, depthshade=False)
    box = np.array([moon - window, moon + window])
    set_equal_aspect_3d(ax, box, padding=0.0)
    ax.set_xlabel(short_axis_label("x"))
    ax.set_ylabel(short_axis_label("y"))
    ax.set_zlabel(short_axis_label("z"))
    ax.set_title("3D zoom on the pass")
    ax.view_init(elev=15, azim=-40)

    # Panel 3: distance to the surface against time around perilune.
    ax = fig.add_subplot(1, 3, 3)
    radius = crtbp.distance_to_moon(states, mu)
    index_peri = int(np.argmin(radius))
    half_window_days = 0.5
    time_days = crtbp.time_to_days(t - t[index_peri])
    keep = np.abs(time_days) < half_window_days
    ax.plot(time_days[keep], altitude_km(radius[keep]), color="tab:blue")
    ax.axhline(0.0, color="0.4", linewidth=2, label="lunar surface")
    ax.scatter([0.0], [altitude_km(r_peri)], color="red", zorder=4)
    ax.annotate(f"{altitude_km(r_peri):,.0f} km", xy=(0.0, altitude_km(r_peri)),
                xytext=(10, 10), textcoords="offset points", color="red")
    ax.set_xlabel("time from perilune [days]")
    ax.set_ylabel("altitude above the lunar surface [km]")
    ax.set_title("altitude through the pass")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center")

    fig.suptitle(f"Perilune passage with the Moon to scale (radius {crtbp.MOON_RADIUS_KM:.0f} km)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 5: stability index against perilune radius
# --------------------------------------------------------------------------

def plot_stability_index(family, path=os.path.join(OUTPUT_DIR, "fig5_stability_index.png")):
    r_peri_km = np.array([crtbp.length_to_km(orbit["perilune_radius"]) for orbit in family])
    nu = np.array([orbit["stability_index"] for orbit in family])
    region = nrho_mask(family)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_peri_km, nu, marker="o", markersize=3, color="tab:blue")
    ax.set_yscale("log")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1, label="|nu| = 1 (stability boundary)")
    if region.any():
        ax.axvspan(r_peri_km[region].min(), r_peri_km[region].max(), color="tab:green", alpha=0.15,
                   label=f"NRHO region (|nu| < {NRHO_STABILITY_BOUND:g})")
    ax.axvline(crtbp.MOON_RADIUS_KM, color="0.5", linestyle=":", label="lunar surface")
    ax.set_xlabel("perilune radius [km]")
    ax.set_ylabel("stability index  max |(lambda + 1/lambda) / 2|")
    ax.set_title("Stability index along the L2 southern halo family")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    secondary = ax.secondary_xaxis("top", functions=(crtbp.length_to_nondim, crtbp.length_to_km))
    secondary.set_xlabel("perilune radius [LU]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 6: animation of a spacecraft on the NRHO
# --------------------------------------------------------------------------

def animate_nrho(orbit, path=os.path.join(OUTPUT_DIR, "fig6_nrho_animation.gif"), mu=MU,
                 n_frames=180, fps=24):
    """
    Save a gif of a marker moving along the orbit in the rotating frame.
    Frames are equally spaced in time, so the marker visibly lingers at
    apolune and whips through perilune.
    """
    t, states = corrector.propagate_orbit(orbit, mu, n_points=n_frames)
    points = states[:, :3]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="0.7", linewidth=1)
    draw_moon_3d(ax, mu)
    set_equal_aspect_3d(ax, points)
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("y [LU]")
    ax.set_zlabel("z [LU]")
    ax.view_init(elev=20, azim=-50)

    trail, = ax.plot([], [], [], color="tab:blue", linewidth=2)
    marker, = ax.plot([], [], [], marker="o", color="red", markersize=6)
    label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        trail_start = max(frame - 30, 0)
        trail.set_data(points[trail_start:frame + 1, 0], points[trail_start:frame + 1, 1])
        trail.set_3d_properties(points[trail_start:frame + 1, 2])
        marker.set_data([points[frame, 0]], [points[frame, 1]])
        marker.set_3d_properties([points[frame, 2]])
        label.set_text(f"t = {crtbp.time_to_days(t[frame]):.2f} days")
        return trail, marker, label

    movie = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Family loading
# --------------------------------------------------------------------------

def load_or_build_family():
    """Load the family saved by validate.py, or compute it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return load_family()


def make_all_figures(family):
    """Produce every figure and return the list of paths."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nrho = pick_representative_nrho(family)
    paths = [plot_family_3d(family),
             plot_nrho_three_views(nrho),
             plot_nrho_speed(nrho),
             plot_perilune_zoom(nrho),
             plot_stability_index(family),
             animate_nrho(nrho)]
    return paths


if __name__ == "__main__":
    halo_family = load_or_build_family()
    for figure_path in make_all_figures(halo_family):
        print("saved", figure_path)
