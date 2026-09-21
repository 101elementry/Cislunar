"""
Static showcase: a handful of precomputed scenarios written out as plain
web pages that need no Python behind them.  Run from the repository
root:

    python -m app.showcase

and open site/index.html, or point any static host (Vercel, GitHub
Pages) at the site/ folder.

Each page holds the same Plotly figures the interface draws
(app/scene.py), with the same browser-side playback and zoom scripts
(app/assets/playback.js and zoom_to_cursor.js), so a scene can be
rotated, zoomed and played like a video.  The inputs cannot be changed
because nothing runs on a server.

Like the rest of app/, nothing here computes physics: every scene is a
model.scenario.Scenario handed to model.runner.run_scenario.
"""

import html
import json
import os
import shutil

import numpy as np

from engine import crtbp
from model import runner
from model.ephemeris import load_ephemeris
from model.family import load_families
from model.scenario import Scenario, Spacecraft, GroundStation, OpticalSensor, ELEMENT_PRESETS
from app import figures
from app.scene import BODY_RADII, FRAME_LABELS, displayed_frame, scene_figure

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_DIRECTORY = os.path.join(REPOSITORY_ROOT, "site")
ASSET_SOURCES = [os.path.join(REPOSITORY_ROOT, "app", "assets", "playback.js"),
                 os.path.join(REPOSITORY_ROOT, "app", "assets", "zoom_to_cursor.js"),
                 os.path.join(REPOSITORY_ROOT, "app", "showcase_assets", "showcase.css"),
                 os.path.join(REPOSITORY_ROOT, "app", "showcase_assets", "showcase.js")]

# The interface gets Plotly from Dash; the static pages load the same
# version from Plotly's CDN.  The playback script reaches into Plotly's
# WebGL objects, so the version is pinned to the one it was tested with.
PLOTLY_SCRIPT = "https://cdn.plot.ly/plotly-4.0.0.min.js"

AUTHOR = "Connor Wherry"
AUTHOR_LINE = "Aeronautical Engineering (Space), University of Sydney"

FAMILIES = load_families()
EPHEMERIS = load_ephemeris()


# --------------------------------------------------------------------------
# The scenes.  Each function returns a dictionary:
#   slug, kicker, title, tagline : page identity
#   scenario                     : model.scenario.Scenario to run
#   views                        : one or two (frame, view, focus) triples
#   pair                         : optional (observer, spacecraft) for the
#                                  time series strip and the access light
#   paragraphs, look_for         : the words on the page
#   facts(results)               : function returning [(label, value), ...]
# --------------------------------------------------------------------------

def orbit_facts(orbit):
    """Period, perilune and stability index of a family member as text."""
    return (f"{crtbp.time_to_days(orbit['period']):.2f} d",
            f"{crtbp.length_to_km(orbit['perilune_radius']):,.0f} km",
            f"{float(orbit['stability_index']):,.2f}")


def scene_halo_to_nrho():
    family = FAMILIES["L2 southern halo"]
    members = [0, 10, 20, 30, 40, 49, 60, 68]
    scenario = Scenario(name="From halo to NRHO", epoch_utc="2026-01-01T00:00:00",
                        duration_days=15.0, time_step_s=300.0)
    for index in members:
        label = f"Member {index}" + (", the 9:2 NRHO" if index == 49 else "")
        scenario.add(Spacecraft(name=label, source="family", family_name="L2 southern halo",
                                family_index=index, propagation="periodic"))

    def facts(results):
        rows = []
        for index in (0, 20, 49, 68):
            period, perilune, stability = orbit_facts(family[index])
            rows.append((f"Member {index}", f"{period}, perilune {perilune}, stability index {stability}"))
        return rows

    return {"slug": "halo-to-nrho", "kicker": "Orbit families", "title": "From halo orbit to NRHO",
            "tagline": "One family of orbits around L2, from a wide halo down to the orbit chosen for Gateway.",
            "scenario": scenario, "views": [("rotating", "moon", "none")], "pair": None,
            "paragraphs": [
                "Every curve here is one member of the same family of periodic orbits around the Earth-Moon L2 "
                "point, drawn in the frame that rotates with the Moon. Each member was found by differential "
                "correction, and the family was grown by stepping from one converged orbit to the next.",
                "The first member is a small, nearly flat halo that hugs L2, far from the Moon. Step along the "
                "family and the orbit grows, tips upright, its closest approach to the Moon falls, and its "
                "instability almost disappears. "
                "That end of the family is the near rectilinear halo orbits. Member 49 completes nine laps "
                "for every two lunar months, and it is the orbit NASA chose for the Gateway station."],
            "look_for": [
                "The spacecraft on the small orbits race past the Moon and linger over the south pole.",
                "The stability index falls from several hundred on the first halo to about one on the NRHOs. "
                "A value of one means a small error no longer grows from lap to lap.",
                "Rotate the view edge on. The NRHOs look almost like straight lines, which is where the name "
                "comes from."],
            "facts": facts}


def scene_manifolds():
    family = FAMILIES["L2 southern halo"]
    index = 8
    scenario = Scenario(name="Invariant manifolds", epoch_utc="2026-01-01T00:00:00",
                        duration_days=float(crtbp.time_to_days(family[index]["period"])) * 2.0, time_step_s=300.0)
    scenario.add(Spacecraft(name="L2 halo", source="family", family_name="L2 southern halo", family_index=index,
                            propagation="periodic", manifolds="both", manifold_branches=40,
                            manifold_time_days=22.0))

    def facts(results):
        period, perilune, stability = orbit_facts(family[index])
        return [("Orbit", f"L2 southern halo, member {index}"), ("Period", period),
                ("Stability index", stability),
                ("Branches", "40 leaving and 40 arriving, each followed for 22 days")]

    return {"slug": "manifolds", "kicker": "Transport", "title": "Free paths on and off an orbit",
            "tagline": "The invariant manifolds of an unstable halo orbit, the natural routes of the "
                       "Earth-Moon system.",
            "scenario": scenario, "views": [("rotating", "moon", "none")], "pair": None,
            "paragraphs": [
                "This halo orbit is unstable. Nudge a spacecraft by a few centimetres per second in one "
                "particular direction and it peels away along a definite path. Do that from every point around "
                "the orbit and the paths form a tube. The red tube is the unstable manifold, the set of paths "
                "that leave the orbit.",
                "The green tube is the stable manifold. A spacecraft placed on it winds onto the orbit with no "
                "burn at all. The directions come from the eigenvectors of the monodromy matrix, which is the "
                "state transition matrix over one full lap. Mission designers use these tubes as low cost "
                "routes between the Earth, the Moon and the Lagrange points."],
            "look_for": [
                "One half of each tube heads toward the Moon and the other half escapes to the far side.",
                "Zoom toward the orbit. The red and green paths meet it tangentially, because close to the "
                "orbit they differ from it by almost nothing.",
                "A stable orbit such as a distant retrograde orbit has no tubes at all."],
            "facts": facts}


def scene_dro_two_frames():
    family = FAMILIES["DRO"]
    index = 12
    scenario = Scenario(name="DRO in two frames", epoch_utc="2026-01-01T00:00:00",
                        duration_days=float(crtbp.time_to_days(family[index]["period"])) * 2.0, time_step_s=300.0)
    scenario.add(Spacecraft(name="Distant retrograde orbit", source="family", family_name="DRO",
                            family_index=index, propagation="periodic"))

    def facts(results):
        period, perilune, stability = orbit_facts(family[index])
        return [("Period", period), ("Closest approach to the Moon", perilune),
                ("Farthest from the Moon", f"{crtbp.length_to_km(family[index]['apolune_radius']):,.0f} km"),
                ("Stability index", f"{stability}, linearly stable")]

    return {"slug": "dro-two-frames", "kicker": "Reference frames", "title": "One orbit, two frames",
            "tagline": "A distant retrograde orbit seen rotating with the Moon and seen from inertial space.",
            "scenario": scenario, "views": [("rotating", "system", "none"), ("earth_inertial", "system", "none")],
            "pair": None,
            "paragraphs": [
                "Both panels show the same spacecraft at the same instant. On the left the frame rotates with "
                "the Moon, so the Earth and Moon stand still and the spacecraft circles the Moon backwards, "
                "about 70,000 km out.",
                "On the right the frame is fixed to the stars and centred on the Earth. Now the Moon moves, and "
                "the spacecraft is plainly in orbit around the Earth. It drifts a little ahead of the Moon and "
                "then a little behind, and the rotating frame turns that drift into a loop. The orbit is so "
                "large that the Earth pulls on the spacecraft about as hard as the Moon does, which is why "
                "two-body thinking fails here.",
                "Orbits like this are stable for decades with no station keeping. Artemis I flew Orion into one "
                "in 2022."],
            "look_for": [
                "Press play and watch the two markers together. The loop on the left is the wobble on the right.",
                "The playback moves the Earth and Moon in the inertial panel and leaves them fixed in the "
                "rotating one."],
            "facts": facts}


def scene_sydney_tracking():
    scenario = Scenario(name="NRHO from Sydney", epoch_utc="2026-01-01T00:00:00",
                        duration_days=30.0, time_step_s=120.0)
    scenario.add(Spacecraft(name="Gateway-like NRHO", source="family", family_name="L2 southern halo",
                            family_index=49, propagation="periodic", diameter_m=6.0, albedo=0.25))
    scenario.add(GroundStation(name="Sydney", latitude_deg=-33.87, longitude_deg=151.21, altitude_km=0.05,
                               min_elevation_deg=15.0, max_sun_elevation_deg=-12.0))
    scenario.add(OpticalSensor(name="Sydney 0.5 m telescope", station="Sydney",
                               limiting_magnitude=18.5, lunar_exclusion_deg=2.0))
    pair = ("Sydney 0.5 m telescope", "Gateway-like NRHO")

    def facts(results):
        windows = results["windows"][pair]
        lengths_h = [(stop - start) / 3600.0 for start, stop in windows]
        return [("Access windows in 30 days", f"{len(windows)}"),
                ("Longest window", f"{max(lengths_h):.1f} h" if lengths_h else "none"),
                ("Fraction of the month observable", f"{100.0 * results['duty_cycle'][pair]:.1f} %"),
                ("Sky model", results["sky_model"])]

    return {"slug": "sydney-tracking", "kicker": "Observation", "title": "Tracking the NRHO from Sydney",
            "tagline": "When can one optical telescope in Sydney actually see a spacecraft near the Moon?",
            "scenario": scenario, "views": [("rotating", "moon", "none"), ("earth_inertial", "system", "none")],
            "pair": pair,
            "paragraphs": [
                "A telescope sees a spacecraft only when several things are true at once. The spacecraft is "
                "above the local horizon. The sky at the site is dark. The spacecraft is lit by the Sun, bright "
                "enough to detect, and not lost in the glare of the Moon. This scene applies all of those "
                "tests to a Gateway-like spacecraft for one month, with the positions of the Sun and Moon "
                "taken from the JPL DE440 ephemeris.",
                "The strip below the scene shows elevation, brightness and separation from the Moon. The green "
                "bands are the access windows, the only times a measurement can be taken. An orbit "
                "determination filter has to work with these sparse windows and nothing else, and that is "
                "the problem my thesis work is aimed at."],
            "look_for": [
                "The access light beside the clock turns green when Sydney can observe the spacecraft.",
                "Around new Moon there are no windows for several nights, because the Moon is up only in "
                "daylight.",
                "The gold track in the right hand panel is Sydney carried around by the rotating Earth."],
            "facts": facts}


def scene_lunar_relay():
    centre, plane, elements = ELEMENT_PRESETS["Elliptical lunar frozen orbit (12 h relay)"]
    scenario = Scenario(name="Lunar relay orbit", epoch_utc="2026-01-01T00:00:00",
                        duration_days=3.0, time_step_s=60.0)
    scenario.add(Spacecraft(name="South pole relay", source="elements", elements=dict(elements),
                            centre=centre, reference_plane=plane, propagation="integrate"))

    def facts(results):
        a_km = elements["a_km"]
        e = elements["e"]
        moon_radius_km = crtbp.length_to_km(crtbp.MOON_RADIUS_ND)
        return [("Semi-major axis", f"{a_km:,.0f} km"), ("Eccentricity", f"{e:.2f}"),
                ("Inclination", f"{elements['i_deg']:.0f} deg to the Moon's orbit plane"),
                ("Altitude at apolune", f"{a_km * (1.0 + e) - moon_radius_km:,.0f} km, over the south pole"),
                ("Altitude at perilune", f"{a_km * (1.0 - e) - moon_radius_km:,.0f} km")]

    return {"slug": "lunar-relay", "kicker": "Two-body elements", "title": "A relay orbit for the lunar south pole",
            "tagline": "An elliptical frozen orbit that hangs over the south pole for most of each lap.",
            "scenario": scenario, "views": [("moon_inertial", "moon", "none"), ("moon_rotating", "moon", "none")],
            "pair": None,
            "paragraphs": [
                "Rovers at the lunar south pole cannot see the Earth most of the time, so they need a relay. "
                "This orbit is the kind planned for that job. It is a twelve hour ellipse with its high point "
                "over the south pole. A spacecraft moves slowly near the high point, so it spends most of each "
                "lap in view of the pole and then whips around the north in under two hours.",
                "The orbit is defined here by classical orbital elements about the Moon, then converted to the "
                "rotating frame and propagated with the full three-body equations. The Earth's pull is "
                "included, so the ellipse is not perfectly closed. The orbit is called frozen because its "
                "shape and orientation were chosen so that this pull does not slowly turn it over."],
            "look_for": [
                "On the left the frame is fixed to the stars and the ellipse holds still. On the right the "
                "frame turns with the Moon once a month, so the same ellipse slowly precesses.",
                "Watch how long the marker stays below the Moon compared with above it."],
            "facts": facts}


SCENES = [scene_halo_to_nrho, scene_manifolds, scene_dro_two_frames, scene_sydney_tracking, scene_lunar_relay]


# --------------------------------------------------------------------------
# Page building
# --------------------------------------------------------------------------

def embedded_json(data):
    """JSON text safe to place inside a <script> element."""
    return json.dumps(data).replace("</", "<\\/")


def thumbnail_svg(trajectories, manifolds, bodies, body_radii, size=(320, 200)):
    """
    A small SVG drawing of the trajectories (and manifold branches, if
    any) for the index page: an orthographic projection along the same
    direction the 3D scene's default camera looks from, with the Moon
    as a disc.
    """
    direction = np.array([0.75, -1.0, 0.45])
    direction = direction / np.linalg.norm(direction)
    right = np.cross(-direction, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, -direction)

    def project(points):
        points = np.atleast_2d(points)
        return np.column_stack([points @ right, -(points @ up)])

    curves = [project(figures.subsample(states[:, :3], 240)) for states in trajectories.values()]
    tubes = []
    for kinds in manifolds.values():
        for kind, branches in kinds.items():
            for branch in branches[::2]:
                tubes.append((kind, project(figures.subsample(branch["states"][:, :3], 80))))
    moon = np.asarray(bodies["moon"])
    moon_centre = project(moon if moon.ndim == 1 else moon[0])[0]
    everything = np.vstack(curves + [tube for _, tube in tubes]
                           + [moon_centre + body_radii["moon"], moon_centre - body_radii["moon"]])
    lower = everything.min(axis=0)
    upper = everything.max(axis=0)
    scale = 0.84 * min(size[0] / (upper[0] - lower[0]), size[1] / (upper[1] - lower[1]))
    offset = np.array(size) / 2.0 - scale * (lower + upper) / 2.0

    parts = [f'<svg viewBox="0 0 {size[0]} {size[1]}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">']
    tube_colors = {"unstable": figures.UNSTABLE, "stable": figures.STABLE}
    for kind, tube in tubes:
        pixels = tube * scale + offset
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pixels)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{tube_colors[kind]}" stroke-width="0.7" '
                     f'opacity="0.55"/>')
    for k, curve in enumerate(curves):
        pixels = curve * scale + offset
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pixels)
        color = figures.SPACECRAFT_COLORS[k % len(figures.SPACECRAFT_COLORS)]
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.4" '
                     f'stroke-linejoin="round" opacity="0.9"/>')
    centre = moon_centre * scale + offset
    radius = max(2.5, body_radii["moon"] * scale)
    parts.append(f'<circle cx="{centre[0]:.1f}" cy="{centre[1]:.1f}" r="{radius:.1f}" fill="{figures.REGOLITH}"/>')
    parts.append("</svg>")
    return "".join(parts)


def page_shell(title, description, body, extra_head="", scripts=""):
    """The HTML document around a page body."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="stylesheet" href="assets/showcase.css">
{extra_head}
</head>
<body>
{body}
{scripts}
</body>
</html>
"""


def scene_page(spec, results, neighbours):
    """HTML of one scene page."""
    scenario = spec["scenario"]
    scene_figures = [scene_figure(scenario, results, frame, view, focus, 0) for frame, view, focus in spec["views"]]
    for figure in scene_figures:
        # The current-time markers need no legend entry of their own here;
        # with eight spacecraft they would fill the top of the scene.
        figure.for_each_trace(lambda trace: trace.update(showlegend=False),
                              selector=lambda trace: (trace.meta or {}).get("role") == "marker")
    page_data = {"figures": [json.loads(figure.to_json()) for figure in scene_figures],
                 "n_samples": int(len(results["times_s"])),
                 "time_step_s": float(scenario.time_step_s),
                 "windows_s": None, "series": None}
    series_block = ""
    access_light = ""
    if spec["pair"] is not None:
        station, sensor = runner.observer_settings(scenario, spec["pair"][0])
        thresholds = {"elevation_deg": station.min_elevation_deg,
                      "apparent_magnitude": sensor.limiting_magnitude,
                      "lunar_separation_deg": sensor.lunar_exclusion_deg}
        series_figure = figures.time_series_figure(results["observations"][spec["pair"]]["geometry"], thresholds,
                                                   results["windows"][spec["pair"]])
        series_figure.update_layout(height=None, autosize=True)
        page_data["series"] = json.loads(series_figure.to_json())
        page_data["windows_s"] = [[float(start), float(stop)] for start, stop in results["windows"][spec["pair"]]]
        series_block = '<div class="series"><div id="series" class="plot"></div></div>'
        access_light = '<span id="access-light" class="access-light">no access</span>'

    stage_class = "stage split" if len(scene_figures) == 2 else "stage"
    holders = ["view-3d", "view-3d-b"]
    views_html = ""
    for k, (frame, view, focus) in enumerate(spec["views"]):
        views_html += (f'<figure class="view"><figcaption>{html.escape(FRAME_LABELS[frame])}</figcaption>'
                       f'<div id="{holders[k]}" class="holder"><div class="plot"></div></div></figure>')

    paragraphs = "".join(f"<p>{html.escape(text)}</p>" for text in spec["paragraphs"])
    look_for = "".join(f"<li>{html.escape(text)}</li>" for text in spec["look_for"])
    facts = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
                    for label, value in spec["facts"](results))
    previous_spec, next_spec = neighbours
    body = f"""
<header class="topbar">
  <a class="brand" href="index.html"><span class="brand-mark"></span>Cislunar</a>
  <nav class="pager">
    <a href="{previous_spec['slug']}.html" aria-label="Previous scene">&larr;</a>
    <a href="index.html">All scenes</a>
    <a href="{next_spec['slug']}.html" aria-label="Next scene">&rarr;</a>
  </nav>
</header>
<main class="scene-page">
  <section class="intro">
    <p class="kicker">{html.escape(spec['kicker'])}</p>
    <h1>{html.escape(spec['title'])}</h1>
    <p class="tagline">{html.escape(spec['tagline'])}</p>
  </section>
  <section class="{stage_class}">{views_html}</section>
  <section class="transport">
    <button id="play-button" class="play" type="button">Play</button>
    <label class="speed">Speed
      <select id="play-speed">
        <option value="2">2 h per second</option>
        <option value="6">6 h per second</option>
        <option value="12" selected>12 h per second</option>
        <option value="24">1 d per second</option>
        <option value="72">3 d per second</option>
      </select>
    </label>
    <input id="scrubber" type="range" min="0" max="{page_data['n_samples'] - 1}" value="0" step="1"
           aria-label="Time">
    <span id="time-readout" class="readout"></span>
    {access_light}
  </section>
  {series_block}
  <section class="notes">
    <div class="prose"><h2>What you are looking at</h2>{paragraphs}</div>
    <div class="side">
      <h2>Things to look for</h2><ul>{look_for}</ul>
      <h2>Numbers</h2><table class="facts">{facts}</table>
      <p class="hint">Drag to rotate. Scroll over an orbit to zoom toward it. Click a legend entry to hide it.</p>
    </div>
  </section>
</main>
<footer class="footer">{html.escape(AUTHOR)}. {html.escape(AUTHOR_LINE)}.</footer>
<script type="application/json" id="page-data">{embedded_json(page_data)}</script>
"""
    scripts = (f'<script src="{PLOTLY_SCRIPT}"></script>\n<script src="assets/playback.js"></script>\n'
               '<script src="assets/zoom_to_cursor.js"></script>\n<script src="assets/showcase.js"></script>')
    return page_shell(f"{spec['title']} | Cislunar", spec["tagline"], body, scripts=scripts)


def index_page(cards):
    """HTML of the landing page; cards is a list of (spec, thumbnail svg)."""
    cards_html = ""
    for spec, thumbnail in cards:
        cards_html += f"""
    <a class="card" href="{spec['slug']}.html">
      <div class="thumb">{thumbnail}</div>
      <p class="kicker">{html.escape(spec['kicker'])}</p>
      <h2>{html.escape(spec['title'])}</h2>
      <p>{html.escape(spec['tagline'])}</p>
    </a>"""
    body = f"""
<header class="topbar">
  <a class="brand" href="index.html"><span class="brand-mark"></span>Cislunar</a>
</header>
<main class="index-page">
  <section class="hero">
    <p class="kicker">Earth-Moon three-body dynamics</p>
    <h1>Orbits that only exist because there are two bodies.</h1>
    <p class="lede">Near the Moon a spacecraft feels the Earth and the Moon at once, and the familiar ellipses of
    two-body orbits give way to halo orbits, near rectilinear halo orbits and distant retrograde orbits. These
    scenes come from a mission analysis tool I wrote in Python for my undergraduate thesis. Every scene is
    interactive. Press play, drag to rotate, and scroll over an orbit to zoom toward it.</p>
  </section>
  <section class="cards">{cards_html}
  </section>
  <section class="method">
    <h2>How the scenes were computed</h2>
    <div class="method-grid">
      <div><h3>Dynamics</h3><p>The circular restricted three-body problem in the rotating frame, integrated with
      an eighth order Runge-Kutta method at a tolerance of 1e-12. The Jacobi constant is used as the check on
      every propagation.</p></div>
      <div><h3>Periodic orbits</h3><p>Differential correction using the state transition matrix, then
      continuation from one converged orbit to the next to grow a family. Stability and manifolds come from the
      eigenvalues and eigenvectors of the monodromy matrix.</p></div>
      <div><h3>Observation</h3><p>Sun and Moon positions from the JPL DE440 ephemeris, Earth rotation from the
      IAU 2006 sidereal time, and a set of access constraints that decide when a ground telescope can take a
      measurement.</p></div>
    </div>
    <p class="fine">The pages are precomputed and static, so the inputs cannot be changed here. The full tool
    runs locally with an editable scenario, parameter sweeps, orbit determination filters and a GMAT export.</p>
  </section>
</main>
<footer class="footer">{html.escape(AUTHOR)}. {html.escape(AUTHOR_LINE)}.</footer>
"""
    return page_shell("Cislunar | Earth-Moon orbits, interactive",
                      "Interactive scenes of halo orbits, NRHOs, manifolds and distant retrograde orbits in the "
                      "Earth-Moon three-body problem.", body)


def build_site(directory=SITE_DIRECTORY):
    """Run every scene and write the site.  Returns the list of files written."""
    os.makedirs(os.path.join(directory, "assets"), exist_ok=True)
    written = []
    for source in ASSET_SOURCES:
        target = os.path.join(directory, "assets", os.path.basename(source))
        shutil.copyfile(source, target)
        written.append(target)

    specs = [make_scene() for make_scene in SCENES]
    cards = []
    for k, spec in enumerate(specs):
        print(f"  running {spec['slug']} ...")
        results = runner.run_scenario(spec["scenario"], FAMILIES, ephemeris=EPHEMERIS)
        neighbours = (specs[k - 1], specs[(k + 1) % len(specs)])
        path = os.path.join(directory, f"{spec['slug']}.html")
        with open(path, "w") as handle:
            handle.write(scene_page(spec, results, neighbours))
        written.append(path)

        trajectories, manifolds, _, bodies, _ = displayed_frame(results, spec["views"][0][0])
        cards.append((spec, thumbnail_svg(trajectories, manifolds, bodies, BODY_RADII)))

    path = os.path.join(directory, "index.html")
    with open(path, "w") as handle:
        handle.write(index_page(cards))
    written.append(path)
    return written


if __name__ == "__main__":
    print("Building the static showcase")
    for path in build_site():
        print(f"  wrote {os.path.relpath(path, REPOSITORY_ROOT)}  ({os.path.getsize(path) / 1024.0:,.0f} kB)")
