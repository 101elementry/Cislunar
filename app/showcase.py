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

from engine import crtbp, frames, interplanetary
from model import runner
from model.interplanetary import transfer_scene
from model.ephemeris import load_ephemeris
from model.family import load_families, resonance, resonance_label, RESONANCE_TOLERANCE
from model.scenario import (Scenario, Spacecraft, GroundStation, OpticalSensor, ELEMENT_PRESETS,
                            rendezvous_example)
from app import figures
from app.scene import (BODY_RADII, displayed_frame, frame_label, scene_figure,
                       series_panels_and_thresholds)

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


def resonance_text(orbit):
    """
    The synodic resonance of a family member as a phrase for a facts
    table: N revolutions in M synodic months, which with the libration
    point is how a mission names one of these orbits (Gateway flies the
    9:2 southern L2 NRHO).  A member that only stands near a ratio says
    how far off it is, and one that stands near none says so.
    """
    found = resonance(orbit)
    if found is None:
        return "no low-order synodic resonance"
    revolutions, months, error = found
    if abs(error) <= RESONANCE_TOLERANCE:
        return f"{revolutions}:{months} synodic resonance"
    return f"near {revolutions}:{months} ({abs(error) * 100.0:.1f} % off)"


# Member 49 is the Gateway-like orbit every mission scene flies.  Naming
# it the same way in every facts table saves a reader from having to work
# out which orbit a scene is about.
GATEWAY_MEMBER = 49


def gateway_orbit_name():
    """One line naming the orbit the mission scenes use."""
    orbit = FAMILIES["L2 southern halo"][GATEWAY_MEMBER]
    return (f"L2 southern halo member {GATEWAY_MEMBER}, {resonance_text(orbit)}, "
            f"{crtbp.time_to_days(orbit['period']):.2f} d period, "
            f"perilune {crtbp.length_to_km(orbit['perilune_radius']):,.0f} km")


def scene_halo_to_nrho():
    family = FAMILIES["L2 southern halo"]
    members = [0, 10, 20, 30, 36, 40, 49, 60, 68]
    # The members the facts table names: the two ends, and the three that
    # stand closest to the resonances missions quote.
    members_shown = [0, 20, 36, 49, 68]
    scenario = Scenario(name="From halo to NRHO", epoch_utc="2026-01-01T00:00:00",
                        duration_days=15.0, time_step_s=300.0)
    for index in members:
        named = resonance_label(family[index])
        label = f"Member {index}" + (f", the {named}" if named else "")
        scenario.add(Spacecraft(name=label, source="family", family_name="L2 southern halo",
                                family_index=index, propagation="periodic"))

    def facts(results):
        rows = [("Family", f"L2 southern halo, {len(family)} members, mu = {crtbp.MU}")]
        for index in members_shown:
            period, perilune, stability = orbit_facts(family[index])
            rows.append((f"Member {index}", f"{resonance_text(family[index])}, {period}, perilune {perilune}, "
                                            f"stability index {stability}"))
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
                "That end of the family is the near rectilinear halo orbits.",
                "An NRHO is named by two things: the libration point and branch it belongs to, and its synodic "
                "resonance. Every orbit on this page is an L2 southern halo, southern because apolune stands "
                "over the lunar south pole. The resonance is the ratio N:M of revolutions of the orbit to "
                "synodic months, the month being the time between two alignments of the Sun with the "
                "Earth-Moon line. Member 49 turns nine times in two synodic months, so it is the 9:2, the "
                "orbit Gateway and CAPSTONE fly; member 36 is the 4:1. The table lists the ratio of every "
                "member it names, and how far that member sits from the exact ratio.",
                "The resonance is a resonance with the Sun, and it is what fixes where the eclipses fall. "
                "Nothing on this page computes it from the dynamics, because the circular restricted "
                "three-body problem has no Sun in it: the ratio is read off the period afterwards. Holding a "
                "resonant orbit for years is a question for a model that carries solar gravity and a real "
                "lunar gravity field, which is the next step for this tool rather than something it does "
                "now."],
            "look_for": [
                "The spacecraft on the small orbits race past the Moon and linger over the south pole.",
                "The stability index falls from several hundred on the first halo to about one on the NRHOs. "
                "A value of one means a small error no longer grows from lap to lap.",
                "Rotate the view edge on. The NRHOs look almost like straight lines, which is where the name "
                "comes from.",
                "The 9:2 is not the last member. The family runs on past it to a perilune under 2,000 km, "
                "close to the 5:1, where the orbit is even more nearly rectilinear."],
            "facts": facts}


def scene_manifolds():
    family = FAMILIES["L2 southern halo"]
    index = 8
    flight_days = 22.0
    # Long enough for every branch to depart and then fly its full length,
    # so the clock covers the whole of the tubes rather than stopping part
    # way through them.
    period_days = float(crtbp.time_to_days(family[index]["period"]))
    scenario = Scenario(name="Invariant manifolds", epoch_utc="2026-01-01T00:00:00",
                        duration_days=period_days + flight_days, time_step_s=300.0)
    scenario.add(Spacecraft(name="L2 halo", source="family", family_name="L2 southern halo", family_index=index,
                            propagation="periodic", manifolds="both", manifold_branches=40,
                            manifold_time_days=flight_days))

    def facts(results):
        period, perilune, stability = orbit_facts(family[index])
        return [("Orbit", f"L2 southern halo, member {index}"), ("Period", period),
                ("Synodic ratio", resonance_text(family[index])),
                ("Stability index", stability),
                ("Branches", "40 departure points around the orbit, two directions off each: "
                              f"80 paths leaving and 80 arriving, each flown for {flight_days:.0f} days")]

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
                "routes between the Earth, the Moon and the Lagrange points.",
                "Every branch here is a trajectory, not a decoration, so the clock runs along it. Press play "
                "and each red path is drawn only as far as a spacecraft that left the orbit at that point has "
                "flown; each green path is drawn from where an arriving spacecraft set out, up to where it has "
                "got to, and it touches down on the orbit at the moment the marker passes the arrival point. "
                "The tubes are the union of a hundred and sixty such trajectories.",
                "This member is a large, strongly unstable halo, not an NRHO. It is the useful case for "
                "manifolds: the stability index of the 9:2 NRHO is about 1.3, and an orbit that barely "
                "diverges has manifolds so weak that riding one takes months."],
            "look_for": [
                "One half of each tube heads toward the Moon and the other half escapes to the far side.",
                "Watch a single green path arrive. It reaches the orbit exactly as the white marker passes "
                "the point it is aimed at, because that is the point it was integrated backwards from.",
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
        return [("Orbit observed", gateway_orbit_name()),
                ("Access windows in 30 days", f"{len(windows)}"),
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
                "determination filter has to work with these sparse windows and nothing else, which is the "
                "problem the estimation part of this work addresses."],
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


def scene_proximity():
    scenario = rendezvous_example()
    pair = ("Chaser camera", "Target")

    def facts(results):
        series = results["observations"][pair]["geometry"]
        windows = results["windows"][pair]
        hours_in_range = float(np.sum(series.range_km <= 500.0)) * scenario.time_step_s / 3600.0
        return [("Target orbit", gateway_orbit_name()),
                ("Separation at the start", f"{series.range_km[0]:,.0f} km, directly behind the target"),
                ("Separation after 3 days", f"{series.range_km[-1]:,.0f} km"),
                ("Time within camera range", f"{hours_in_range:.1f} h of {scenario.duration_days * 24.0:.0f} h"),
                ("Camera access", f"{100.0 * results['duty_cycle'][pair]:.0f} % of the span, "
                                  f"{len(windows)} window" + ("" if len(windows) == 1 else "s"))]

    return {"slug": "proximity", "kicker": "Relative motion", "title": "Holding station near the NRHO",
            "tagline": "A chaser left 50 km behind a target on the Gateway orbit, with no control at all.",
            "scenario": scenario, "views": [("lvlh:Target", "moon", "none"), ("rotating", "moon", "none")],
            "pair": pair, "access_text": "camera has access",
            "paragraphs": [
                "The left panel is the view rendezvous is flown in. The target sits at the origin and the axes "
                "turn with it: along its direction of travel, across its orbit plane, and radially away from "
                "the Moon. The chaser starts 50 km behind the target with no relative velocity, which in a "
                "circular Earth orbit would keep it there indefinitely.",
                "Near the Moon it does not stay. The two spacecraft are on slightly different three-body "
                "orbits, and this run starts at perilune, 3,200 km from the Moon, where a small difference "
                "in position grows fastest. Within a day the "
                "chaser has drifted beyond the range at which its camera can see the target. The "
                "Clohessy-Wiltshire equations used for rendezvous in low Earth orbit assume a circular "
                "two-body orbit and cannot describe this, so the motion here is propagated with the full "
                "three-body equations for both spacecraft.",
                "The chaser carries a camera. It can observe when the target is sunlit, bright enough, within "
                "range, and not too close to the Sun or the Moon on the sky. A camera measures direction only "
                "and never range, which makes estimating the relative orbit from these images an "
                "angles-only navigation problem."],
            "look_for": [
                "The dotted red sphere is a 10 km keep-out zone around the target.",
                "The access light turns off once the chaser drifts past 500 km, the range limit of its camera.",
                "The right panel shows both spacecraft on the NRHO at the same instant. At that scale they "
                "are indistinguishable, which is why a relative frame is needed."],
            "facts": facts}


def burn_sizes(scenario, vehicle_name):
    """[(day, size in m/s)] of the vehicle's burns, from the scenario's burn list."""
    vehicle = scenario.spacecraft_named(vehicle_name)
    return [(burn["time_days"], float(np.linalg.norm(burn["delta_v_m_s"]))) for burn in vehicle.burns]


def scene_lander():
    scenario = Scenario.load(os.path.join(REPOSITORY_ROOT, "scenarios", "lander_to_nrho.json"))

    def facts(results):
        burns = burn_sizes(scenario, "Lander")
        separation = crtbp.length_to_km(np.linalg.norm(
            results["trajectories"]["Lander"][:, :3] - results["trajectories"]["Gateway"][:, :3], axis=1))
        return [("Parking orbit", "100 km circular, polar"),
                ("Target orbit", gateway_orbit_name()),
                ("Transfer to the first hold point", f"{burns[0][0] * 24.0:.0f} hours"),
                ("Burn to stop at 30 km", f"{burns[0][1]:.0f} m/s"),
                ("Stepped approach, four burns", f"{sum(size for _, size in burns[1:]):.1f} m/s"),
                ("Distance at the end of the run", f"{separation[-1]:.2f} km"),
                ("Whole leg with the departure burn", "about 715 m/s, see output/artemis_profile.csv")]

    return {"slug": "lander-ascent", "kicker": "Crewed missions", "title": "A lander climbs to the NRHO",
            "tagline": "From a 100 km polar lunar orbit to a stepped approach on the Gateway orbit.",
            "scenario": scenario, "views": [("moon_rotating", "moon", "none"), ("lvlh:Gateway", "moon", "none")],
            "pair": None,
            "paragraphs": [
                "This is the ascent leg of a lunar landing mission. The lander has left the surface and waits "
                "in a low polar orbit. It burns once to leave that orbit and coasts for twelve hours. The "
                "scene starts just after that first burn. The descent to the surface is the same transfer "
                "flown in reverse.",
                "It does not fly straight at the station. It arrives at a hold point 30 km behind it and "
                "stops there, as the station sees it. After an hour it moves to a second hold at 12 km, just "
                "outside the 10 km keep-out sphere, and stops again. The last hop closes to 500 metres. Each "
                "stop is a chance to check the vehicle and wave it off, which is how crewed vehicles approach "
                "a station. Between hops the lander drifts freely, and the next hop starts from wherever "
                "that has left it.",
                "The transfer was found in two steps. Lambert's problem about the Moon alone gives a first "
                "guess, with the departure point chosen so that the burn is along the direction of travel. "
                "That guess is then corrected with the full Earth-Moon equations. Without the first guess "
                "the same corrector settles on a transfer five times more expensive. The cheap route arrives "
                "just after the station passes perilune."],
            "look_for": [
                "On the right the view rides with the station. The lander closes from thousands of kilometres, "
                "then the path breaks into short hops with a pause between them.",
                "The red dotted sphere is the 10 km keep-out zone. Only the final hop goes inside it.",
                "The whole stepped approach costs about 14 m/s, small next to the 700 m/s of the transfer."],
            "facts": facts}


def scene_crew():
    scenario = Scenario.load(os.path.join(REPOSITORY_ROOT, "scenarios", "crew_to_nrho.json"))
    pair = ("Sydney 0.5 m telescope", "Crew vehicle")

    def facts(results):
        burns = burn_sizes(scenario, "Crew vehicle")
        windows = results["windows"][pair]
        return [("Parking orbit", "200 km circular Earth orbit"),
                ("Target orbit", gateway_orbit_name()),
                ("Injection burn", "about 3,130 m/s, before the scene starts"),
                ("Flyby burn, 150 km above the Moon", f"{burns[0][1]:.0f} m/s on day {burns[0][0]:.1f}"),
                ("NRHO insertion burn", f"{burns[1][1]:.0f} m/s on day {burns[1][0]:.1f}"),
                ("Direct two-burn transfer, for comparison", "921 m/s insertion"),
                ("Sydney access windows on the vehicle", f"{len(windows)} during the run")]

    return {"slug": "crew-transfer", "kicker": "Crewed missions", "title": "A crew vehicle flies to the NRHO",
            "tagline": "From low Earth orbit past the Moon to the Gateway orbit, tracked from Sydney on the way.",
            "scenario": scenario, "views": [("rotating", "system", "none"), ("earth_inertial", "system", "none")],
            "pair": pair, "access_text": "Sydney sees the vehicle",
            "paragraphs": [
                "A crew vehicle leaves a 200 km Earth orbit with a single burn of about 3.1 km/s, the "
                "trans-lunar injection. The scene starts just after it. Four days later the vehicle passes "
                "150 km above the far side of the Moon and brakes there, where it is moving fastest and a "
                "change of speed is worth most. A day and a half later a small third burn puts it beside "
                "the station on the NRHO.",
                "Flying past the Moon roughly halves the cost of arriving. Going directly to the NRHO with two "
                "burns needs 921 m/s at arrival. With the flyby the two burns after leaving the Earth add up "
                "to about 470 m/s. The flyby burn was found as the smallest burn at closest approach whose "
                "incoming path, followed back in time, came from a 200 km Earth orbit.",
                "The strip below shows when a half metre telescope in Sydney could see the vehicle during "
                "the coast. Tracking an outbound vehicle from the ground with angles alone is the same "
                "estimation problem as tracking the station, with far less time to solve it, and a flyby "
                "burn on the far side of the Moon is a manoeuvre no ground telescope can watch."],
            "look_for": [
                "The access light shows when Sydney can observe the vehicle.",
                "The path bends sharply at the Moon. Most of that bend is the Moon's gravity and the rest is "
                "the braking burn.",
                "After the insertion burn the vehicle and the station move together round the NRHO."],
            "facts": facts}


def thumbnail_top_down(paths, colors, size=(320, 200)):
    """SVG of paths seen from above the x-y plane, with a dot at the origin for the Sun."""
    everything = np.vstack([path[:, :2] for path in paths.values()])
    extent = np.abs(everything).max()
    scale = 0.45 * min(size) / extent
    parts = [f'<svg viewBox="0 0 {size[0]} {size[1]}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">']
    for name, path in paths.items():
        pixels = np.column_stack([size[0] / 2.0 + scale * path[:, 0], size[1] / 2.0 - scale * path[:, 1]])
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pixels[::3])
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="1.4"/>')
    parts.append(f'<circle cx="{size[0] / 2.0}" cy="{size[1] / 2.0}" r="3" fill="{figures.SUN}"/></svg>')
    return "".join(parts)


def scene_mars_transfer():
    departure_utc = "2035-06-28T00:00:00"
    flight_days = 200.0

    def custom():
        departure_jd = frames.julian_date(departure_utc)
        data = transfer_scene(EPHEMERIS, departure_jd, flight_days)
        result = data["transfer"]
        v_depart = float(np.sqrt(result["c3_km2_s2"]))
        v_arrive = float(np.linalg.norm(result["v_infinity_arrive"]))
        leo_burn = interplanetary.departure_burn_from_circular_orbit(v_depart)
        staged_burn = interplanetary.departure_burn_from_lunar_distance(v_depart)
        # The clock starts at the first sample of the grid, margin days before departure.
        first_day = (np.datetime64(departure_utc) - np.timedelta64(int(round(departure_jd - data["jd"][0])), "D"))
        clock = {"epoch_utc": str(first_day), "time_step_s": 86400.0, "n_samples": int(len(data["jd"])),
                 "days_only": True}
        figure = figures.heliocentric_figure(data["paths_au"], index=0, trail_samples=30, clock=clock)
        prepared = {"figures": [figure], "captions": ["Sun-centred, ecliptic"], "n_samples": clock["n_samples"],
                    "time_step_s": 86400.0, "series": None, "windows_s": None,
                    "facts": [("Departure", departure_utc[:10]), ("Flight time", f"{flight_days:.0f} days"),
                              ("Launch energy C3", f"{result['c3_km2_s2']:.2f} km2/s2"),
                              ("Arrival excess speed", f"{v_arrive:.2f} km/s"),
                              ("Departure burn from a 400 km orbit", f"{leo_burn:.2f} km/s"),
                              ("Departure burn staged from lunar distance", f"{staged_burn:.2f} km/s"),
                              ("Capture into a one-sol Mars orbit", f"{interplanetary.capture_burn(v_arrive):.2f} km/s"),
                              ("Lambert end point against Mars", f"{data['arrival_miss_km']:.0f} km")]}
        colors = {"Earth": figures.EARTHSHINE, "Mars": "#d9775a", "Vehicle": "#f2f2f2"}
        return prepared, thumbnail_top_down(data["paths_au"], colors)

    return {"slug": "mars-transfer", "kicker": "Interplanetary", "title": "Earth to Mars, the 2035 window",
            "tagline": "The cheapest transfer of the June 2035 launch window, and what leaving from the NRHO saves.",
            "custom": custom, "speeds": [(120, "5 d / s"), (240, "10 d / s"), (480, "20 d / s")],
            "default_speed": 240,
            "paragraphs": [
                "A launch window to Mars opens every 26 months, when the two planets are placed so that a "
                "transfer orbit leaving the Earth arrives where Mars will be. The vehicle here leaves at the "
                "best date of the 2035 window and coasts for 200 days on an orbit about the Sun. The orbit "
                "comes from Lambert's problem: given two positions and the time between them, find the orbit "
                "that joins them. The planet positions are from the JPL DE440 ephemeris.",
                "The cost at each end depends on where the burn is made. From a 400 km circular Earth orbit "
                "the departure burn is 3.6 km/s. A vehicle that has been assembled and fuelled in the NRHO "
                "instead falls toward the Earth, reaches a low perigee at almost escape speed, and burns "
                "there. The same departure then costs about 0.55 km/s at perigee. Getting from the NRHO to "
                "that perigee takes two more burns, one on the NRHO and one close to the Moon, of about "
                "0.48 km/s together, so the whole departure is about 1.0 km/s against 3.6. The propellant "
                "still has to be lifted to the NRHO, but it can go in separate launches on slow, efficient "
                "routes, which is the argument for staging there."],
            "look_for": [
                "The vehicle leaves ahead of Mars and the two meet. Mars moves more slowly, so the vehicle "
                "aims at where Mars will be.",
                "The transfer orbit is slightly tilted, because the orbit of Mars is inclined 1.85 degrees "
                "to the Earth's.",
                "Miss this window and the next is in August 2037, at a much higher launch energy.",
                "A crew that stays only a month cannot wait for the cheap way home. Passing Venus on the way "
                "out brings such a mission to about 640 days and 5.4 km/s, with an entry the heat shield "
                "can take."],
            "facts": None}


SCENES = [scene_halo_to_nrho, scene_manifolds, scene_dro_two_frames, scene_sydney_tracking,
          scene_proximity, scene_lander, scene_crew, scene_lunar_relay, scene_mars_transfer]


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


TOPBAR = """<header class="topbar">
  <a class="brand" href="index.html">Cislunar</a>
  <nav class="nav"><a href="index.html#scenes">Scenes</a><a href="index.html#method">Method</a><a href="index.html#model">Model</a></nav>
</header>"""

FOOTER = (f'<footer class="footer"><span>{html.escape(AUTHOR)}. {html.escape(AUTHOR_LINE)}.</span>'
          '<span>Dynamics: circular restricted three-body problem, Earth and Moon as point masses '
          '(<a href="index.html#model">assumptions</a>). Sun and Moon for observation geometry from '
          'JPL DE440.</span></footer>')

PAGE_SCRIPTS = (f'<script src="{PLOTLY_SCRIPT}"></script>\n<script src="assets/playback.js"></script>\n'
                '<script src="assets/zoom_to_cursor.js"></script>\n<script src="assets/showcase.js"></script>')


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
<!-- Vercel Web Analytics: page views by country and device, no cookies.
     Only counts once Analytics is switched on for the project. -->
<script defer src="/_vercel/insights/script.js"></script>
</head>
<body>
{body}
{scripts}
</body>
</html>
"""


DEFAULT_SPEEDS = [(2, "2 h / s"), (6, "6 h / s"), (12, "12 h / s"), (24, "1 d / s"), (72, "3 d / s")]


def prepare_scenario_scene(spec, results):
    """
    What a page needs from a scenario run: the figures of its views,
    their captions, the clock, the optional time series and windows,
    and the table of values.
    """
    scenario = spec["scenario"]
    scene_figures = [scene_figure(scenario, results, frame, view, focus, 0) for frame, view, focus in spec["views"]]
    prepared = {"figures": scene_figures, "captions": [frame_label(frame) for frame, _, _ in spec["views"]],
                "n_samples": int(len(results["times_s"])), "time_step_s": float(scenario.time_step_s),
                "series": None, "windows_s": None, "facts": spec["facts"](results)}
    if spec["pair"] is not None:
        host, sensor = runner.observer_settings(scenario, spec["pair"][0])
        panels, thresholds = series_panels_and_thresholds(host, sensor)
        prepared["series"] = figures.time_series_figure(results["observations"][spec["pair"]]["geometry"], thresholds,
                                                        results["windows"][spec["pair"]], panels=panels)
        prepared["windows_s"] = [[float(start), float(stop)] for start, stop in results["windows"][spec["pair"]]]
    return prepared


def scene_page(spec, prepared, neighbours, number):
    """HTML of one scene page; number is its place in the list, from 1."""
    scene_figures = prepared["figures"]
    for figure in scene_figures:
        # The current-time markers need no legend entry of their own here;
        # with eight spacecraft they would fill the top of the scene.
        figure.for_each_trace(lambda trace: trace.update(showlegend=False),
                              selector=lambda trace: (trace.meta or {}).get("role") == "marker")
    page_data = {"figures": [json.loads(figure.to_json()) for figure in scene_figures],
                 "n_samples": prepared["n_samples"], "time_step_s": prepared["time_step_s"],
                 "windows_s": prepared["windows_s"], "series": None}
    series_block = ""
    access_light = ""
    if prepared["series"] is not None:
        prepared["series"].update_layout(height=None, autosize=True)
        page_data["series"] = json.loads(prepared["series"].to_json())
        series_block = '<div class="series"><div id="series" class="plot"></div></div>'
        page_data["access_text"] = spec.get("access_text", "Sydney has access")
        access_light = '<span id="access-light" class="access-light">no access</span>'
    speeds = spec.get("speeds", DEFAULT_SPEEDS)
    default_speed = spec.get("default_speed", 12)
    speed_options = "".join(f'<option value="{value}"{" selected" if value == default_speed else ""}>'
                            f'{html.escape(label)}</option>' for value, label in speeds)

    stage_class = "stage split" if len(scene_figures) == 2 else "stage"
    holders = ["view-3d", "view-3d-b"]
    views_html = ""
    for k, caption in enumerate(prepared["captions"]):
        views_html += (f'<figure class="view"><figcaption>{html.escape(caption)}</figcaption>'
                       f'<div id="{holders[k]}" class="holder"><div class="plot"></div></div></figure>')

    paragraphs = "".join(f"<p>{html.escape(text)}</p>" for text in spec["paragraphs"])
    look_for = "".join(f"<li><span>{html.escape(text)}</span></li>" for text in spec["look_for"])
    facts = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
                    for label, value in prepared["facts"])
    previous_spec, next_spec = neighbours
    body = f"""
{TOPBAR}
<main>
  <section class="scene-head">
    <div>
      <p class="label">{number:02d} / {html.escape(spec['kicker'])}</p>
      <h1>{html.escape(spec['title'])}</h1>
    </div>
    <p class="tagline">{html.escape(spec['tagline'])}</p>
  </section>
  <section class="{stage_class}">{views_html}</section>
  <section class="transport">
    <button id="play-button" class="play" type="button">Play</button>
    <label class="speed"><span class="label">Rate</span>
      <select id="play-speed">{speed_options}
      </select>
    </label>
    <input id="scrubber" type="range" min="0" max="{page_data['n_samples'] - 1}" value="0" step="1"
           aria-label="Time">
    <span id="time-readout" class="readout"></span>
    {access_light}
  </section>
  {series_block}
  <section class="notes">
    <div class="prose"><p class="label">Overview</p>{paragraphs}</div>
    <div class="side">
      <p class="label">What to look for</p><ol class="look-for">{look_for}</ol>
      <p class="label">Values from this run</p><table class="facts">{facts}</table>
      <p class="hint">Drag to rotate. Click a scene, then scroll over an orbit to zoom toward it. Click a legend entry to hide it.</p>
    </div>
  </section>
  <nav class="pager">
    <a href="{previous_spec['slug']}.html"><span class="label">Previous</span>
      <span class="pager-title">{html.escape(previous_spec['title'])}</span></a>
    <a href="{next_spec['slug']}.html"><span class="label">Next</span>
      <span class="pager-title">{html.escape(next_spec['title'])}</span></a>
  </nav>
</main>
{FOOTER}
<script type="application/json" id="page-data">{embedded_json(page_data)}</script>
"""
    return page_shell(f"{spec['title']} | Cislunar", spec["tagline"], body, scripts=PAGE_SCRIPTS)


def index_page(cards, hero_data):
    """
    HTML of the landing page.  cards is a list of (spec, thumbnail svg);
    hero_data is the page data of the scene that plays behind the title.
    """
    rows_html = ""
    for k, (spec, thumbnail) in enumerate(cards):
        rows_html += f"""
    <a class="scene-row" href="{spec['slug']}.html">
      <span class="scene-number">{k + 1:02d}</span>
      <span><span class="label">{html.escape(spec['kicker'])}</span>
        <span class="scene-title" style="display:block">{html.escape(spec['title'])}</span></span>
      <p class="scene-tagline">{html.escape(spec['tagline'])}</p>
      <span class="thumb">{thumbnail}</span>
    </a>"""
    body = f"""
{TOPBAR}
<main>
  <section class="hero">
    <div id="view-3d" class="holder"><div class="plot"></div></div>
    <div class="hero-text">
      <p class="label">Earth-Moon three-body problem</p>
      <h1>Cislunar trajectory analysis</h1>
      <p>Periodic orbits, natural transport and optical tracking near the Moon, computed from first principles.
      Undergraduate thesis software, University of Sydney.</p>
    </div>
    <div class="hero-clock"><span class="label">L2 southern halo family, rotating frame</span>
      <span id="time-readout" class="readout"></span></div>
  </section>
  <section class="section" id="scenes">
    <div class="section-head"><p class="label">Scenes</p><p class="label">{len(cards):02d} interactive</p></div>
    <div class="scene-list">{rows_html}
    </div>
  </section>
  <section class="section" id="method">
    <div class="section-head"><p class="label">Method</p></div>
    <div class="method-grid">
      <div><p class="label">01</p><h3>Dynamics</h3><p>The circular restricted three-body problem in the rotating
      frame, integrated with an eighth order Runge-Kutta method at a tolerance of 1e-12. The Jacobi constant
      checks every propagation.</p></div>
      <div><p class="label">02</p><h3>Periodic orbits</h3><p>Differential correction with the state transition
      matrix, then continuation from one converged orbit to the next. Stability and manifolds come from the
      eigenvalues and eigenvectors of the monodromy matrix.</p></div>
      <div><p class="label">03</p><h3>Observation</h3><p>Sun and Moon from the JPL DE440 ephemeris, Earth rotation
      from IAU 2006 sidereal time, and access constraints that decide when a telescope can take a
      measurement.</p></div>
    </div>
    <p class="fine">These pages are precomputed, so the inputs are fixed. The full tool runs locally with an
    editable scenario, parameter sweeps, orbit determination filters and a GMAT export.</p>
  </section>
  <section class="section" id="model">
    <div class="section-head"><p class="label">Model</p><p class="label">what it carries and what it does not</p></div>
    <div class="assumption-grid">
      <div><p class="label">In the equations of motion</p>
      <ul class="assumptions">
        <li>The circular restricted three-body problem, Earth-Moon, mass ratio mu = 0.01215058560962404,
        with LU = 384,400 km and TU = 375,190.26 s.</li>
        <li>Earth and Moon as <strong>point masses</strong> on a common circular orbit, in a frame that rotates
        with them at a constant rate. The spacecraft has no mass.</li>
        <li>Nothing else. Every trajectory on this site, including the two-body elements scene and both
        mission legs, is integrated with these equations and no others.</li>
        <li>Integration with DOP853 at a relative and absolute tolerance of 1e-12; the Jacobi constant is
        checked afterwards and holds to about 1e-11 over ten time units.</li>
      </ul></div>
      <div><p class="label">Not in them</p>
      <ul class="assumptions">
        <li>No lunar gravity field beyond the point mass: no J2, no GRAIL spherical harmonics, no mascons.
        Perilune here is a few thousand kilometres, where the higher harmonics are small, but they are not
        zero and they are not modelled.</li>
        <li>No solar gravity, no solar radiation pressure, no Earth oblateness, no lunar librations.</li>
        <li>No eccentricity or inclination in the Moon's orbit. The real orbit has e = 0.055 and is tilted
        about 5 degrees to the ecliptic; the model's is a circle.</li>
        <li>No ephemeris dynamics. JPL DE440 is used only to place the Sun, Moon, Earth and the observing
        site for the observation geometry, never in the equations of motion.</li>
      </ul></div>
    </div>
    <p class="fine">What that costs. A CRTBP periodic orbit repeats exactly; a real NRHO does not, because
    solar gravity and the Moon's true orbit change its period and perilune from one revolution to the next,
    which is also why a real one needs a station keeping burn every revolution. The synodic resonance that
    names these orbits, the 9:2 and the 4:1, is a resonance with the Sun, so it is a label computed from the
    period here rather than anything the dynamics enforce. And in the estimation work the simulated truth and
    the filter share these same equations, so the filter meets no dynamic mismodelling: the errors and the
    smallest detectable manoeuvres that come out of it are a floor, and moving the whole ladder onto an
    ephemeris model is the step that would test it.</p>
  </section>
</main>
{FOOTER}
<script type="application/json" id="page-data">{embedded_json(hero_data)}</script>
"""
    return page_shell("Cislunar | Earth-Moon trajectory analysis",
                      "Interactive scenes of halo orbits, NRHOs, manifolds and distant retrograde orbits in the "
                      "Earth-Moon three-body problem.", body, scripts=PAGE_SCRIPTS)


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
        if "custom" in spec:
            prepared, thumbnail = spec["custom"]()
        else:
            results = runner.run_scenario(spec["scenario"], FAMILIES, ephemeris=EPHEMERIS)
            prepared = prepare_scenario_scene(spec, results)
            thumbnail_frame = spec["views"][0][0]
            if thumbnail_frame.startswith("lvlh:"):
                thumbnail_frame = "rotating"
            trajectories, manifolds, _, bodies, _ = displayed_frame(results, thumbnail_frame)
            thumbnail = thumbnail_svg(trajectories, manifolds, bodies, BODY_RADII)
        neighbours = (specs[k - 1], specs[(k + 1) % len(specs)])
        path = os.path.join(directory, f"{spec['slug']}.html")
        with open(path, "w") as handle:
            handle.write(scene_page(spec, prepared, neighbours, k + 1))
        written.append(path)
        cards.append((spec, thumbnail))

    hero_spec = specs[0]
    hero_results = runner.run_scenario(hero_spec["scenario"], FAMILIES, ephemeris=EPHEMERIS)
    frame, view, focus = hero_spec["views"][0]
    hero_figure = scene_figure(hero_spec["scenario"], hero_results, frame, view, focus, 0)
    hero_figure.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    hero_data = {"figures": [json.loads(hero_figure.to_json())], "n_samples": int(len(hero_results["times_s"])),
                 "time_step_s": float(hero_spec["scenario"].time_step_s), "windows_s": None, "series": None}
    path = os.path.join(directory, "index.html")
    with open(path, "w") as handle:
        handle.write(index_page(cards, hero_data))
    written.append(path)
    return written


if __name__ == "__main__":
    print("Building the static showcase")
    for path in build_site():
        print(f"  wrote {os.path.relpath(path, REPOSITORY_ROOT)}  ({os.path.getsize(path) / 1024.0:,.0f} kB)")
