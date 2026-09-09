"""
Dash interface for the cislunar mission tool.  Run from the repository
root:  python -m app.main   then open http://127.0.0.1:8050

Layout
  top     scenario settings, run / save / load, view selector
  left    scenario tree, add and remove objects, property form
  centre  3D rotating-frame view
  right   access windows for one observer-spacecraft pair
  bottom  elevation, magnitude and lunar separation against time,
          and a time slider that moves a marker in the 3D view

Callbacks only move data between the scenario store, model.runner and
the figure builders.  Nothing here computes physics or geometry.
"""

import base64
import json
import time
import uuid
from datetime import datetime, timedelta

import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output, State, ALL, ctx, no_update

from engine import crtbp, frames, propagation
from model import orbits, runner, sweep
from model.family import load_families, nearest_member, member_label, DEFAULT_FAMILY_NAME
from model.scenario import (Scenario, Spacecraft, GroundStation, OpticalSensor, example_scenario,
                            ELEMENT_PRESETS)
from app import figures

dash_app = Dash(__name__, title="Cislunar mission tool", suppress_callback_exceptions=True)

# Results of the last run, kept in memory on the server.  The app runs
# locally for one user, so a module-level dictionary is enough; the
# browser only holds the run id.
RESULTS = {}
FAMILIES = load_families()
FAMILY_NAMES = list(FAMILIES.keys())
FAMILY_LABELS = {name: [member_label(index, orbit) for index, orbit in enumerate(family)]
                 for name, family in FAMILIES.items()}
FIXED_POINTS = propagation.fixed_points()

TABLE_HEADER = {"backgroundColor": "#0e121b", "color": "#66718a", "fontWeight": "600", "border": "1px solid #232b3a"}
TABLE_DATA = {"backgroundColor": "#121722", "color": "#e7eaf0", "border": "1px solid #232b3a"}

TREE_GLYPH = {"spacecraft": ("◆", "glyph-spacecraft"),
              "ground_station": ("▲", "glyph-station"),
              "optical_sensor": ("●", "glyph-sensor")}


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

def setting(label, component):
    return html.Div([html.Label(label), component], className="setting")


def field(label, component, hint=None):
    children = [html.Label(label), component]
    if hint:
        children.append(html.Span(hint, className="hint"))
    return html.Div(children, className="field")


def panel(title, body, header_extra=None, body_class="panel-body", panel_id=None):
    header_children = [html.Span(title, className="panel-title")]
    if header_extra is not None:
        header_children.append(header_extra)
    children = [html.Div(header_children, className="panel-header"), html.Div(body, className=body_class)]
    if panel_id is None:
        return html.Div(children, className="panel")
    return html.Div(children, className="panel", id=panel_id)


def toggle_button(label, button_id):
    return html.Button(label, id=button_id, className="toggle active", n_clicks=0)


initial_scenario = example_scenario()

dash_app.layout = html.Div([
    dcc.Store(id="scenario-store", data=initial_scenario.to_dict()),
    dcc.Store(id="selected-store", data=None),
    dcc.Store(id="results-store", data=None),
    dcc.Store(id="layout-store", data={"left": True, "right": True, "bottom": True}),
    dcc.Store(id="split-store", data=False),
    dcc.Interval(id="play-interval", interval=300, disabled=True),
    dcc.Download(id="download"),

    # ---- top bar ----
    html.Div([
        html.Div([html.Span("Cislunar mission tool", className="brand-title"),
                  html.Span("Earth-Moon CRTBP · mu = 0.012151", className="brand-subtitle")],
                 className="brand"),
        html.Div([
            setting("Scenario", dcc.Input(id="scenario-name", type="text", className="wide", debounce=True,
                                          value=initial_scenario.name)),
            setting("Epoch (UTC)", dcc.Input(id="scenario-epoch", type="text", className="wide mono", debounce=True,
                                             value=initial_scenario.epoch_utc)),
            setting("Duration [days]", dcc.Input(id="scenario-duration", type="number", className="narrow",
                                                 debounce=True, value=initial_scenario.duration_days, min=0.1)),
            setting("Step [s]", dcc.Input(id="scenario-step", type="number", className="narrow", debounce=True,
                                          value=initial_scenario.time_step_s, min=1)),
        ], className="settings"),
        html.Div([
            html.Span("Panels", className="panel-title"),
            toggle_button("Tree", "toggle-left"),
            toggle_button("Windows", "toggle-right"),
            toggle_button("Series", "toggle-bottom"),
        ], className="toggles"),
        html.Div([
            html.Button("Run analysis", id="run-button", className="primary", n_clicks=0),
            html.Button("Save", id="save-button", n_clicks=0),
            dcc.Upload(html.Div("Load", className="upload-box"), id="load-upload", multiple=False),
        ], className="actions"),
    ], className="topbar"),

    # ---- main row ----
    html.Div([
        panel("Scenario tree", [
            html.Div(id="tree", className="tree"),
            html.Div([
                dcc.Dropdown(id="add-type", className="dash-dropdown", clearable=False, value="spacecraft",
                             options=[{"label": "Spacecraft", "value": "spacecraft"},
                                      {"label": "Ground station", "value": "ground_station"},
                                      {"label": "Optical sensor", "value": "optical_sensor"}]),
                html.Button("Add", id="add-button", className="small", n_clicks=0),
                html.Button("Remove", id="remove-button", className="small danger", n_clicks=0),
            ], className="add-row"),
            html.Details([
                html.Summary("Add several family members"),
                html.Div([html.Label("Family"),
                          dcc.Dropdown(id="range-family", className="dash-dropdown", clearable=False,
                                       value=DEFAULT_FAMILY_NAME,
                                       options=[{"label": f"{name} ({len(FAMILIES[name])} members)", "value": name}
                                                for name in FAMILY_NAMES])], className="field"),
                html.Div([
                    html.Div([html.Label("From"), dcc.Input(id="range-from", type="number", value=0, min=0, step=1)],
                             className="field"),
                    html.Div([html.Label("To"), dcc.Input(id="range-to", type="number", value=None, min=0, step=1)],
                             className="field"),
                    html.Div([html.Label("Every"), dcc.Input(id="range-step", type="number", value=10, min=1, step=1)],
                             className="field"),
                ], className="form-row three"),
                html.Button("Add members", id="add-range-button", className="small", n_clicks=0),
                html.Span("Leave To empty for the whole family. In the halo families index 0 is the largest "
                          "orbit and the last index has the lowest perilune.", className="hint"),
            ], className="details"),
            html.Details([
                html.Summary("How to set up a simulation"),
                html.Ol([
                    html.Li("Set the epoch, duration and step at the top."),
                    html.Li("Add spacecraft: an orbit family member (periodic, station-kept), your own "
                            "rotating-frame initial state, or two-body elements about the Moon or Earth "
                            "(lunar relay, GEO, LEO)."),
                    html.Li("A typed state can be corrected to a periodic orbit from its form; a family "
                            "member can be picked by perilune radius or period."),
                    html.Li("Add a ground station and, optionally, an optical sensor on it."),
                    html.Li("Select an object, edit its fields, press Apply."),
                    html.Li("Press Run analysis. Pick an observer-spacecraft pair on the right."),
                    html.Li("Save JSON to keep the scenario; scripts/ shows how to sweep it without the GUI."),
                    html.Li("Press Play on the timeline to run the clock; the markers move and each spacecraft "
                            "trails a fading tail. Split shows two frames side by side at the same instant."),
                    html.Li("In the scene, drag to turn and scroll to zoom. Scrolling over an orbit, a body or "
                            "a marker zooms toward that point; over empty space it zooms toward the centre. "
                            "The Focus menu recentres on the Moon, Earth, a libration point or a spacecraft."),
                ], className="help"),
            ], className="details"),
            html.Div(className="divider"),
            html.Div(id="form", className="form"),
            html.Div([html.Button("Apply", id="apply-button", className="primary small", n_clicks=0, hidden=True)],
                     className="form-actions"),
            html.Div(id="form-status", className="form-status"),
        ], panel_id="left-panel"),
        panel("Scene", html.Div([
                  dcc.Graph(id="view-3d", style={"height": "100%"}, responsive=True, config={"displaylogo": False}),
                  dcc.Graph(id="view-3d-b", style={"height": "100%"}, responsive=True, config={"displaylogo": False},
                            className="scene-b"),
              ], id="scene-grid", className="scene-grid"),
              header_extra=html.Div([
                  html.Span(id="run-status", className="status"),
                  html.Button("Split", id="split-button", className="toggle", n_clicks=0),
                  dcc.Dropdown(id="frame-select-b", className="dash-dropdown medium scene-b", clearable=False,
                               value="moon_inertial",
                               options=[{"label": "Rotating, barycentric", "value": "rotating"},
                                        {"label": "Rotating, Moon-centred", "value": "moon_rotating"},
                                        {"label": "Inertial, Moon-centred", "value": "moon_inertial"},
                                        {"label": "Inertial, Earth-centred", "value": "earth_inertial"},
                                        {"label": "Inertial, barycentric", "value": "inertial"}]),
                  dcc.Dropdown(id="view-select", className="dash-dropdown narrow", clearable=False,
                               value="moon", options=[{"label": "Moon region", "value": "moon"},
                                                      {"label": "Whole system", "value": "system"}]),
                  dcc.Dropdown(id="frame-select", className="dash-dropdown medium", clearable=False,
                               value="rotating",
                               options=[{"label": "Rotating, barycentric", "value": "rotating"},
                                        {"label": "Rotating, Moon-centred", "value": "moon_rotating"},
                                        {"label": "Inertial, Moon-centred", "value": "moon_inertial"},
                                        {"label": "Inertial, Earth-centred", "value": "earth_inertial"},
                                        {"label": "Inertial, barycentric", "value": "inertial"}]),
                  dcc.Dropdown(id="focus-select", className="dash-dropdown narrow", clearable=False, value="none",
                               options=[{"label": "Focus: free", "value": "none"}]),
              ], className="header-controls", id="scene-header-controls"),
              body_class="panel-body flush"),
        panel("Access windows", [
            dcc.Dropdown(id="pair-select", className="dash-dropdown", clearable=False, placeholder="observer → spacecraft"),
            html.Details([
                html.Summary("What this shows"),
                html.P("An access window is a run of time steps in which the chosen observer can see the "
                       "chosen spacecraft: every constraint passes at once. The constraints come from the "
                       "station (elevation cutoff, darkness) and the sensor (limiting magnitude, lunar "
                       "exclusion), plus the spacecraft being sunlit. Each chip gives the fraction of the "
                       "span that one constraint alone would allow, so you can see which one is cutting. "
                       "Duty cycle is the fraction of the whole span inside windows.", className="help-text"),
            ], className="details"),
            html.Div(id="summary"),
            dash_table.DataTable(id="windows-table",
                                 columns=[{"name": "#", "id": "index"},
                                          {"name": "Start (UTC)", "id": "start"},
                                          {"name": "Stop (UTC)", "id": "stop"},
                                          {"name": "Duration [h]", "id": "duration"}],
                                 data=[], page_size=12, style_as_list_view=True,
                                 style_table={"overflowX": "auto"},
                                 style_header=TABLE_HEADER, style_data=TABLE_DATA,
                                 style_cell={"padding": "5px 8px", "textAlign": "left", "whiteSpace": "nowrap"},
                                 style_cell_conditional=[{"if": {"column_id": "index"}, "width": "30px"},
                                                         {"if": {"column_id": "duration"}, "textAlign": "right"}]),
            html.Details([
                html.Summary("Parameter sweep"),
                html.P("Run the scenario once per value of one setting and compare the access statistics. "
                       "The same rows come from model.sweep in a script.", className="help-text"),
                html.Div([field("Object", dcc.Dropdown(id="sweep-object", className="dash-dropdown", clearable=False)),
                          field("Setting", dcc.Dropdown(id="sweep-attribute", className="dash-dropdown", clearable=False))],
                         className="form-row"),
                html.Div([field("From", dcc.Input(id="sweep-from", type="number", value=0)),
                          field("To", dcc.Input(id="sweep-to", type="number", value=60)),
                          field("Steps", dcc.Input(id="sweep-steps", type="number", value=7, min=2, max=200, step=1))],
                         className="form-row three"),
                html.Div([html.Button("Run sweep", id="sweep-button", className="small primary", n_clicks=0),
                          html.Button("Download CSV", id="sweep-download-button", className="small", n_clicks=0),
                          html.Span(id="sweep-status", className="status")], className="form-actions"),
                dash_table.DataTable(id="sweep-table", data=[], columns=[], page_size=10, style_as_list_view=True,
                                     style_table={"overflowX": "auto"}, sort_action="native",
                                     style_header=TABLE_HEADER, style_data=TABLE_DATA,
                                     style_cell={"padding": "4px 8px", "textAlign": "right", "whiteSpace": "nowrap"}),
                dcc.Download(id="sweep-download"),
            ], className="details"),
        ], panel_id="right-panel"),
    ], className="main", id="main-row"),

    # ---- bottom ----
    html.Div([
        panel("Time series", [
            html.Details([
                html.Summary("What this shows"),
                html.P("Green bands are the access windows. Dashed lines are the thresholds: the spacecraft must "
                       "be above the elevation line, brighter (lower on the reversed magnitude axis) than the "
                       "limiting magnitude, and farther from the Moon than the exclusion angle. Darkness and "
                       "shadow are not drawn but still apply. The pale line is the slider time.",
                       className="help-text"),
            ], className="details panel-note"),
            dcc.Graph(id="time-series", figure=figures.empty_time_series_figure(), responsive=True,
                      style={"height": "210px"}, config={"displaylogo": False}),
            html.Div([
                html.Span("Time", className="panel-title"),
                html.Button("Play", id="play-button", className="small play", n_clicks=0),
                dcc.Dropdown(id="play-speed", className="dash-dropdown speed", clearable=False, value=5,
                             options=[{"label": "1 step / tick", "value": 1}, {"label": "5 steps / tick", "value": 5},
                                      {"label": "20 steps / tick", "value": 20}, {"label": "60 steps / tick", "value": 60},
                                      {"label": "240 steps / tick", "value": 240}]),
                html.Div(dcc.Slider(id="time-slider", min=0, max=1, step=1, value=0, marks={},
                                    tooltip={"placement": "top", "always_visible": False}),
                         className="slider"),
                html.Span(id="time-readout", className="time-readout"),
            ], className="slider-row"),
        ], body_class="panel-body flush", panel_id="bottom-panel"),
    ], className="bottom"),
], className="app")


# --------------------------------------------------------------------------
# Panel visibility
# --------------------------------------------------------------------------

@dash_app.callback(Output("layout-store", "data"),
                   Input("toggle-left", "n_clicks"), Input("toggle-right", "n_clicks"),
                   Input("toggle-bottom", "n_clicks"), State("layout-store", "data"),
                   prevent_initial_call=True)
def toggle_panels(left_clicks, right_clicks, bottom_clicks, layout):
    key = {"toggle-left": "left", "toggle-right": "right", "toggle-bottom": "bottom"}[ctx.triggered_id]
    layout = dict(layout)
    layout[key] = not layout[key]
    return layout


@dash_app.callback(Output("main-row", "className"), Output("left-panel", "hidden"),
                   Output("right-panel", "hidden"), Output("bottom-panel", "hidden"),
                   Output("toggle-left", "className"), Output("toggle-right", "className"),
                   Output("toggle-bottom", "className"),
                   Input("layout-store", "data"))
def apply_panel_layout(layout):
    main_class = "main"
    if not layout["left"]:
        main_class = main_class + " hide-left"
    if not layout["right"]:
        main_class = main_class + " hide-right"

    def button_class(shown):
        return "toggle active" if shown else "toggle"

    return (main_class, not layout["left"], not layout["right"], not layout["bottom"],
            button_class(layout["left"]), button_class(layout["right"]), button_class(layout["bottom"]))


# --------------------------------------------------------------------------
# Playback and split view
# --------------------------------------------------------------------------

@dash_app.callback(Output("play-interval", "disabled"), Output("play-button", "children"),
              Output("play-button", "className"),
              Input("play-button", "n_clicks"), State("play-interval", "disabled"), prevent_initial_call=True)
def toggle_play(n_clicks, disabled):
    playing = disabled
    return (not playing), ("Pause" if playing else "Play"), ("small play active" if playing else "small play")


@dash_app.callback(Output("time-slider", "value", allow_duplicate=True),
              Input("play-interval", "n_intervals"), State("time-slider", "value"), State("time-slider", "max"),
              State("play-speed", "value"), prevent_initial_call=True)
def advance_time(n_intervals, value, maximum, speed):
    """Move the clock forward by `speed` samples per tick, wrapping at the end."""
    if maximum is None or maximum <= 0:
        return no_update
    return int((value or 0) + int(speed or 1)) % (int(maximum) + 1)


@dash_app.callback(Output("split-store", "data"), Output("split-button", "className"),
              Output("scene-grid", "className"), Output("scene-header-controls", "className"),
              Input("split-button", "n_clicks"), State("split-store", "data"), prevent_initial_call=True)
def toggle_split(n_clicks, split):
    split = not split
    return (split, ("toggle active" if split else "toggle"), ("scene-grid split" if split else "scene-grid"),
            ("header-controls split" if split else "header-controls"))


# --------------------------------------------------------------------------
# Small helpers used by callbacks
# --------------------------------------------------------------------------

def epoch_plus_seconds(epoch_utc, seconds):
    """UTC string for a time given in seconds past the epoch."""
    moment = datetime.fromisoformat(epoch_utc) + timedelta(seconds=float(seconds))
    return moment.strftime("%Y-%m-%d %H:%M")


def parse_state_text(text):
    """'x, y, z, vx, vy, vz' -> list of six floats."""
    values = [float(part) for part in text.replace(";", ",").split(",") if part.strip() != ""]
    if len(values) != 6:
        raise ValueError("initial state needs six numbers")
    return values


def pair_key(option_value):
    """Dropdown values are 'observer|spacecraft' strings; results use tuples."""
    observer, _, spacecraft = option_value.partition("|")
    return observer, spacecraft


# --------------------------------------------------------------------------
# Tree
# --------------------------------------------------------------------------

def tree_item(obj, selected, meta, child=False):
    glyph, glyph_class = TREE_GLYPH[obj.kind]
    class_name = "tree-item"
    if obj.name == selected:
        class_name = class_name + " selected"
    if child:
        class_name = class_name + " child"
    return html.Button([html.Span(glyph, className=f"tree-glyph {glyph_class}"),
                        html.Span(obj.name, className="tree-name"),
                        html.Span(meta, className="tree-meta")],
                       id={"type": "tree-item", "name": obj.name}, className=class_name, n_clicks=0)


@dash_app.callback(Output("tree", "children"),
              Input("scenario-store", "data"), Input("selected-store", "data"))
def render_tree(scenario_data, selected):
    scenario = Scenario.from_dict(scenario_data)
    items = [html.Div("Spacecraft", className="tree-group")]
    for spacecraft in scenario.spacecraft:
        if spacecraft.source == "family":
            meta = f"#{spacecraft.family_index}"
        elif spacecraft.source == "elements":
            meta = f"{spacecraft.centre} elements"
        else:
            meta = "periodic" if spacecraft.period_tu > 0.0 else "state"
        items.append(tree_item(spacecraft, selected, meta))
    if len(scenario.spacecraft) == 0:
        items.append(html.Div("none", className="empty"))
    items.append(html.Div("Ground stations", className="tree-group"))
    for station in scenario.ground_stations:
        items.append(tree_item(station, selected, f"{station.latitude_deg:+.1f}, {station.longitude_deg:+.1f}"))
        for sensor in scenario.sensors_of(station.name):
            items.append(tree_item(sensor, selected, f"m<{sensor.limiting_magnitude:g}", child=True))
    if len(scenario.ground_stations) == 0:
        items.append(html.Div("none", className="empty"))
    return items


# --------------------------------------------------------------------------
# Property form
# --------------------------------------------------------------------------

def prop_input(name, value, **kwargs):
    return dcc.Input(id={"type": "prop", "field": name}, value=value, debounce=False, **kwargs)


def prop_dropdown(name, value, options):
    return dcc.Dropdown(id={"type": "prop", "field": name}, value=value, options=options,
                        clearable=False, className="dash-dropdown")


def pick_input(name, value, **kwargs):
    """Inputs that belong to a form action rather than to a property."""
    return dcc.Input(id={"type": "pick", "field": name}, value=value, **kwargs)


def action_button(label, name, primary=False):
    class_name = "primary small" if primary else "small"
    return html.Button(label, id={"type": "form-action", "name": name}, className=class_name, n_clicks=0)


def spacecraft_form(obj, scenario):
    """The property form of a spacecraft: one block per orbit source."""
    family_name = obj.family_name if obj.family_name in FAMILIES else DEFAULT_FAMILY_NAME
    fields = [field("Defined by", prop_dropdown("source", obj.source,
                    [{"label": "Rotating-frame initial state", "value": "state"},
                     {"label": "Orbit family member", "value": "family"},
                     {"label": "Two-body elements (Moon or Earth)", "value": "elements"}]))]

    # ---- family ----
    family_block = [
        field("Family", prop_dropdown("family_name", family_name,
                                      [{"label": name, "value": name} for name in FAMILY_NAMES])),
        field("Member", prop_dropdown("family_index", int(np.clip(obj.family_index, 0, len(FAMILIES[family_name]) - 1)),
                                      [{"label": label, "value": index}
                                       for index, label in enumerate(FAMILY_LABELS[family_name])])),
        html.Div([field("Perilune [km]", pick_input("perilune_km", None, type="number", min=0)),
                  field("Period [days]", pick_input("period_days", None, type="number", min=0))],
                 className="form-row"),
        html.Div([action_button("Pick nearest member", "nearest")], className="form-actions"),
        html.Span("Fill one or both and press the button: the member closest to those values is "
                  "selected.", className="hint"),
    ]
    fields.append(html.Div(family_block, className="form-block", hidden=obj.source != "family",
                           id={"type": "source-block", "source": "family"}))

    # ---- state ----
    state_block = [
        field("Initial state  x, y, z, vx, vy, vz",
              prop_input("initial_state", ", ".join(f"{v:.10g}" for v in obj.initial_state),
                         type="text", className="mono"),
              hint="rotating frame, LU and LU/TU"),
        field("Known period [TU]", prop_input("period_tu", obj.period_tu, type="number", min=0),
              hint="0 if the state is not a converged periodic orbit"),
        html.Details([
            html.Summary("Correct to a periodic orbit"),
            html.P("A state on the xz-plane with velocity perpendicular to it (y = vx = vz = 0) is corrected "
                   "with the symmetric halo or planar corrector, holding one component fixed. Any other state "
                   "uses the general corrector and needs a period guess.", className="help-text"),
            html.Div([field("Hold fixed", dcc.Dropdown(id={"type": "pick", "field": "fixed"}, value="auto",
                                                       clearable=False, className="dash-dropdown",
                                                       options=[{"label": "automatic", "value": "auto"},
                                                                {"label": "x0", "value": "x0"},
                                                                {"label": "z0", "value": "z0"},
                                                                {"label": "vy0", "value": "vy0"}])),
                      field("Period guess [days]", pick_input("period_guess_days", None, type="number", min=0))],
                     className="form-row"),
            html.Div([action_button("Correct to periodic", "correct", primary=True)], className="form-actions"),
        ], className="details"),
    ]
    fields.append(html.Div(state_block, className="form-block", hidden=obj.source != "state",
                           id={"type": "source-block", "source": "state"}))

    # ---- elements ----
    elements = obj.elements
    element_block = [
        field("Preset", dcc.Dropdown(id={"type": "pick", "field": "preset"}, value=None, placeholder="choose a preset",
                                     className="dash-dropdown",
                                     options=[{"label": name, "value": name} for name in ELEMENT_PRESETS])),
        html.Div([action_button("Load preset", "preset")], className="form-actions"),
        html.Div([field("Central body", prop_dropdown("centre", obj.centre,
                                                      [{"label": "Moon", "value": "moon"},
                                                       {"label": "Earth", "value": "earth"}])),
                  field("Reference plane", prop_dropdown("reference_plane", obj.reference_plane,
                                                         [{"label": "Moon orbit plane", "value": "moon orbit"},
                                                          {"label": "Earth equator", "value": "earth equator"}]))],
                 className="form-row"),
        html.Div([field("a [km]", prop_input("element_a_km", elements["a_km"], type="number", min=0)),
                  field("e", prop_input("element_e", elements["e"], type="number", min=0, max=0.999, step=0.0001)),
                  field("i [deg]", prop_input("element_i_deg", elements["i_deg"], type="number"))],
                 className="form-row three"),
        html.Div([field("RAAN [deg]", prop_input("element_raan_deg", elements["raan_deg"], type="number")),
                  field("Arg. periapsis [deg]", prop_input("element_argp_deg", elements["argp_deg"], type="number")),
                  field("True anomaly [deg]", prop_input("element_true_anomaly_deg", elements["true_anomaly_deg"],
                                                         type="number"))],
                 className="form-row three"),
        html.Span("Osculating elements at the epoch. Inclination and node are measured from the chosen "
                  "reference plane; the Earth-equator option places the node against the equinox.",
                  className="hint"),
    ]
    fields.append(html.Div(element_block, className="form-block", hidden=obj.source != "elements",
                           id={"type": "source-block", "source": "elements"}))

    # ---- common ----
    fields.append(field("Propagation", prop_dropdown("propagation", obj.propagation,
                        [{"label": "Integrate initial state", "value": "integrate"},
                         {"label": "Repeat converged period (station-kept)", "value": "periodic"}]),
                        hint="periodic needs a family member or a corrected state"))
    fields.append(html.Div([field("Diameter [m]", prop_input("diameter_m", obj.diameter_m, type="number", min=0)),
                            field("Albedo", prop_input("albedo", obj.albedo, type="number", min=0, max=1, step=0.01))],
                           className="form-row"))
    fields.append(html.Div([field("Manifolds", prop_dropdown("manifolds", obj.manifolds,
                                                             [{"label": "none", "value": "none"},
                                                              {"label": "unstable", "value": "unstable"},
                                                              {"label": "stable", "value": "stable"},
                                                              {"label": "both", "value": "both"}])),
                            field("Branches", prop_input("manifold_branches", obj.manifold_branches, type="number",
                                                         min=1, max=64, step=1)),
                            field("Length [days]", prop_input("manifold_time_days", obj.manifold_time_days,
                                                              type="number", min=0.1))],
                           className="form-row three"))
    fields.append(html.Span("Manifolds are drawn for periodic orbits only.", className="hint"))

    # ---- description of the resulting orbit ----
    epoch_jd = frames.julian_date(scenario.epoch_utc)
    fields.append(html.Div([html.Div(line, className="describe-line")
                            for line in orbits.describe(obj, FAMILIES, epoch_jd)], className="describe"))
    return fields


@dash_app.callback(Output("form", "children"), Output("apply-button", "hidden"),
              Input("selected-store", "data"), State("scenario-store", "data"))
def render_form(selected, scenario_data):
    scenario = Scenario.from_dict(scenario_data)
    obj = scenario.find(selected) if selected else None
    if obj is None:
        return html.Div("Select an object to edit its properties.", className="empty"), True

    fields = [html.Div([html.Div(obj.kind.replace("_", " "), className="form-kind"),
                        html.Div(obj.name, className="form-title")]),
              field("Name", prop_input("name", obj.name, type="text"))]

    if isinstance(obj, Spacecraft):
        fields.extend(spacecraft_form(obj, scenario))
    elif isinstance(obj, GroundStation):
        fields.append(html.Div([field("Latitude [deg]", prop_input("latitude_deg", obj.latitude_deg, type="number")),
                                field("Longitude [deg]", prop_input("longitude_deg", obj.longitude_deg, type="number"))],
                               className="form-row"))
        fields.append(field("Altitude [km]", prop_input("altitude_km", obj.altitude_km, type="number")))
        fields.append(html.Div([field("Min elevation [deg]", prop_input("min_elevation_deg", obj.min_elevation_deg, type="number")),
                                field("Max Sun elev. [deg]", prop_input("max_sun_elevation_deg", obj.max_sun_elevation_deg, type="number"))],
                               className="form-row"))
        fields.append(html.Span("Station is dark when the Sun is below the max Sun elevation "
                                "(-6 civil, -12 nautical, -18 astronomical twilight).", className="hint"))
    elif isinstance(obj, OpticalSensor):
        fields.append(field("Ground station", prop_dropdown("station", obj.station,
                            [{"label": s.name, "value": s.name} for s in scenario.ground_stations])))
        fields.append(html.Div([field("Limiting magnitude", prop_input("limiting_magnitude", obj.limiting_magnitude, type="number")),
                                field("Lunar exclusion [deg]", prop_input("lunar_exclusion_deg", obj.lunar_exclusion_deg, type="number"))],
                               className="form-row"))
        fields.append(html.Span("An L2 NRHO stays within about 10 degrees of the Moon as seen from Earth.",
                                className="hint"))
        fields.append(html.Div([field("Max range [km]", prop_input("max_range_km", obj.max_range_km, type="number", min=0)),
                                field("Max slew rate [deg/s]", prop_input("max_slew_rate_deg_s", obj.max_slew_rate_deg_s,
                                                                          type="number", min=0, step=0.001))],
                               className="form-row"))
        fields.append(html.Span("0 means no limit. Range suits a radar or link budget; slew rate a mount limit "
                                "(cislunar targets move a few thousandths of a degree per second).", className="hint"))

    return fields, False


@dash_app.callback(Output({"type": "source-block", "source": ALL}, "hidden"),
              Input({"type": "prop", "field": "source"}, "value"),
              State({"type": "source-block", "source": ALL}, "id"), prevent_initial_call=True)
def show_source_block(source, block_ids):
    """Only the block for the chosen orbit source is visible."""
    return [block_id["source"] != source for block_id in block_ids]


@dash_app.callback(Output({"type": "prop", "field": "family_index"}, "options"),
              Output({"type": "prop", "field": "family_index"}, "value"),
              Input({"type": "prop", "field": "family_name"}, "value"),
              State({"type": "prop", "field": "family_index"}, "value"), prevent_initial_call=True)
def family_members_for(family_name, current_index):
    """The member list follows the chosen family."""
    if family_name not in FAMILIES:
        return no_update, no_update
    labels = FAMILY_LABELS[family_name]
    index = int(np.clip(current_index or 0, 0, len(labels) - 1))
    return [{"label": label, "value": k} for k, label in enumerate(labels)], index


# --------------------------------------------------------------------------
# Scenario edits: selection, add, remove, apply, settings, load
# --------------------------------------------------------------------------

def apply_form_values(scenario, obj, prop_values, prop_ids):
    """
    Write the property form back onto the object.  Element fields are
    gathered into the elements dictionary; the initial state text is
    parsed; numbers are coerced to the field's type.  Returns the
    object's final name (renaming is done here too).
    """
    values = {prop_id["field"]: value for prop_id, value in zip(prop_ids, prop_values)}
    new_name = values.pop("name", obj.name) or obj.name
    for key, value in values.items():
        if key.startswith("element_"):
            if value is not None:
                obj.elements[key[len("element_"):]] = float(value)
            continue
        if key == "initial_state":
            try:
                value = parse_state_text(value)
            except ValueError:
                continue
        elif key in ("family_index", "manifold_branches"):
            if value is None:
                continue
            value = int(value)
        elif isinstance(getattr(obj, key), float):
            if value is None:
                continue
            value = float(value)
        setattr(obj, key, value)
    if new_name != obj.name:
        new_name = scenario.unique_name(new_name)
        scenario.rename(obj.name, new_name)
    return obj.name


def status_message(text_value, kind="ok"):
    return html.Div(text_value, className=f"status-line {kind}")


@dash_app.callback(Output("scenario-store", "data"), Output("selected-store", "data"),
              Output("scenario-name", "value"), Output("scenario-epoch", "value"),
              Output("scenario-duration", "value"), Output("scenario-step", "value"),
              Output("form-status", "children"),
              Input({"type": "tree-item", "name": ALL}, "n_clicks"),
              Input("add-button", "n_clicks"), Input("add-range-button", "n_clicks"),
              Input("remove-button", "n_clicks"),
              Input("apply-button", "n_clicks"), Input("load-upload", "contents"),
              Input({"type": "form-action", "name": ALL}, "n_clicks"),
              Input("scenario-name", "value"), Input("scenario-epoch", "value"),
              Input("scenario-duration", "value"), Input("scenario-step", "value"),
              State("add-type", "value"), State("selected-store", "data"),
              State("range-family", "value"),
              State("range-from", "value"), State("range-to", "value"), State("range-step", "value"),
              State({"type": "prop", "field": ALL}, "value"), State({"type": "prop", "field": ALL}, "id"),
              State({"type": "pick", "field": ALL}, "value"), State({"type": "pick", "field": ALL}, "id"),
              State("scenario-store", "data"),
              prevent_initial_call=True)
def edit_scenario(tree_clicks, add_clicks, add_range_clicks, remove_clicks, apply_clicks, upload_contents,
                  action_clicks, name, epoch, duration, step, add_type, selected, range_family,
                  range_from, range_to, range_step, prop_values, prop_ids, pick_values, pick_ids, scenario_data):
    trigger = ctx.triggered_id
    scenario = Scenario.from_dict(scenario_data)
    settings_unchanged = (no_update, no_update, no_update, no_update)
    no_status = no_update

    if isinstance(trigger, dict) and trigger.get("type") == "tree-item":
        # A click on a tree row selects it; ignore the spurious trigger that
        # fires when rows are re-rendered with n_clicks = 0.
        if not any(clicks for clicks in tree_clicks):
            return (no_update, no_update) + settings_unchanged + (no_status,)
        return no_update, trigger["name"], *settings_unchanged, ""

    if isinstance(trigger, dict) and trigger.get("type") == "form-action":
        if not any(clicks for clicks in action_clicks):
            return (no_update, no_update) + settings_unchanged + (no_status,)
        obj = scenario.find(selected)
        if not isinstance(obj, Spacecraft):
            return (no_update, no_update) + settings_unchanged + (no_status,)
        apply_form_values(scenario, obj, prop_values, prop_ids)
        picks = {pick_id["field"]: value for pick_id, value in zip(pick_ids, pick_values)}
        action = trigger["name"]

        if action == "nearest":
            perilune_km = picks.get("perilune_km")
            period_days = picks.get("period_days")
            if perilune_km is None and period_days is None:
                return (no_update, no_update) + settings_unchanged + (
                    status_message("Enter a perilune radius, a period, or both.", "warn"),)
            family = FAMILIES[obj.family_name]
            obj.family_index = nearest_member(family, perilune_km, period_days)
            chosen = family[obj.family_index]
            message = (f"Member {obj.family_index}: period {crtbp.time_to_days(chosen['period']):.3f} d, "
                       f"perilune {crtbp.length_to_km(chosen['perilune_radius']):,.0f} km, "
                       f"apolune {crtbp.length_to_km(chosen['apolune_radius']):,.0f} km, "
                       f"stability index {chosen['stability_index']:.2f}.")
            return scenario.to_dict(), obj.name, *settings_unchanged, status_message(message)

        if action == "preset":
            preset = picks.get("preset")
            if preset not in ELEMENT_PRESETS:
                return (no_update, no_update) + settings_unchanged + (
                    status_message("Choose a preset first.", "warn"),)
            obj.centre, obj.reference_plane, elements = ELEMENT_PRESETS[preset]
            obj.elements = dict(elements)
            obj.source = "elements"
            obj.propagation = "integrate"
            return scenario.to_dict(), obj.name, *settings_unchanged, status_message(f"Loaded {preset}.")

        if action == "correct":
            fixed = picks.get("fixed")
            fixed = None if fixed in (None, "auto") else fixed
            period_guess = picks.get("period_guess_days")
            epoch_jd = frames.julian_date(scenario.epoch_utc)
            try:
                info = orbits.correct_to_periodic(obj, FAMILIES, epoch_jd, fixed=fixed,
                                                  period_guess_days=period_guess)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                return (no_update, no_update) + settings_unchanged + (
                    status_message(f"Correction failed: {error}", "error"),)
            message = (f"Converged with the {info['corrector']} corrector in {info['iterations']} iterations "
                       f"(residual {info['residual']:.1e}): period {info['period_days']:.4f} d, "
                       f"C = {info['jacobi']:.5f}, perilune {info['perilune_km']:,.0f} km, "
                       f"apolune {info['apolune_km']:,.0f} km, stability index {info['stability_index']:.2f}. "
                       f"Propagation set to periodic.")
            return scenario.to_dict(), obj.name, *settings_unchanged, status_message(message)

        return (no_update, no_update) + settings_unchanged + (no_status,)

    if trigger == "add-button":
        if add_type == "spacecraft":
            new = scenario.add(Spacecraft(name="Spacecraft"))
        elif add_type == "ground_station":
            new = scenario.add(GroundStation(name="Station"))
        else:
            parent = scenario.find(selected)
            if isinstance(parent, OpticalSensor):
                parent = scenario.find(parent.station)
            if not isinstance(parent, GroundStation):
                parent = scenario.ground_stations[0] if scenario.ground_stations else None
            if parent is None:
                parent = scenario.add(GroundStation(name="Station"))
            new = scenario.add(OpticalSensor(name="Sensor", station=parent.name))
        return scenario.to_dict(), new.name, *settings_unchanged, ""

    if trigger == "add-range-button":
        family_name = range_family if range_family in FAMILIES else DEFAULT_FAMILY_NAME
        family = FAMILIES[family_name]
        first = int(np.clip(range_from or 0, 0, len(family) - 1))
        last = int(np.clip(range_to if range_to is not None else len(family) - 1, 0, len(family) - 1))
        stride = max(1, int(range_step or 1))
        new = None
        short = family_name.replace(" halo", "").replace(" ", "-")
        for index in range(first, last + 1, stride):
            new = scenario.add(Spacecraft(name=f"{short} #{index}", source="family", family_name=family_name,
                                          family_index=index, propagation="periodic"))
        return scenario.to_dict(), (new.name if new else no_update), *settings_unchanged, ""

    if trigger == "remove-button":
        if selected:
            scenario.remove(selected)
        return scenario.to_dict(), None, *settings_unchanged, ""

    if trigger == "apply-button":
        obj = scenario.find(selected)
        if obj is None:
            return (no_update, no_update) + settings_unchanged + (no_status,)
        apply_form_values(scenario, obj, prop_values, prop_ids)
        return scenario.to_dict(), obj.name, *settings_unchanged, status_message(
            "Applied. Press Run analysis to update the scene and the windows.")

    if trigger == "load-upload":
        _, _, encoded = upload_contents.partition(",")
        loaded = Scenario.from_json(base64.b64decode(encoded).decode("utf-8"))
        return (loaded.to_dict(), None, loaded.name, loaded.epoch_utc,
                loaded.duration_days, loaded.time_step_s, "")

    # Otherwise one of the scenario settings changed.
    if name:
        scenario.name = name
    if epoch:
        try:
            datetime.fromisoformat(epoch)
            scenario.epoch_utc = epoch
        except ValueError:
            pass
    if duration:
        scenario.duration_days = float(duration)
    if step:
        scenario.time_step_s = float(step)
    return scenario.to_dict(), no_update, *settings_unchanged, no_status


@dash_app.callback(Output("download", "data"), Input("save-button", "n_clicks"),
              State("scenario-store", "data"), prevent_initial_call=True)
def save_scenario(n_clicks, scenario_data):
    scenario = Scenario.from_dict(scenario_data)
    file_name = scenario.name.strip().replace(" ", "_") or "scenario"
    return dict(content=scenario.to_json(), filename=f"{file_name}.json")


# --------------------------------------------------------------------------
# Run the analysis
# --------------------------------------------------------------------------

@dash_app.callback(Output("results-store", "data"), Output("run-status", "children"),
              Output("pair-select", "options"), Output("pair-select", "value"),
              Output("time-slider", "max"), Output("time-slider", "marks"), Output("time-slider", "value"),
              Input("run-button", "n_clicks"), State("scenario-store", "data"),
              State("pair-select", "value"))
def run_analysis(n_clicks, scenario_data, current_pair):
    scenario = Scenario.from_dict(scenario_data)
    started = time.perf_counter()
    results = runner.run_scenario(scenario, FAMILIES)
    elapsed = time.perf_counter() - started

    run_id = str(uuid.uuid4())
    RESULTS.clear()
    RESULTS[run_id] = {"scenario": scenario, "results": results}

    options = [{"label": f"{observer}  →  {spacecraft}", "value": f"{observer}|{spacecraft}"}
               for observer, spacecraft in results["observations"]]
    values = [option["value"] for option in options]
    pair = current_pair if current_pair in values else (values[0] if values else None)

    n_samples = len(results["times_s"])
    day_stride = max(1, int(round(scenario.duration_days / 7)))
    marks = {}
    for day in range(0, int(np.floor(scenario.duration_days)) + 1, day_stride):
        index = int(round(day * crtbp.SECONDS_PER_DAY / scenario.time_step_s))
        if index < n_samples:
            marks[index] = f"{day} d"

    status = f"{n_samples:,} samples · {len(scenario.spacecraft)} spacecraft · {elapsed:.1f} s"
    return run_id, status, options, pair, n_samples - 1, marks, 0


# --------------------------------------------------------------------------
# Results: windows table and summary
# --------------------------------------------------------------------------

@dash_app.callback(Output("windows-table", "data"), Output("summary", "children"),
              Input("pair-select", "value"), Input("results-store", "data"))
def update_windows(pair, run_id):
    if run_id not in RESULTS or not pair:
        return [], html.Div("Run the analysis to compute access windows.", className="empty")
    scenario = RESULTS[run_id]["scenario"]
    results = RESULTS[run_id]["results"]
    key = pair_key(pair)
    windows = results["windows"][key]
    observation = results["observations"][key]

    rows = []
    for index, (start, stop) in enumerate(windows, start=1):
        rows.append({"index": index,
                     "start": epoch_plus_seconds(scenario.epoch_utc, start),
                     "stop": epoch_plus_seconds(scenario.epoch_utc, stop),
                     "duration": f"{(stop - start) / 3600.0:.2f}"})

    total_hours = sum(stop - start for start, stop in windows) / 3600.0
    duty = results["duty_cycle"][key]
    stats = html.Div([
        html.Div([html.Div("Duty cycle", className="stat-label"),
                  html.Div(f"{100.0 * duty:.1f}", className="stat-value"),
                  html.Div("% of span observable", className="stat-unit")], className="stat"),
        html.Div([html.Div("Windows", className="stat-label"),
                  html.Div(f"{len(windows)}", className="stat-value"),
                  html.Div(f"{total_hours:.1f} h total", className="stat-unit")], className="stat"),
        html.Div([html.Div("Longest", className="stat-label"),
                  html.Div(f"{max([(stop - start) for start, stop in windows], default=0.0) / 3600.0:.1f}",
                           className="stat-value"),
                  html.Div("hours", className="stat-unit")], className="stat"),
    ], className="stat-row")

    # How much of the span each individual constraint allows, so it is
    # obvious which one is doing the cutting.  Names come from the
    # constraint functions themselves.
    chips = []
    for name, column in zip(observation["constraint_names"], observation["constraint_masks"].T):
        chips.append(html.Span(f"{name}: {100.0 * column.mean():.0f}%", className="chip"))

    # Coverage of this spacecraft by all observers together.
    coverage = results["coverage"].get(key[1])
    coverage_note = []
    if coverage is not None and len(runner.observers(scenario)) > 1:
        coverage_note = [html.Div(f"All observers together see {key[1]} {100.0 * coverage['duty_cycle']:.1f}% of the "
                                  f"span in {len(coverage['windows'])} windows; at most "
                                  f"{int(coverage['count'].max())} at once.", className="coverage-note")]
    return rows, [stats, html.Div(chips, className="constraints")] + coverage_note


# --------------------------------------------------------------------------
# 3D view, time series and slider
# --------------------------------------------------------------------------

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


@dash_app.callback(Output("focus-select", "options"), Output("focus-select", "value"),
              Input("scenario-store", "data"), State("focus-select", "value"))
def focus_options(scenario_data, current):
    scenario = Scenario.from_dict(scenario_data)
    options = [{"label": "Focus: free", "value": "none"},
               {"label": "Focus: Moon", "value": "body:moon"},
               {"label": "Focus: Earth", "value": "body:earth"},
               {"label": "Focus: L1", "value": "point:L1"},
               {"label": "Focus: L2", "value": "point:L2"}]
    options.extend({"label": f"Focus: {spacecraft.name}", "value": f"spacecraft:{spacecraft.name}"}
                   for spacecraft in scenario.spacecraft)
    values = [option["value"] for option in options]
    return options, current if current in values else "none"


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


def scene_figure(results, frame, view, focus, index):
    """The 3D figure of one frame at one time index."""
    trajectories, manifold_branches, stations, bodies, points = displayed_frame(results, frame)
    markers = {name: states[index] for name, states in trajectories.items()}
    trail = max(2, len(results["times_s"]) // 12)
    return figures.trajectory_figure(trajectories, bodies, BODY_RADII, index=index, points=points,
                                     station_positions=stations, marker_states=markers,
                                     manifolds=manifold_branches, view=view,
                                     frame_label=FRAME_LABELS[frame],
                                     focus_point=focus_point_for(focus, trajectories, bodies, points, index),
                                     focus_key=focus, trail_samples=trail)


@dash_app.callback(Output("view-3d", "figure"), Output("view-3d-b", "figure"), Output("time-series", "figure"),
              Output("time-readout", "children"),
              Input("results-store", "data"), Input("pair-select", "value"),
              Input("time-slider", "value"), Input("view-select", "value"), Input("frame-select", "value"),
              Input("frame-select-b", "value"), Input("split-store", "data"), Input("focus-select", "value"))
def update_views(run_id, pair, slider_index, view, frame, frame_b, split, focus):
    if run_id not in RESULTS:
        bodies = {"earth": FIXED_POINTS["earth"], "moon": FIXED_POINTS["moon"]}
        points = {"L1": FIXED_POINTS["L1"], "L2": FIXED_POINTS["L2"]}
        figure_3d = figures.trajectory_figure({}, bodies, BODY_RADII, points=points, view=view,
                                              focus_point=focus_point_for(focus, {}, bodies, points, 0),
                                              focus_key=focus)
        return figure_3d, figure_3d, figures.empty_time_series_figure(), ""

    scenario = RESULTS[run_id]["scenario"]
    results = RESULTS[run_id]["results"]
    index = int(np.clip(slider_index or 0, 0, len(results["times_s"]) - 1))
    current_time_s = results["times_s"][index]

    figure_3d = scene_figure(results, frame, view, focus, index)
    figure_3d_b = scene_figure(results, frame_b, view, focus, index) if split else no_update

    if pair:
        key = pair_key(pair)
        station, sensor = runner.observer_settings(scenario, key[0])
        thresholds = {"elevation_deg": station.min_elevation_deg if station else None,
                      "apparent_magnitude": sensor.limiting_magnitude if sensor else None,
                      "lunar_separation_deg": sensor.lunar_exclusion_deg if sensor else None}
        figure_series = figures.time_series_figure(results["observations"][key]["geometry"], thresholds,
                                                   results["windows"][key], current_time_s)
    else:
        figure_series = figures.empty_time_series_figure("No observer-spacecraft pairs in this scenario")

    readout = f"{epoch_plus_seconds(scenario.epoch_utc, current_time_s)} UTC  " \
              f"(+{current_time_s / crtbp.SECONDS_PER_DAY:.3f} d, {results['times_nondim'][index]:.4f} TU)"
    return figure_3d, figure_3d_b, figure_series, readout


# --------------------------------------------------------------------------
# Parameter sweep
# --------------------------------------------------------------------------

SWEEP_ROWS = {}


@dash_app.callback(Output("sweep-object", "options"), Output("sweep-object", "value"),
              Input("scenario-store", "data"), State("sweep-object", "value"))
def sweep_objects(scenario_data, current):
    scenario = Scenario.from_dict(scenario_data)
    options = [{"label": "Scenario", "value": "scenario"}]
    options.extend({"label": obj.name, "value": obj.name} for obj in scenario.all_objects())
    values = [option["value"] for option in options]
    return options, current if current in values else values[0]


@dash_app.callback(Output("sweep-attribute", "options"), Output("sweep-attribute", "value"),
              Input("sweep-object", "value"), State("scenario-store", "data"), State("sweep-attribute", "value"))
def sweep_attributes(target_name, scenario_data, current):
    scenario = Scenario.from_dict(scenario_data)
    if target_name == "scenario":
        kind = "scenario"
    else:
        obj = scenario.find(target_name)
        if obj is None:
            return [], None
        kind = obj.kind
    options = [{"label": label, "value": attribute} for attribute, label in sweep.SWEEPABLE[kind]]
    values = [option["value"] for option in options]
    return options, current if current in values else values[0]


@dash_app.callback(Output("sweep-table", "data"), Output("sweep-table", "columns"), Output("sweep-status", "children"),
              Input("sweep-button", "n_clicks"), State("sweep-object", "value"), State("sweep-attribute", "value"),
              State("sweep-from", "value"), State("sweep-to", "value"), State("sweep-steps", "value"),
              State("scenario-store", "data"), prevent_initial_call=True)
def run_sweep(n_clicks, target_name, attribute, start, stop, steps, scenario_data):
    if not target_name or not attribute or start is None or stop is None:
        return [], [], "Choose an object, a setting and a range."
    scenario = Scenario.from_dict(scenario_data)
    values = np.linspace(float(start), float(stop), int(max(2, steps or 2)))
    if attribute == "family_index":
        values = np.unique(np.round(values).astype(int))
    started = time.perf_counter()
    try:
        rows = sweep.sweep(scenario, target_name, attribute, values, FAMILIES)
    except (ValueError, KeyError, IndexError) as error:
        return [], [], f"Sweep failed: {error}"
    elapsed = time.perf_counter() - started
    SWEEP_ROWS.clear()
    SWEEP_ROWS["rows"] = rows

    shown = []
    for row in rows:
        shown.append({attribute: f"{row[attribute]:g}",
                      "pair": f"{row['observer']} → {row['spacecraft']}",
                      "duty %": f"{100.0 * row['duty_cycle']:.1f}",
                      "windows": row["n_windows"],
                      "total h": f"{row['total_hours']:.1f}",
                      "longest h": f"{row['longest_hours']:.2f}"})
    columns = [{"name": name, "id": name} for name in shown[0].keys()] if shown else []
    return shown, columns, f"{len(values)} runs in {elapsed:.1f} s"


@dash_app.callback(Output("sweep-download", "data"), Input("sweep-download-button", "n_clicks"),
              State("sweep-attribute", "value"), prevent_initial_call=True)
def download_sweep(n_clicks, attribute):
    rows = SWEEP_ROWS.get("rows")
    if not rows:
        return no_update
    import io
    buffer = io.StringIO()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    import csv
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return dict(content=buffer.getvalue(), filename=f"sweep_{attribute}.csv")


if __name__ == "__main__":
    dash_app.run(debug=False, host="127.0.0.1", port=8050)
