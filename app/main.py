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

from engine import crtbp, propagation
from model import runner
from model.family import load_family
from model.scenario import Scenario, Spacecraft, GroundStation, OpticalSensor, example_scenario
from app import figures

dash_app = Dash(__name__, title="Cislunar mission tool", suppress_callback_exceptions=True)

# Results of the last run, kept in memory on the server.  The app runs
# locally for one user, so a module-level dictionary is enough; the
# browser only holds the run id.
RESULTS = {}
FAMILY = load_family()
FAMILY_LABELS = [f"{index}: T = {crtbp.time_to_days(orbit['period']):.2f} d, "
                 f"perilune {crtbp.length_to_km(orbit['perilune_radius']):,.0f} km, C = {orbit['jacobi']:.4f}"
                 for index, orbit in enumerate(FAMILY)]
FIXED_POINTS = propagation.fixed_points()

TREE_GLYPH = {"spacecraft": ("◆", "glyph-spacecraft"),
              "ground_station": ("▲", "glyph-station"),
              "optical_sensor": ("●", "glyph-sensor")}


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

def setting(label, component):
    return html.Div([html.Label(label), component], className="setting")


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
    dcc.Download(id="download"),

    # ---- top bar ----
    html.Div([
        html.Div([html.Span("Cislunar mission tool", className="brand-title"),
                  html.Span("Earth-Moon CRTBP, rotating frame", className="brand-subtitle")],
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
            setting("View", dcc.Dropdown(id="view-select", className="dash-dropdown narrow", clearable=False,
                                         value="moon", options=[{"label": "Moon region", "value": "moon"},
                                                                {"label": "Whole system", "value": "system"}])),
        ], className="settings"),
        html.Div([
            html.Span("Panels", className="panel-title"),
            toggle_button("Tree", "toggle-left"),
            toggle_button("Windows", "toggle-right"),
            toggle_button("Time series", "toggle-bottom"),
        ], className="toggles"),
        html.Div([
            html.Button("Run analysis", id="run-button", className="primary", n_clicks=0),
            html.Button("Save JSON", id="save-button", n_clicks=0),
            dcc.Upload(html.Div("Load JSON", className="upload-box"), id="load-upload", multiple=False),
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
                html.Summary("Add several halo family members"),
                html.Div([
                    html.Div([html.Label("From"), dcc.Input(id="range-from", type="number", value=0, min=0,
                                                            max=len(FAMILY) - 1, step=1)], className="field"),
                    html.Div([html.Label("To"), dcc.Input(id="range-to", type="number", value=len(FAMILY) - 1, min=0,
                                                          max=len(FAMILY) - 1, step=1)], className="field"),
                    html.Div([html.Label("Every"), dcc.Input(id="range-step", type="number", value=10, min=1, step=1)],
                             className="field"),
                ], className="form-row three"),
                html.Button("Add members", id="add-range-button", className="small", n_clicks=0),
                html.Span(f"Family has {len(FAMILY)} members: index 0 is the largest halo, "
                          f"{len(FAMILY) - 1} has the lowest perilune.", className="hint"),
            ], className="details"),
            html.Details([
                html.Summary("How to set up a simulation"),
                html.Ol([
                    html.Li("Set the epoch, duration and step at the top."),
                    html.Li("Add spacecraft: a halo family member (periodic, station-kept) or your own "
                            "rotating-frame initial state (integrated)."),
                    html.Li("Add a ground station and, optionally, an optical sensor on it."),
                    html.Li("Select an object, edit its fields, press Apply."),
                    html.Li("Press Run analysis. Pick an observer-spacecraft pair on the right."),
                    html.Li("Save JSON to keep the scenario; scripts/ shows how to sweep it without the GUI."),
                ], className="help"),
            ], className="details"),
            html.Div(className="divider"),
            html.Div(id="form", className="form"),
            html.Div([html.Button("Apply", id="apply-button", className="primary small", n_clicks=0, hidden=True)],
                     className="form-actions"),
        ], panel_id="left-panel"),
        panel("Rotating frame", dcc.Graph(id="view-3d", style={"height": "100%"}, responsive=True,
                                          config={"displaylogo": False}),
              header_extra=html.Span(id="run-status", className="status"),
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
                                 style_cell={"padding": "5px 8px", "textAlign": "left", "whiteSpace": "nowrap"},
                                 style_cell_conditional=[{"if": {"column_id": "index"}, "width": "30px"},
                                                         {"if": {"column_id": "duration"}, "textAlign": "right"}]),
        ], panel_id="right-panel"),
    ], className="main", id="main-row"),

    # ---- bottom ----
    html.Div([
        panel("Time series", [
            dcc.Graph(id="time-series", figure=figures.empty_time_series_figure(), responsive=True,
                      style={"height": "360px"}, config={"displaylogo": False}),
            html.Div("Green bands are the access windows. Dashed lines are the thresholds: the spacecraft must "
                     "be above the elevation line, brighter (lower on the reversed magnitude axis) than the "
                     "limiting magnitude, and farther from the Moon than the exclusion angle. Darkness and "
                     "shadow are not drawn but still apply. The black line is the slider time.",
                     className="help-text panel-note"),
            html.Div([
                html.Span("Time", className="panel-title"),
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
        meta = f"family #{spacecraft.family_index}" if spacecraft.source == "family" else "state"
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

def field(label, component, hint=None):
    children = [html.Label(label), component]
    if hint:
        children.append(html.Span(hint, className="hint"))
    return html.Div(children, className="field")


def prop_input(name, value, **kwargs):
    return dcc.Input(id={"type": "prop", "field": name}, value=value, debounce=False, **kwargs)


def prop_dropdown(name, value, options):
    return dcc.Dropdown(id={"type": "prop", "field": name}, value=value, options=options,
                        clearable=False, className="dash-dropdown")


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
        fields.append(field("Defined by", prop_dropdown("source", obj.source,
                            [{"label": "Initial state", "value": "state"},
                             {"label": "Halo family member", "value": "family"}])))
        fields.append(field("Family member", prop_dropdown("family_index", obj.family_index,
                            [{"label": label, "value": index} for index, label in enumerate(FAMILY_LABELS)]),
                            hint="from output/halo_family.npz"))
        fields.append(field("Propagation", prop_dropdown("propagation", obj.propagation,
                            [{"label": "Integrate initial state", "value": "integrate"},
                             {"label": "Repeat converged period (station-kept)", "value": "periodic"}])))
        fields.append(field("Initial state  x, y, z, vx, vy, vz",
                            prop_input("initial_state", ", ".join(f"{v:.10g}" for v in obj.initial_state),
                                       type="text", className="mono"),
                            hint="rotating frame, LU and LU/TU; used when defined by initial state"))
        fields.append(html.Div([field("Diameter [m]", prop_input("diameter_m", obj.diameter_m, type="number", min=0)),
                                field("Albedo", prop_input("albedo", obj.albedo, type="number", min=0, max=1, step=0.01))],
                               className="form-row"))
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

    return fields, False


# --------------------------------------------------------------------------
# Scenario edits: selection, add, remove, apply, settings, load
# --------------------------------------------------------------------------

@dash_app.callback(Output("scenario-store", "data"), Output("selected-store", "data"),
              Output("scenario-name", "value"), Output("scenario-epoch", "value"),
              Output("scenario-duration", "value"), Output("scenario-step", "value"),
              Input({"type": "tree-item", "name": ALL}, "n_clicks"),
              Input("add-button", "n_clicks"), Input("add-range-button", "n_clicks"),
              Input("remove-button", "n_clicks"),
              Input("apply-button", "n_clicks"), Input("load-upload", "contents"),
              Input("scenario-name", "value"), Input("scenario-epoch", "value"),
              Input("scenario-duration", "value"), Input("scenario-step", "value"),
              State("add-type", "value"), State("selected-store", "data"),
              State("range-from", "value"), State("range-to", "value"), State("range-step", "value"),
              State({"type": "prop", "field": ALL}, "value"), State({"type": "prop", "field": ALL}, "id"),
              State("scenario-store", "data"),
              prevent_initial_call=True)
def edit_scenario(tree_clicks, add_clicks, add_range_clicks, remove_clicks, apply_clicks, upload_contents,
                  name, epoch, duration, step, add_type, selected, range_from, range_to, range_step,
                  prop_values, prop_ids, scenario_data):
    trigger = ctx.triggered_id
    scenario = Scenario.from_dict(scenario_data)
    settings_unchanged = (no_update, no_update, no_update, no_update)

    if isinstance(trigger, dict) and trigger.get("type") == "tree-item":
        # A click on a tree row selects it; ignore the spurious trigger that
        # fires when rows are re-rendered with n_clicks = 0.
        if not any(clicks for clicks in tree_clicks):
            return (no_update, no_update) + settings_unchanged
        return no_update, trigger["name"], *settings_unchanged

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
        return scenario.to_dict(), new.name, *settings_unchanged

    if trigger == "add-range-button":
        first = int(np.clip(range_from or 0, 0, len(FAMILY) - 1))
        last = int(np.clip(range_to if range_to is not None else len(FAMILY) - 1, 0, len(FAMILY) - 1))
        stride = max(1, int(range_step or 1))
        new = None
        for index in range(first, last + 1, stride):
            new = scenario.add(Spacecraft(name=f"Halo #{index}", source="family", family_index=index,
                                          propagation="periodic"))
        return scenario.to_dict(), (new.name if new else no_update), *settings_unchanged

    if trigger == "remove-button":
        if selected:
            scenario.remove(selected)
        return scenario.to_dict(), None, *settings_unchanged

    if trigger == "apply-button":
        obj = scenario.find(selected)
        if obj is None:
            return (no_update, no_update) + settings_unchanged
        values = {prop_id["field"]: value for prop_id, value in zip(prop_ids, prop_values)}
        new_name = values.pop("name", obj.name) or obj.name
        for key, value in values.items():
            if key == "initial_state":
                try:
                    value = parse_state_text(value)
                except ValueError:
                    continue
            elif key in ("family_index",):
                value = int(value)
            elif isinstance(getattr(obj, key), float):
                if value is None:
                    continue
                value = float(value)
            setattr(obj, key, value)
        if new_name != obj.name:
            new_name = scenario.unique_name(new_name)
            scenario.rename(obj.name, new_name)
        return scenario.to_dict(), obj.name, *settings_unchanged

    if trigger == "load-upload":
        _, _, encoded = upload_contents.partition(",")
        loaded = Scenario.from_json(base64.b64decode(encoded).decode("utf-8"))
        return (loaded.to_dict(), None, loaded.name, loaded.epoch_utc,
                loaded.duration_days, loaded.time_step_s)

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
    return scenario.to_dict(), no_update, *settings_unchanged


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
    results = runner.run_scenario(scenario, FAMILY)
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

    status = f"{n_samples:,} samples, {len(scenario.spacecraft)} spacecraft, computed in {elapsed:.1f} s"
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
    return rows, [stats, html.Div(chips, className="constraints")]


# --------------------------------------------------------------------------
# 3D view, time series and slider
# --------------------------------------------------------------------------

@dash_app.callback(Output("view-3d", "figure"), Output("time-series", "figure"), Output("time-readout", "children"),
              Input("results-store", "data"), Input("pair-select", "value"),
              Input("time-slider", "value"), Input("view-select", "value"))
def update_views(run_id, pair, slider_index, view):
    if run_id not in RESULTS:
        figure_3d = figures.rotating_frame_figure({}, FIXED_POINTS, view=view)
        return figure_3d, figures.empty_time_series_figure(), ""

    scenario = RESULTS[run_id]["scenario"]
    results = RESULTS[run_id]["results"]
    index = int(np.clip(slider_index or 0, 0, len(results["times_s"]) - 1))
    current_time_s = results["times_s"][index]

    markers = {name: states[index] for name, states in results["trajectories"].items()}
    figure_3d = figures.rotating_frame_figure(results["trajectories"], FIXED_POINTS,
                                              station_positions=results["stations"],
                                              marker_states=markers, view=view)

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
    return figure_3d, figure_series, readout


if __name__ == "__main__":
    dash_app.run(debug=False, host="127.0.0.1", port=8050)
