"""
Dash interface for the cislunar mission tool.  Run from the repository
root:  python -m mission.app

Callbacks here only shuffle data between the scenario store, the
analysis functions and the figure builders; no physics lives in this
file.
"""

import numpy as np
from dash import Dash, dcc, html, Input, Output, State

from mission import analysis, figures
from mission.scenario import Scenario, example_scenario

app = Dash(__name__)
app.title = "Cislunar mission tool"

app.layout = html.Div([
    dcc.Store(id="scenario-store", data=example_scenario().to_dict()),
    html.Div([
        html.H3("Cislunar mission tool", style={"margin": "4px 0"}),
        html.Button("Propagate", id="propagate-button", n_clicks=0),
    ], style={"padding": "6px 10px", "borderBottom": "1px solid #ccc"}),
    dcc.Graph(id="view-3d", style={"height": "80vh"}),
])


@app.callback(Output("view-3d", "figure"),
              Input("propagate-button", "n_clicks"),
              State("scenario-store", "data"))
def update_view(n_clicks, scenario_data):
    scenario = Scenario.from_dict(scenario_data)
    times_nondim = scenario.time_grid_nondim()
    family = analysis.load_family()
    trajectories = {}
    for spacecraft in scenario.spacecraft:
        trajectories[spacecraft.name] = analysis.propagate_spacecraft(spacecraft, times_nondim, family)
    return figures.rotating_frame_figure(trajectories, analysis.fixed_geometry())


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
