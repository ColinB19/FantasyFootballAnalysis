from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from . import ids
from ..data.loader import DataSchema

# RECEIVER_DATA = pd.read_excel("./data/final_gbg_receiver_data.xlsx")

def render(app: Dash, data: pd.DataFrame) -> html.Div:
    all_teams = data[DataSchema.TEAM].unique().tolist()
    all_teams.sort()

    @app.callback(
        Output(ids.TEAM_DROPDOWN, "value"),
        Input(ids.SELECT_ALL_TEAMS_BUTTON, "n_clicks"),
    )
    def select_all_teams(_: int) -> list[str]:
        return all_teams

    return html.Div(
        children=[
            html.H6("TEAM"),
            dcc.Dropdown(
                id=ids.TEAM_DROPDOWN,
                options=[{"label": year, "value": year} for year in all_teams],
                value=all_teams,
                multi=True,
            ),
            html.Button(
                className="dropdown-button",
                children=["Select All"],
                id=ids.SELECT_ALL_TEAMS_BUTTON,
                n_clicks=0,
            ),
        ]
    )