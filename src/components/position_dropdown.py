from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from . import ids
from ..data.loader import DataSchema

# RECEIVER_DATA = pd.read_excel("./data/final_gbg_receiver_data.xlsx")

def render(app: Dash, data: pd.DataFrame) -> html.Div:
    all_positions = data[DataSchema.POSITION].unique().tolist()
    all_positions.sort()

    @app.callback(
        Output(ids.POSITION_DROPDOWN, "value"),
        Input(ids.SELECT_ALL_POSITIONS_BUTTON, "n_clicks"),
    )
    def select_all_positions(_: int) -> list[str]:
        return all_positions

    return html.Div(
        children=[
            html.H6("POSITION"),
            dcc.Dropdown(
                id=ids.POSITION_DROPDOWN,
                options=[{"label": pos, "value": pos} for pos in all_positions],
                value=all_positions,
                multi=True,
            ),
            html.Button(
                className="dropdown-button",
                children=["Select All"],
                id=ids.SELECT_ALL_POSITIONS_BUTTON,
                n_clicks=0,
            ),
        ]
    )