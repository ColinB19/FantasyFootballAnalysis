from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from . import ids
from ..data.loader import DataSchema
from collections import OrderedDict

# RECEIVER_DATA = pd.read_excel("./data/final_gbg_receiver_data.xlsx")


def render(app: Dash, data: pd.DataFrame) -> html.Div:
    all_players = OrderedDict(
        sorted(
            pd.Series(data[DataSchema.PLAYER_NAME], index=data[DataSchema.PLAYER_ID])
            .to_dict()
            .items()
        )
    )

    # @app.callback(
    #     Output(ids.POSITION_DROPDOWN, "value")
    # )
    # def select_all_players(_: int) -> list[str]:
    #     return all_players

    return html.Div(
        children=[
            html.H6("PLAYER"),
            dcc.Dropdown(
                id=ids.PLAYER_DROPDOWN,
                options=[{"label": name, "value": pid} for pid, name in all_players.items()],
                value=all_players[next(iter(all_players))],
                multi=False,
            ),
        ]
    )
