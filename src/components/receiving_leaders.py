import plotly.express as px
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from . import ids
from ..data.loader import DataSchema

# RECEIVER_DATA = pd.read_excel("./data/final_gbg_receiver_data.xlsx")

def render(app: Dash, data: pd.DataFrame) -> html.Div:
    @app.callback(
        Output(ids.REC_LEADERS, "children"),
        [Input(ids.TEAM_DROPDOWN, "value"), Input(ids.POSITION_DROPDOWN, "value")],
    )
    def update_bar_chart(teams: list[str], positions: list[str]) -> html.Div:
        filtered_data = (
            data[(data.team.isin(teams)) & (data.position.isin(positions))]
            .groupby([DataSchema.PLAYER_ID, DataSchema.PLAYER_NAME, DataSchema.TEAM])
            .agg(
                receiving_yards=(DataSchema.RECEIVING_YARDS, "sum"),
                receptions=(DataSchema.RECEPTIONS, "sum"),
                targets=(DataSchema.TOTAL_TARGETS, "sum"),
            )
            .reset_index()
            .sort_values(DataSchema.RECEIVING_YARDS, ascending=False)
            .iloc[:20]
        )

        if filtered_data.shape[0] == 0:
            return html.Div("No data selected.", id=ids.REC_LEADERS)

        fig = px.bar(
            filtered_data,
            x=DataSchema.PLAYER_NAME,
            y=DataSchema.RECEIVING_YARDS,
            hover_data=[DataSchema.RECEPTIONS, "targets"],
            color=DataSchema.RECEPTIONS,
            text=DataSchema.RECEIVING_YARDS,
        )

        return html.Div(dcc.Graph(figure=fig), id=ids.REC_LEADERS)

    return html.Div(id=ids.REC_LEADERS)
