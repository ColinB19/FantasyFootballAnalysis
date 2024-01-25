from dash import Dash, html, dcc, page_container, page_registry
import pandas as pd

from . import receiving_leaders, team_dropdown, position_dropdown


def create_layout(app: Dash, data:pd.DataFrame) -> html.Div:
    return html.Div(
        className="app-div",
        children=[
            html.H1("Multi-page app with Dash Pages"),
            html.Div(
                [
                    html.Div(
                        dcc.Link(
                            f"{page['name']} - {page['path']}", 
                            href=page["relative_path"]
                        )
                    )
                    for page in page_registry.values()
                ]
            ),
            page_container,
            # html.Hr(),
            # html.Div(
            #     className="dropdown-container",
            #     children=[
            #         team_dropdown.render(app, data),
            #         position_dropdown.render(app, data),
            #     ],
            # ),
            # receiving_leaders.render(app, data),
        ]
    )
    # return html.Div(
    #     className="app-div",
    #     children=[
    #         html.H1(app.title),
    #         html.Hr(),
    #         html.Div(
    #             className="dropdown-container",
    #             children=[
    #                 team_dropdown.render(app, data),
    #                 position_dropdown.render(app, data),
    #             ],
    #         ),
    #         receiving_leaders.render(app, data),
    #     ],
    # )

def home_layout(app: Dash, data:pd.DataFrame) -> html.Div:
     return html.Div(
        className="app-div",
        children=[html.Hr(),
            html.Div(
                className="dropdown-container",
                children=[
                    team_dropdown.render(app, data),
                    position_dropdown.render(app, data),
                ],
            ),
            receiving_leaders.render(app, data)
        ])


def player_layout(app: Dash, data:pd.DataFrame) -> html.Div:
    return html.Div(
        className="app-div",
        children=[
            html.H1(app.title),
            html.Hr(),
            html.Div(
                className="dropdown-container",
                children=[
                    player_dropdown.render(app, data)
                ],
            ),
            # receiving_leaders.render(app, data), some other viz
        ],
    )