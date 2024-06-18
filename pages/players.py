import dash
from dash import html

from ..src.components.layout import player_layouts

dash.register_page(__name__)

layout = player_layout(app, data)