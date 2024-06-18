from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

from src.components.layout import create_layout
from src.data.loader import load_fantasy_data


# walkthrough: https://www.youtube.com/watch?v=GlRauKqI08Y
PATH = "./data/final_gbg_rr.xlsx"

def main() -> None:
    data = load_fantasy_data(PATH)
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Fantasy Football Dashboard"
    app.layout = create_layout(app, data)
    app.run()


if __name__ == "__main__":
    main()