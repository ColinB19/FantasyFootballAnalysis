import dash
from dash import Dash, html, dcc

from dash_bootstrap_components.themes import BOOTSTRAP

# from src.components.layout import create_layout
# from src.data.loader import load_fantasy_data

PATH = "./data/final_gbg_rr.xlsx"

def load_fantasy_data(path:str) -> pd.DataFrame:
    # load the xlsx file
    data = pd.read_excel(path)
    return data

app = Dash(__name__, use_pages=True)

app.layout = html.Div(
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
    )

# TODO: Get this into the layout file and figure out how to pass teh data to it

if __name__ == "__main__":
    data = load_fantasy_data(PATH)
    # app.layout = create_layout(app, data)
    app.run(debug=True)
