from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import joblib

# Initialize Flask app
server = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset/student-mat.csv")

# Load trained model (Ensure 'model.pkl' is in the project folder)
model = joblib.load("model.pkl")

# Create Dash app inside Flask
app = dash.Dash(__name__, server=server, routes_pathname_prefix="/dashboard/")

# Dashboard Layout
app.layout = html.Div([
    html.H1("📊 Student Performance Prediction", style={"textAlign": "center", "color": "#2c3e50"}),

    html.Div([
        html.Label("Study Time (hours)"),
        dcc.Slider(id="study-time", min=1, max=10, step=0.5, value=5,
                   marks={i: str(i) for i in range(1, 11)}),
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.Label("Failures"),
        dcc.Dropdown(id="failures", options=[
            {"label": str(i), "value": i} for i in range(4)], value=0, clearable=False),
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.Label("Absences"),
        dcc.Input(id="absences", type="number", placeholder="Enter number of absences", value=2),
    ], style={"margin-bottom": "20px"}),

    html.Button("Predict Performance", id="predict-button", n_clicks=0, style={"backgroundColor": "#27ae60", "color": "white", "padding": "10px", "border": "none"}),

    html.H3(id="prediction-output", style={"color": "blue", "margin-top": "20px"}),

    html.Div([
        dcc.Graph(id="performance-graph"),
    ])
])


# Prediction Callback
@app.callback(
    [Output("prediction-output", "children"),
     Output("performance-graph", "figure")],
    Input("predict-button", "n_clicks"),
    State("study-time", "value"),
    State("failures", "value"),
    State("absences", "value")
)
def predict_performance(n_clicks, study_time, failures, absences):
    if n_clicks > 0:
        # Convert input into DataFrame
        input_data = pd.DataFrame([[study_time, failures, absences]],
                                  columns=["studytime", "failures", "absences"])
        # Predict using the model
        prediction = model.predict(input_data)[0]

        # Generate a graph comparing user input vs dataset
        filtered_df = df[(df["studytime"] == study_time)]
        fig = px.histogram(filtered_df, x="G3", nbins=10, title="Distribution of Final Grades (G3)",
                           labels={"G3": "Final Grade"}, color_discrete_sequence=["#3498db"])

        return f"Predicted Performance: {prediction}", fig

    return "", px.histogram(df, x="G3")


@server.route("/")
def index():
    return "Student Performance Prediction API is Running!"


if __name__ == "__main__":
    server.run(debug=True)
