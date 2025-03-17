from flask import Flask, redirect
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import joblib
import os

# Initialize Flask app
server = Flask(__name__)

# Load dataset (Ensure correct file path and delimiter)
DATASET_PATH = "dataset/student-mat.csv"
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH, delimiter=";")
else:
    df = pd.DataFrame()  # Empty DataFrame to prevent errors
    print(f"⚠️ Warning: Dataset file '{DATASET_PATH}' not found!")

# Load trained model safely
model_path = os.path.join(os.path.dirname(__file__), "student_performance_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print(f"⚠️ Warning: Model file '{model_path}' not found!")

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
        dcc.Dropdown(id="failures", options=[{"label": str(i), "value": i} for i in range(4)],
                     value=0, clearable=False),
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.Label("Absences"),
        dcc.Input(id="absences", type="number", placeholder="Enter number of absences", value=2),
    ], style={"margin-bottom": "20px"}),

    html.Button("Predict Performance", id="predict-button", n_clicks=0,
                style={"backgroundColor": "#27ae60", "color": "white", "padding": "10px", "border": "none"}),

    html.H3(id="prediction-output", style={"color": "blue", "margin-top": "20px"}),

    html.Div([
        dcc.Graph(id="performance-graph"),
    ])
])

# Prediction Function
def predict_performance(study_time, failures, absences):
    if model is None:
        return "Error: Model not found. Please upload 'student_performance_model.pkl'."

    try:
        # Ensure all required features are included
        input_data = {col: 0 for col in model.feature_names_in_}  # Default values

        # Update only the known values
        input_data["studytime"] = study_time
        input_data["failures"] = failures
        input_data["absences"] = absences

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make Prediction
        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# Prediction Callback
@app.callback(
    [Output("prediction-output", "children"),
     Output("performance-graph", "figure")],
    Input("predict-button", "n_clicks"),
    State("study-time", "value"),
    State("failures", "value"),
    State("absences", "value")
)
def update_output(n_clicks, study_time, failures, absences):
    if n_clicks > 0:
        prediction = predict_performance(study_time, failures, absences)
        prediction_text = f"Predicted Final Grade: {prediction}"

        # Plot histogram only if dataset is available
        if not df.empty:
            fig = px.histogram(df, x="G3", title="Distribution of Final Grades")
        else:
            fig = px.Figure()

        return prediction_text, fig

    return "", px.Figure()

# Redirect root to dashboard
@server.route("/")
def index():
    return redirect("/dashboard/")

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
