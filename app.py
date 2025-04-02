from flask import Flask, redirect
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Initialize Flask app
server = Flask(__name__)

# Load dataset
DATASET_PATH = "dataset/student-mat.csv"
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH, delimiter=";")
    df = df.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)  # Encode categorical data
else:
    df = pd.DataFrame()
    print(f"⚠️ Warning: Dataset '{DATASET_PATH}' not found!")

# Define features and target
FEATURES = ["studytime", "failures", "absences", "G1", "G2", "schoolsup", "famsup"]
TARGET = "G3"

# Train model if dataset is available
MODEL_PATH = "student_performance_model.pkl"
if not df.empty:
    X = df[FEATURES]
    y = (df[TARGET] >= 10).astype(int)  # Convert G3 into binary classification (pass/fail)

    # Handle class imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomizedSearchCV for hyperparameter tuning
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": ["balanced", None]
    }

    rf_model = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(rf_model, param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_

    # Evaluate model performance
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    print("Optimal Threshold:", optimal_threshold)

    joblib.dump((model, optimal_threshold), MODEL_PATH)
    print("✅ Model trained and saved successfully.")
else:
    model, optimal_threshold = None, 0.5
    print("⚠️ Model training skipped due to missing dataset.")

# Load model
if os.path.exists(MODEL_PATH):
    model, optimal_threshold = joblib.load(MODEL_PATH)
else:
    print(f"⚠️ Warning: Model file '{MODEL_PATH}' not found!")
    model, optimal_threshold = None, 0.5

# Create Dash app inside Flask
app = dash.Dash(__name__, server=server, routes_pathname_prefix="/dashboard/")

# Dashboard Layout
app.layout = html.Div([
    html.H1("📊 Student Performance Prediction", style={"textAlign": "center", "color": "#2c3e50"}),

    html.Div([
        html.Label("Study Time (hours)"),
        dcc.Slider(id="study-time", min=1, max=10, step=0.5, value=5, marks={i: str(i) for i in range(1, 11)})
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.Label("Failures"),
        dcc.Dropdown(id="failures", options=[{"label": str(i), "value": i} for i in range(4)], value=0,
                     clearable=False),
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.Label("Absences"),
        dcc.Input(id="absences", type="number", placeholder="Enter number of absences", value=2)
    ], style={"margin-bottom": "20px"}),

    html.Button("Predict Performance", id="predict-button", n_clicks=0,
                style={"backgroundColor": "#27ae60", "color": "white", "padding": "10px", "border": "none",
                       "margin-top": "10px"}),

    html.H3(id="prediction-output", style={"color": "blue", "margin-top": "20px"}),

    html.Div([
        dcc.Graph(id="performance-graph"),
    ])
])

# Prediction Function
def predict_performance(study_time, failures, absences):
    if model is None:
        return "Error: Model not found. Please retrain the model."

    try:
        input_data = pd.DataFrame([{"studytime": study_time, "failures": failures, "absences": absences,
                                    "G1": 10, "G2": 10, "schoolsup": 0, "famsup": 1}])  # Default values
        prediction_prob = model.predict_proba(input_data)[0][1]
        prediction = "Pass" if prediction_prob >= optimal_threshold else "Fail"
        return prediction
    except Exception as e:
        return f"Prediction Error: {str(e)}"

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
        prediction_text = f"Predicted Performance: {prediction}"

        fig = px.histogram(df, x="G3", title="Distribution of Final Grades") if not df.empty else go.Figure()
        return prediction_text, fig

    return "", go.Figure()

@server.route("/")
def index():
    return redirect("/dashboard/")

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
