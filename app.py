from flask import Flask, request, jsonify
import pandas as pd
import pickle

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
import os
import pandas as pd

# Get the absolute path of the CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "dataset", "student-mat.csv")

# Read the CSV file
df = pd.read_csv(csv_path, delimiter=";")



# Encode categorical variables
categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert G3 into classification target
df["G3_class"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
df.drop(columns=["G3"], inplace=True)

# Define features and target
X = df.drop(columns=["G3_class"])
y = df["G3_class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "student_performance_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Flask API for Deployment

app = Flask(__name__)

# Load the trained model and label encoders
with open('student_performance_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as encoders_file:
    label_encoders = pickle.load(encoders_file)

@app.route('/')
def home():
    return "Student Performance Prediction API is Running!"

if __name__ == '__main__':
    app.run(debug=True)
