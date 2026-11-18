
import pandas as pd
from pathlib import Path
from tabpfn.model_loading import load_fitted_tabpfn_model

# Get project folder
project_path = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(project_path / "Dataset" / "student-mat.csv", sep=";")

# Prepare input features
X = df.drop(columns=["G2", "G3"])

# Load saved model
model_path = project_path / "Model" / "student_performance_tabpfn.tabpfn_fit"

model = load_fitted_tabpfn_model(model_path)

print("Saved model loaded successfully!")

# Predict the grade of one student
prediction = model.predict(X.iloc[[0]])

print("\nPredicted final grade:", round(float(prediction[0]), 2))
print("Actual final grade:", df["G3"].iloc[0])
