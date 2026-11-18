
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import save_fitted_tabpfn_model

# Load dataset
project_path = Path(__file__).resolve().parent.parent
file_path = project_path / "Dataset" / "student-mat.csv"

df = pd.read_csv(file_path, sep=";")

# Separate input features and target
X = df.drop(columns=["G2", "G3"])
y = df["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
print("Training TabPFN model...")

model = TabPFNRegressor()
model.fit(X_train, y_train)

print("Model training completed!")

# Save fitted model
model_path = project_path / "Model" / "student_performance_tabpfn.tabpfn_fit"

save_fitted_tabpfn_model(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))
