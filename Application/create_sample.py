
import pandas as pd
from pathlib import Path

# Locate project folder
project_path = Path(__file__).resolve().parent.parent

# Load original dataset
df = pd.read_csv(
    project_path / "Dataset" / "student-mat.csv",
    sep=";"
)

# Select 5 students and remove G2 and G3
sample = df.drop(columns=["G2", "G3"]).head(5)

# Save sample CSV
output_path = project_path / "Dataset" / "sample_students.csv"
sample.to_csv(output_path, index=False)

print("Sample CSV created successfully!")
print("Number of students:", len(sample))
print("Number of input features:", len(sample.columns))
print("Saved at:", output_path)
