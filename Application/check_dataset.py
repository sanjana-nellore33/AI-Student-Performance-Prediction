
from pathlib import Path
import pandas as pd

# Locate the dataset
file_path = Path(__file__).resolve().parent.parent / "Dataset" / "student-mat.csv"

# Load the dataset
df = pd.read_csv(file_path, sep=";")

# Display dataset details
print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())
