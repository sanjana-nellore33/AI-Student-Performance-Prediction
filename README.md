
# AI-Powered Student Performance Prediction

## Project Overview
This project predicts students' final academic grades using a pretrained TabPFN machine learning model.

The application is developed using Python and Streamlit with the UCI Student Performance dataset.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TabPFN
- Matplotlib
- Seaborn

## Dataset
UCI Student Performance Dataset

- Total records: 395
- Total columns: 33
- Input features: 31
- Target variable: G3 (Final Grade)

## Main Features
1. Individual Student Prediction
2. Bulk Student Prediction using CSV and Excel
3. Student Performance Dashboard
4. Input Data Validation
5. Prediction History with permanent CSV storage
6. Downloadable Prediction Results

## Model Performance

| Metric | Result |
|---|---:|
| MAE | 1.52 |
| RMSE | 2.10 |
| R2 Score | 0.78 |

## Project Structure

- Dataset – Dataset files and prediction history
- Notebooks – Experiment notebooks
- Model – Trained TabPFN model
- Application – Streamlit application and training scripts
- Screenshots – Project screenshots
- Documentation – Project documentation
- requirements.txt – Required Python libraries

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Run the application:

   streamlit run Application/app.py

3. Open the local URL displayed in the terminal.

## Expected Output
The application predicts estimated student final grades and provides dashboard visualizations and prediction history.
