
import io
import streamlit as st
import pandas as pd
from pathlib import Path
from tabpfn.model_loading import load_fitted_tabpfn_model

# Project paths
project_path = Path(__file__).resolve().parent.parent
dataset_path = project_path / "Dataset" / "student-mat.csv"
model_path = project_path / "Model" / "student_performance_tabpfn.tabpfn_fit"

# Permanent prediction history file
history_path = project_path / "Dataset" / "prediction_history.csv"

# Page configuration
st.set_page_config(
    page_title="AI Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load dataset
df = pd.read_csv(dataset_path, sep=";")
X = df.drop(columns=["G2", "G3"])


# Load prediction history from CSV
def load_history():

    if history_path.exists():
        return pd.read_csv(history_path)

    return pd.DataFrame()


# Save new predictions permanently
def save_history(new_records):

    new_df = pd.DataFrame(new_records)

    if history_path.exists():
        old_df = pd.read_csv(history_path)
        combined_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )
    else:
        combined_df = new_df

    combined_df.to_csv(history_path, index=False)


# Load saved model
@st.cache_resource
def load_model():
    return load_fitted_tabpfn_model(model_path)


# Validate student data
def validate_data(input_data):

    errors = []
    clean_data = input_data.copy()

    for feature in X.columns:

        if clean_data[feature].isnull().any():
            errors.append(
                f"{feature}: Contains missing values."
            )
            continue

        if pd.api.types.is_numeric_dtype(X[feature]):

            converted = pd.to_numeric(
                clean_data[feature],
                errors="coerce"
            )

            if converted.isnull().any():
                errors.append(
                    f"{feature}: Must contain valid numbers."
                )
                continue

            minimum = X[feature].min()
            maximum = X[feature].max()

            if (
                (converted < minimum).any()
                or (converted > maximum).any()
            ):
                errors.append(
                    f"{feature}: Values must be between "
                    f"{minimum} and {maximum}."
                )

            clean_data[feature] = converted

        else:

            valid_options = X[feature].dropna().unique()

            invalid_values = (
                set(clean_data[feature].unique())
                - set(valid_options)
            )

            if invalid_values:
                errors.append(
                    f"{feature}: Invalid values found: "
                    f"{invalid_values}"
                )

    return clean_data, errors


# Sidebar navigation
st.sidebar.title("🎓 Student Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Individual Prediction",
        "Bulk Prediction",
        "Dashboard",
        "Prediction History"
    ]
)


# HOME PAGE
if page == "Home":

    st.title("🎓 AI-Powered Student Performance Prediction")

    st.write(
        "Predict student final grades using the pretrained "
        "TabPFN Machine Learning model."
    )

    st.divider()

    st.header("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Input Features", len(X.columns))
    col3.metric("Prediction Model", "TabPFN")

    st.divider()

    st.header("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "1.52")
    col2.metric("RMSE", "2.10")
    col3.metric("R² Score", "0.78")

    st.divider()

    st.header("ℹ️ About This Project")

    st.write("""
    This application predicts student final grades using
    academic and personal information.

    **Available Features:**
    - Individual student prediction
    - Bulk CSV and Excel predictions
    - Prediction dashboard
    - Downloadable results
    - Input data validation
    - Permanent prediction history
    """)


# INDIVIDUAL PREDICTION PAGE
elif page == "Individual Prediction":

    st.title("👨‍🎓 Individual Student Prediction")

    st.write(
        "Enter the student's details below to estimate "
        "their final grade."
    )

    with st.form("student_prediction_form"):

        student_data = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(X.columns):

            current_col = col1 if i % 2 == 0 else col2

            with current_col:

                if pd.api.types.is_numeric_dtype(X[feature]):

                    minimum = int(X[feature].min())
                    maximum = int(X[feature].max())
                    default = int(X[feature].median())

                    student_data[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1
                    )

                else:

                    options = sorted(
                        X[feature].dropna().unique().tolist()
                    )

                    student_data[feature] = st.selectbox(
                        feature,
                        options
                    )

        submitted = st.form_submit_button(
            "Predict Final Grade",
            use_container_width=True
        )

    if submitted:

        try:

            input_df = pd.DataFrame(
                [student_data],
                columns=X.columns
            )

            clean_data, errors = validate_data(input_df)

            if errors:

                for error in errors:
                    st.error(error)

            else:

                model = load_model()
                prediction = model.predict(clean_data)

                predicted_grade = round(float(prediction[0]), 2)

                st.success(
                    "Prediction completed successfully!"
                )

                st.metric(
                    "Predicted Final Grade",
                    f"{predicted_grade:.2f} / 20"
                )

                st.info(
                    "This is an estimated grade generated by "
                    "the TabPFN model, not a guaranteed final result."
                )

                # Save individual prediction permanently
                history_record = student_data.copy()
                history_record["Prediction Type"] = "Individual"
                history_record["Predicted_G3"] = predicted_grade

                save_history([history_record])

                st.success(
                    "Prediction saved to permanent history!"
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# BULK PREDICTION PAGE
elif page == "Bulk Prediction":

    st.title("📁 Bulk Student Prediction")

    st.write(
        "Upload a CSV or Excel file containing student details "
        "to predict grades for multiple students."
    )

    # Download sample template
    st.subheader("Download Sample Template")

    template_data = X.head(5).to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Sample CSV Template",
        data=template_data,
        file_name="sample_student_template.csv",
        mime="text/csv"
    )

    st.divider()

    # Upload dataset
    st.subheader("Upload Student Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)

            st.subheader("Uploaded Dataset")
            st.dataframe(uploaded_df.head())

            # Check required columns
            missing_columns = [
                col for col in X.columns
                if col not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            elif uploaded_df.empty:

                st.error(
                    "The uploaded file contains no student records."
                )

            else:

                st.info(
                    f"Total student records uploaded: {len(uploaded_df)}"
                )

                if st.button("Predict All Student Grades"):

                    input_data = uploaded_df[X.columns].copy()

                    clean_data, errors = validate_data(input_data)

                    if errors:

                        st.error(
                            "Please correct the following data errors:"
                        )

                        for error in errors:
                            st.warning(error)

                    else:

                        model = load_model()
                        predictions = model.predict(clean_data)

                        result_df = uploaded_df.copy()

                        result_df["Predicted_G3"] = predictions.round(2)

                        st.success(
                            "Bulk predictions completed successfully!"
                        )

                        st.subheader("Prediction Results")
                        st.dataframe(result_df)

                        # Save bulk predictions permanently
                        history_df = result_df.copy()
                        history_df["Prediction Type"] = "Bulk"

                        save_history(
                            history_df.to_dict("records")
                        )

                        st.success(
                            "Bulk predictions saved to permanent history!"
                        )

                        # Download CSV
                        csv_data = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv_data,
                            file_name="student_predictions.csv",
                            mime="text/csv"
                        )

                        # Download Excel
                        excel_buffer = io.BytesIO()

                        with pd.ExcelWriter(
                            excel_buffer,
                            engine="openpyxl"
                        ) as writer:

                            result_df.to_excel(
                                writer,
                                index=False,
                                sheet_name="Predictions"
                            )

                        st.download_button(
                            label="Download Predictions Excel",
                            data=excel_buffer.getvalue(),
                            file_name="student_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        except Exception as e:

            st.error(f"Error processing file: {e}")


# DASHBOARD PAGE
elif page == "Dashboard":

    st.title("📊 Student Performance Dashboard")

    st.write(
        "Explore student academic performance through "
        "interactive charts and statistics."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(df))
    col2.metric(
        "Average Final Grade",
        round(df["G3"].mean(), 2)
    )
    col3.metric("Highest Grade", int(df["G3"].max()))
    col4.metric("Lowest Grade", int(df["G3"].min()))

    st.divider()

    st.subheader("1. Final Grade Distribution")

    grade_counts = df["G3"].value_counts().sort_index()
    st.bar_chart(grade_counts)

    st.subheader("2. Study Time vs Average Final Grade")

    study_performance = df.groupby("studytime")["G3"].mean()
    st.bar_chart(study_performance)

    st.subheader("3. First Period Grade vs Final Grade")

    st.scatter_chart(
        df,
        x="G1",
        y="G3"
    )

    st.subheader("4. Average Final Grade by Gender")

    gender_performance = df.groupby("sex")["G3"].mean()
    st.bar_chart(gender_performance)

    st.success("Dashboard loaded successfully!")


# PREDICTION HISTORY PAGE
elif page == "Prediction History":

    st.title("📜 Prediction History")

    st.write(
        "View individual and bulk student predictions. "
        "Your history is saved permanently in a CSV file."
    )

    st.divider()

    # Read permanent history
    history_df = load_history()

    if not history_df.empty:

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.subheader("Previous Predictions")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # Download prediction history
        history_csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction History",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        st.divider()

        if st.button("Clear Prediction History"):

            if history_path.exists():
                history_path.unlink()

            st.success("Prediction history cleared!")

            st.rerun()

    else:

        st.info(
            "No prediction history available yet. "
            "Make an individual or bulk prediction first."
        )
