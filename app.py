import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing ML App", layout="centered")

st.title("Bank Marketing Classification App")

# ===============================
# Feature order used during training
# ===============================
FEATURE_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default', 'balance',
    'housing', 'loan', 'contact', 'day_of_week', 'month',
    'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

uploaded_file = st.file_uploader(
    "Upload Test CSV File (with or without target column y)",
    type=["csv"]
)

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Check for target column
    # -------------------------------
    if 'y' in df.columns:
        y_true = df['y']
        X = df.drop(columns=['y'])
        has_target = True
    else:
        X = df.copy()
        has_target = False

    # -------------------------------
    # Ensure correct feature order
    # -------------------------------
    try:
        X = X[FEATURE_COLUMNS]
    except KeyError:
        st.error(
            "Uploaded CSV does not match expected feature columns.\n"
            "Please upload a CSV with the same features used during training."
        )
        st.stop()

    # -------------------------------
    # Load scaler and apply when needed
    # -------------------------------
    scaler = joblib.load("models/scaler.pkl")

    if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X = scaler.transform(X)

    # -------------------------------
    # Load selected model
    # -------------------------------
    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Decision Tree": "models/decision_tree.pkl",
        "KNN": "models/knn.pkl",
        "Naive Bayes": "models/naive_bayes.pkl",
        "Random Forest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl"
    }

    model = joblib.load(model_files[model_choice])
    y_pred = model.predict(X)

    # -------------------------------
    # Output
    # -------------------------------
    if has_target:
        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.subheader("Predictions (first 20 rows)")
        output_df = X.copy()
        output_df["prediction"] = y_pred
        st.dataframe(output_df.head(20))
