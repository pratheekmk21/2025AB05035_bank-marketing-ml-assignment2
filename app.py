import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Bank Marketing Classification App")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

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

    if 'y' in df.columns:
    y_true = df['y']
    X = df.drop('y', axis=1)
    has_target = True
else:
    X = df
    has_target = False

    scaler = joblib.load("models/scaler.pkl")

    if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X = scaler.transform(X)

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

if has_target:
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
else:
    st.subheader("Predictions")
    df['prediction'] = y_pred
    st.dataframe(df.head(20))
