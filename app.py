# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

sns.set_style("whitegrid")


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="UNSW-NB15 Intrusion Detection",
    page_icon="🔐",
    layout="wide"
)

# Title & Description
st.title("🔐 UNSW-NB15 Intrusion Detection System")
st.markdown(
    """
    Upload a test CSV file and select a trained model to evaluate performance.
    """
)

# Download Sample CSV
if os.path.exists("test_sample.csv"):
    with open("test_sample.csv", "rb") as f:
        st.download_button(
            label="📥 Download Sample Test CSV",
            data=f,
            file_name="test_sample.csv",
            mime="text/csv"
        )

# Model Selection
MODEL_PATH = "models"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

selected_model_name = st.selectbox(
    "Select Model",
    list(model_files.keys())
)

# Upload File
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# Model Evaluation
if uploaded_file is not None:

    try:
        # Load test data
        test_df = pd.read_csv(uploaded_file)

        if "label" not in test_df.columns:
            st.error("Uploaded file must contain 'label' column.")
            st.stop()

        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]

        # Load selected model
        model = joblib.load(os.path.join(MODEL_PATH, model_files[selected_model_name]))

        # Naive Bayes
        if selected_model_name == "Naive Bayes":
            X_test_processed = model["preprocessor"].transform(X_test)
            y_pred = model["model"].predict(X_test_processed.toarray())
            y_proba = model["model"].predict_proba(X_test_processed.toarray())[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display Metrics
        st.markdown("## 📊 Evaluation Metrics")
        
        metric_cols = st.columns(6)
        
        metric_cols[0].metric("Accuracy", f"{acc:.4f}")
        metric_cols[1].metric("AUC", f"{auc:.4f}")
        metric_cols[2].metric("Precision", f"{prec:.4f}")
        metric_cols[3].metric("Recall", f"{rec:.4f}")
        metric_cols[4].metric("F1 Score", f"{f1:.4f}")
        metric_cols[5].metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.markdown("## 🔢 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Normal (0)", "Attack (1)"],
            yticklabels=["Normal (0)", "Attack (1)"],
            ax=ax
        )
        
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("Actual Label", fontsize=12)
        ax.set_title(f"{selected_model_name} Confusion Matrix", fontsize=14)
        
        plt.tight_layout()
        
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
