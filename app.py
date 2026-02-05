# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils import (
    clean_text,
    train_model,
    predict,
    apply_forgetting_rule,
    get_top_features,
    confidence_overlap
)

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(
    page_title="Selective Machine Unlearning",
    layout="wide"
)

st.title("🔐 Selective Machine Unlearning Framework")
st.caption("Unlearning • Verification • Explainability • Trust")

st.divider()

# -------------------------------------------------
# Dataset Upload
# -------------------------------------------------
st.header("📂 Step 1: Upload Dataset")
file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset loaded")

    st.dataframe(df.head())

    st.divider()

    # -------------------------------------------------
    # Column Selection
    # -------------------------------------------------
    st.header("🧩 Step 2: Column Selection")

    user_col = st.selectbox("User ID Column", df.columns)
    text_col = st.selectbox("Text Column", df.columns)
    label_col = st.selectbox("Label Column (0 / 1)", df.columns)

    df = df[[user_col, text_col, label_col]].dropna()
    df["clean_text"] = df[text_col].apply(clean_text)

    st.divider()

    # -------------------------------------------------
    # Forgetting Rule Selection
    # -------------------------------------------------
    st.header("🧠 Step 3: Forgetting Rule")

    rule = st.selectbox(
        "Choose Forgetting Rule",
        ["User-based", "Label-based (Selective)", "Keyword-based"]
    )

    target_users = st.multiselect(
        "Select User(s) to Forget",
        df[user_col].unique()
    )

    keyword = None
    if rule == "Keyword-based":
        keyword = st.text_input("Keyword to Forget", value="bad")

    st.divider()

    # -------------------------------------------------
    # Run Unlearning
    # -------------------------------------------------
    if st.button("🚀 Run Unlearning", use_container_width=True):

        if not target_users:
            st.warning("Select at least one user")
            st.stop()

        # ==============================
        # Layer 1 + 2: Before Unlearning
        # ==============================
        model_before, tfidf_before = train_model(
            df, "clean_text", label_col
        )

        user_data = df[df[user_col].isin(target_users)]

        preds_before, probs_before = predict(
            model_before,
            tfidf_before,
            user_data["clean_text"]
        )

        acc_before = accuracy_score(
            df[label_col],
            model_before.predict(
                tfidf_before.transform(df["clean_text"])
            )
        )

        # ==============================
        # Layer 1: Apply Unlearning
        # ==============================
        df_after = apply_forgetting_rule(
            df,
            rule,
            user_col,
            target_users,
            "clean_text",
            label_col,
            keyword
        )

        if df_after.empty:
            st.error("All data removed. Try different rule.")
            st.stop()

        # ==============================
        # Layer 2: After Unlearning
        # ==============================
        model_after, tfidf_after = train_model(
            df_after, "clean_text", label_col
        )

        preds_after, probs_after = predict(
            model_after,
            tfidf_after,
            user_data["clean_text"]
        )

        acc_after = accuracy_score(
            df_after[label_col],
            model_after.predict(
                tfidf_after.transform(df_after["clean_text"])
            )
        )

        # ==============================
        # Layer 3: Forgetting Score
        # ==============================
        pred_change = np.mean(preds_before != preds_after)
        conf_change = np.mean(np.abs(probs_before - probs_after))
        forgetting_score = 0.5 * pred_change + 0.5 * conf_change

        # ==============================
        # Layer 4: Explainability
        # ==============================
        top_pos_before, top_neg_before = get_top_features(
            model_before, tfidf_before
        )
        top_pos_after, top_neg_after = get_top_features(
            model_after, tfidf_after
        )

        # ==============================
        # Layer 5: Trust / Re-ID Resistance
        # ==============================
        trust_score = confidence_overlap(
            probs_before, probs_after
        )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------
        st.divider()
        st.header("📊 Results")

        # Accuracy Table
        st.subheader("Model Accuracy")
        st.table(pd.DataFrame({
            "Stage": ["Before Unlearning", "After Unlearning"],
            "Accuracy": [round(acc_before, 4), round(acc_after, 4)]
        }))

        # Scores
        c1, c2 = st.columns(2)
        c1.metric("Forgetting Score", round(forgetting_score, 3))
        c2.metric("Trust / Privacy Score", round(trust_score, 3))

        # Explainability
        st.subheader("Explainability (Feature-Level)")
        col1, col2 = st.columns(2)

        col1.write("Top Positive Words (Before)")
        col1.write(list(top_pos_before))

        col2.write("Top Positive Words (After)")
        col2.write(list(top_pos_after))

        # Verification Plots
        st.subheader("Behavioral Verification")

        st.line_chart(pd.DataFrame({
            "Prediction Before": preds_before,
            "Prediction After": preds_after
        }))

        st.line_chart(pd.DataFrame({
            "Confidence Before": probs_before,
            "Confidence After": probs_after
        }))

        # Summary
        st.subheader("Unlearning Summary")
        st.table(pd.DataFrame({
            "Metric": ["Total Samples", "Remaining Samples", "Forgotten Samples"],
            "Value": [
                df.shape[0],
                df_after.shape[0],
                df.shape[0] - df_after.shape[0]
            ]
        }))
