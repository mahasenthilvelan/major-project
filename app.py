# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils import clean_text, train_model, predict, apply_forgetting_rule


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Selective Machine Unlearning",
    layout="wide"
)

st.title("🔐 Selective Machine Unlearning Framework")
st.caption("Rule-based • Verifiable • Explainable • Privacy-aware")

st.divider()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Framework Layers")
st.sidebar.markdown("""
1. Unlearning Engine  
2. Verification Layer  
3. Forgetting Score  
4. Explainability  
5. Trust / Re-ID Resistance  
""")

st.sidebar.info(
    "This system demonstrates selective machine unlearning "
    "without damaging overall model performance."
)


# =========================================================
# STEP 1: DATA UPLOAD
# =========================================================
st.header("📂 Step 1: Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded")

    with st.expander("Preview Dataset"):
        st.dataframe(df.head())

    st.divider()

    # =====================================================
    # STEP 2: COLUMN SELECTION
    # =====================================================
    st.header("🧩 Step 2: Column Selection")

    col1, col2, col3 = st.columns(3)
    with col1:
        user_col = st.selectbox("User ID Column", df.columns)
    with col2:
        text_col = st.selectbox("Text Column", df.columns)
    with col3:
        label_col = st.selectbox("Label Column (0/1)", df.columns)

    df = df[[user_col, text_col, label_col]].dropna()
    df["clean_text"] = df[text_col].apply(clean_text)

    st.divider()

    # =====================================================
    # STEP 3: FORGETTING RULE
    # =====================================================
    st.header("🧠 Step 3: Forgetting Rule")

    rule = st.selectbox(
        "Choose forgetting rule",
        ["User-based", "Label-based (Selective)", "Keyword-based"]
    )

    target_users = st.multiselect(
        "Select user(s) to forget",
        df[user_col].unique()
    )

    keyword = None
    if rule == "Keyword-based":
        keyword = st.text_input("Keyword to forget")

    if rule == "Label-based (Selective)":
        st.info(
            "Only samples of selected users with the chosen label are removed. "
            "This ensures fine-grained and safe unlearning."
        )

    st.divider()

    # =====================================================
    # STEP 4: RUN UNLEARNING
    # =====================================================
    if st.button("🚀 Run Unlearning", use_container_width=True):

        if not target_users:
            st.warning("Please select at least one user")
            st.stop()

        progress = st.progress(0)

        # ---------------- BEFORE ----------------
        progress.progress(20)
        model_before, tfidf_before = train_model(df, "clean_text", label_col)

        user_data = df[df[user_col].isin(target_users)]
        preds_b, probs_b = predict(
            model_before, tfidf_before, user_data["clean_text"]
        )

        acc_before = accuracy_score(
            df[label_col],
            model_before.predict(tfidf_before.transform(df["clean_text"]))
        )

        # ---------------- APPLY FORGETTING ----------------
        progress.progress(50)
        df_after = apply_forgetting_rule(
            df,
            rule,
            user_col,
            target_users,
            label_col,
            keyword
        )

        if df_after.empty:
            st.error("All data removed. Change forgetting rule.")
            st.stop()

        # ---------------- AFTER ----------------
        progress.progress(80)
        model_after, tfidf_after = train_model(df_after, "clean_text", label_col)

        preds_a, probs_a = predict(
            model_after, tfidf_after, user_data["clean_text"]
        )

        acc_after = accuracy_score(
            df_after[label_col],
            model_after.predict(tfidf_after.transform(df_after["clean_text"]))
        )

        progress.progress(100)
        st.success("Unlearning completed")

        st.divider()

        # =====================================================
        # RESULTS
        # =====================================================
        st.header("📊 Results Dashboard")

        # Accuracy table
        st.subheader("Model Accuracy")
        st.table(pd.DataFrame({
            "Stage": ["Before Unlearning", "After Unlearning"],
            "Accuracy": [round(acc_before, 4), round(acc_after, 4)]
        }))

        # Forgetting metrics
        pred_change = np.mean(preds_b != preds_a)
        conf_change = np.mean(np.abs(probs_b - probs_a))
        forgetting_score = 0.5 * pred_change + 0.5 * min(conf_change, 1)

        trust_score = min(
            max(np.var(probs_a) - np.var(probs_b) + 0.5, 0),
            1
        )

        # Gauge function
        def draw_gauge(score, title):
            fig, ax = plt.subplots(figsize=(4, 2.5))
            theta = np.linspace(0, np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), color="lightgray", linewidth=12)

            filled = np.linspace(0, score * np.pi, 100)
            color = "red" if score < 0.4 else "orange" if score < 0.7 else "green"
            ax.plot(np.cos(filled), np.sin(filled), color=color, linewidth=12)

            ax.text(0, -0.05, f"{score:.2f}", ha="center", fontsize=20)
            ax.set_title(title)
            ax.axis("off")
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(draw_gauge(forgetting_score, "Forgetting Strength"))
        with c2:
            st.pyplot(draw_gauge(trust_score, "Privacy / Trust Level"))

        # Line charts
        st.subheader("Behavioral Verification")
        st.line_chart(pd.DataFrame({
            "Prediction Before": preds_b,
            "Prediction After": preds_a
        }))

        st.subheader("Confidence Change")
        st.line_chart(pd.DataFrame({
            "Confidence Before": probs_b,
            "Confidence After": probs_a
        }))

        # Summary
        st.subheader("Forgetting Summary")
        st.table(pd.DataFrame({
            "Metric": ["Total Samples", "Remaining Samples", "Forgotten Samples"],
            "Value": [
                df.shape[0],
                df_after.shape[0],
                df.shape[0] - df_after.shape[0]
            ]
        }))
