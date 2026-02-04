import streamlit as st
import pandas as pd
import numpy as np
from utils import clean_text, train_model, predict, apply_forgetting_rule

st.set_page_config(page_title="Selective Machine Unlearning", layout="wide")
st.title("Selective Machine Unlearning Framework")

# -----------------------------------
# STEP 1: Upload Dataset
# -----------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # STEP 2: Column Selection (IMPORTANT)
    # -----------------------------------
    st.subheader("Select Required Columns")

    user_col = st.selectbox("Select User ID Column", df.columns)
    text_col = st.selectbox("Select Text Column", df.columns)
    label_col = st.selectbox("Select Label Column (0/1)", df.columns)

    # Preprocess
    df = df[[user_col, text_col, label_col]].dropna()
    df['clean_text'] = df[text_col].apply(clean_text)

    # -----------------------------------
    # STEP 3: Forgetting Rule Selection
    # -----------------------------------
    st.subheader("Select Forgetting Rule")

    rule = st.selectbox(
        "Choose Forgetting Rule",
        ["User-based", "Label-based", "Keyword-based"]
    )

    user_list = df[user_col].unique()
    target_user = st.selectbox("Select User to Forget", user_list)

    keyword = None
    if rule == "Keyword-based":
        keyword = st.text_input("Enter keyword to forget")

    # -----------------------------------
    # STEP 4: Run Unlearning
    # -----------------------------------
    if st.button("Run Unlearning"):
        # Train before unlearning
        model_before, tfidf_before = train_model(df, 'clean_text', label_col)
        user_data = df[df[user_col] == target_user]
        preds_before, probs_before = predict(
            model_before, tfidf_before, user_data['clean_text']
        )

        # Apply forgetting rule
        df_after = apply_forgetting_rule(
            df, rule, target_user, user_col, label_col, keyword
        )

        # Train after unlearning
        model_after, tfidf_after = train_model(df_after, 'clean_text', label_col)
        preds_after, probs_after = predict(
            model_after, tfidf_after, user_data['clean_text']
        )

        # -----------------------------------
        # STEP 5: Results
        # -----------------------------------
        st.subheader("Results")

        pred_change = np.mean(preds_before != preds_after)
        conf_change = np.mean(np.abs(probs_before - probs_after))
        forgetting_score = 0.5 * pred_change + 0.5 * min(conf_change, 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction Change Rate", round(pred_change, 3))
        col2.metric("Confidence Change", round(conf_change, 3))
        col3.metric("Forgetting Score", round(forgetting_score, 3))

        # -----------------------------------
        # STEP 6: URRT
        # -----------------------------------
        consistency_before = np.var(probs_before)
        consistency_after = np.var(probs_after)
        reid_score = consistency_after - consistency_before

        st.subheader("User Re-identification Resistance Test")
        if reid_score > 0:
            st.success("Strong resistance to user re-identification")
        else:
            st.warning("Possible identity leakage detected")
