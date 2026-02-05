# ================================
# Selective Machine Unlearning App
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="Machine Unlearning", layout="wide")
st.title("🔐 Selective Machine Unlearning Framework")
st.caption("Unlearning • Verification • Forgetting • Explainability • Trust")

# --------------------------------
# Helper Functions
# --------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def train_model(df, text_col, label_col):
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    X = tfidf.fit_transform(df[text_col])
    y = df[label_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, tfidf

def predict(model, tfidf, texts):
    X = tfidf.transform(texts)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs

def apply_forgetting(df, rule, user_col, users, text_col, label_col, keyword):
    if rule == "User-based":
        return df[~df[user_col].isin(users)]

    if rule == "Label-based (Selective)":
        return df[~((df[user_col].isin(users)) & (df[label_col] == 0))]

    if rule == "Keyword-based":
        return df[~((df[user_col].isin(users)) &
                    (df[text_col].str.contains(keyword, na=False)))]

    return df

def get_top_words(model, tfidf, n=10):
    words = np.array(tfidf.get_feature_names_out())
    weights = model.coef_[0]
    top = words[np.argsort(weights)[-n:]]
    return list(top)

def trust_score(before, after):
    diff = np.abs(before[:len(after)] - after[:len(before)])
    return round(1 - np.mean(diff), 3)

# --------------------------------
# Step 1: Upload Dataset
# --------------------------------
st.header("📂 Step 1: Upload Dataset")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset loaded")

    st.dataframe(df.head())

    # --------------------------------
    # Step 2: Column Selection
    # --------------------------------
    st.header("🧩 Step 2: Column Selection")

    user_col = st.selectbox("User ID column", df.columns)
    text_col = st.selectbox("Text column", df.columns)
    label_col = st.selectbox("Label column (0/1)", df.columns)

    df = df[[user_col, text_col, label_col]].dropna()
    df["clean_text"] = df[text_col].apply(clean_text)

    # --------------------------------
    # Step 3: Forgetting Rule
    # --------------------------------
    st.header("🧠 Step 3: Forgetting Rule")

    rule = st.selectbox(
        "Choose forgetting rule",
        ["User-based", "Label-based (Selective)", "Keyword-based"]
    )

    users = st.multiselect("Select user(s) to forget", df[user_col].unique())

    keyword = ""
    if rule == "Keyword-based":
        keyword = st.text_input("Keyword to forget", value="bad")

    # --------------------------------
    # Run Unlearning
    # --------------------------------
    if st.button("🚀 Run Unlearning", use_container_width=True):

        if not users:
            st.warning("Please select at least one user")
            st.stop()

        # -------- BEFORE UNLEARNING (Layer 1 + 2)
        model_before, tfidf_before = train_model(df, "clean_text", label_col)

        user_data = df[df[user_col].isin(users)]

        preds_b, probs_b = predict(
            model_before, tfidf_before, user_data["clean_text"]
        )

        acc_before = accuracy_score(
            df[label_col],
            model_before.predict(tfidf_before.transform(df["clean_text"]))
        )

        # -------- APPLY UNLEARNING (Layer 1)
        df_after = apply_forgetting(
            df, rule, user_col, users, "clean_text", label_col, keyword
        )

        if df_after.empty:
            st.error("All data removed. Try another rule.")
            st.stop()

        # -------- AFTER UNLEARNING (Layer 2)
        model_after, tfidf_after = train_model(df_after, "clean_text", label_col)

        preds_a, probs_a = predict(
            model_after, tfidf_after, user_data["clean_text"]
        )

        acc_after = accuracy_score(
            df_after[label_col],
            model_after.predict(tfidf_after.transform(df_after["clean_text"]))
        )

        # -------- Layer 3: Forgetting Score
        pred_change = np.mean(preds_b != preds_a)
        conf_change = np.mean(np.abs(probs_b - probs_a))
        forgetting_score = round(0.5 * pred_change + 0.5 * conf_change, 3)

        # -------- Layer 4: Explainability
        top_before = get_top_words(model_before, tfidf_before)
        top_after = get_top_words(model_after, tfidf_after)

        # -------- Layer 5: Trust / Re-ID
        trust = trust_score(probs_b, probs_a)

        # --------------------------------
        # Results
        # --------------------------------
        st.divider()
        st.header("📊 Results")

        st.subheader("Accuracy")
        st.table(pd.DataFrame({
            "Stage": ["Before", "After"],
            "Accuracy": [round(acc_before, 4), round(acc_after, 4)]
        }))

        c1, c2 = st.columns(2)
        c1.metric("Forgetting Score", forgetting_score)
        c2.metric("Trust Score", trust)

        st.subheader("Explainability (Top Words)")
        col1, col2 = st.columns(2)
        col1.write("Before Unlearning")
        col1.write(top_before)
        col2.write("After Unlearning")
        col2.write(top_after)

        st.subheader("Verification (Predictions)")
        st.line_chart(pd.DataFrame({
            "Before": preds_b,
            "After": preds_a
        }))

        st.subheader("Confidence Change")
        st.line_chart(pd.DataFrame({
            "Before": probs_b,
            "After": probs_a
        }))

        st.subheader("Unlearning Summary")
        st.table(pd.DataFrame({
            "Metric": ["Total Samples", "Remaining Samples", "Forgotten Samples"],
            "Value": [
                df.shape[0],
                df_after.shape[0],
                df.shape[0] - df_after.shape[0]
            ]
        }))
