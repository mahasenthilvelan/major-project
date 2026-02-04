import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE CONFIG & THEME
# =========================================================
st.set_page_config(
    page_title="Selective Machine Unlearning",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    h1, h2, h3 { color: #1f2937; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔐 Selective Machine Unlearning Framework")
st.caption("Rule-based • Explainable • Verifiable • Privacy-aware")

st.divider()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def train_model(df, text_col, label_col):
    tfidf = TfidfVectorizer(max_features=4000, stop_words='english')
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


def apply_forgetting_rule(df, rule, user_ids, user_col, label_col=None, keyword=None):
    if rule == "User-based":
        return df[~df[user_col].isin(user_ids)]

    if rule == "Label-based":
        return df[~((df[user_col].isin(user_ids)) & (df[label_col] == 0))]

    if rule == "Keyword-based":
        return df[~((df[user_col].isin(user_ids)) &
                    (df['clean_text'].str.contains(keyword, na=False)))]

    return df


def draw_gauge(score, title, subtitle):
    fig, ax = plt.subplots(figsize=(4, 2.5))

    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="#e5e7eb", linewidth=12)

    filled = np.linspace(0, score * np.pi, 100)
    color = "#ef4444" if score < 0.4 else "#f59e0b" if score < 0.7 else "#22c55e"
    ax.plot(np.cos(filled), np.sin(filled), color=color, linewidth=12)

    ax.text(0, -0.05, f"{score:.2f}", ha="center", fontsize=20, fontweight="bold")
    ax.text(0, -0.30, subtitle, ha="center", fontsize=9)

    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig

# =========================================================
# LAYER 6: WEB APPLICATION
# =========================================================

st.header("📂 Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload any CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")

    with st.expander("Preview Dataset"):
        st.dataframe(df.head())

    st.divider()

    # =====================================================
    # COLUMN SELECTION (DATASET AGNOSTIC)
    # =====================================================
    st.header("🧩 Step 2: Select Columns")

    col1, col2, col3 = st.columns(3)
    with col1:
        user_col = st.selectbox("User ID Column", df.columns)
    with col2:
        text_col = st.selectbox("Text Column", df.columns)
    with col3:
        label_col = st.selectbox("Label Column (0/1)", df.columns)

    df = df[[user_col, text_col, label_col]].dropna()
    df['clean_text'] = df[text_col].apply(clean_text)

    st.divider()

    # =====================================================
    # FORGETTING RULE CONFIGURATION
    # =====================================================
    st.header("🧠 Step 3: Forgetting Rule Configuration")

    rule = st.selectbox(
        "Choose Forgetting Rule",
        ["User-based", "Label-based", "Keyword-based"]
    )

    user_list = df[user_col].unique()
    target_users = st.multiselect("Select User(s) to Forget", user_list)

    keyword = None
    if rule == "Keyword-based":
        keyword = st.text_input("Enter keyword to forget")

    st.divider()

    # =====================================================
    # RUN PIPELINE
    # =====================================================
    if st.button("🚀 Run Machine Unlearning", use_container_width=True):

        if not target_users:
            st.warning("Please select at least one user")
            st.stop()

        # ------------------ BEFORE UNLEARNING ------------------
        model_before, tfidf_before = train_model(df, 'clean_text', label_col)

        user_data = df[df[user_col].isin(target_users)]
        preds_before, probs_before = predict(
            model_before, tfidf_before, user_data['clean_text']
        )

        acc_before = accuracy_score(
            df[label_col],
            model_before.predict(tfidf_before.transform(df['clean_text']))
        )

        # ------------------ APPLY FORGETTING -------------------
        df_after = apply_forgetting_rule(
            df, rule, target_users, user_col, label_col, keyword
        )

        # ------------------ AFTER UNLEARNING -------------------
        model_after, tfidf_after = train_model(df_after, 'clean_text', label_col)

        preds_after, probs_after = predict(
            model_after, tfidf_after, user_data['clean_text']
        )

        acc_after = accuracy_score(
            df_after[label_col],
            model_after.predict(tfidf_after.transform(df_after['clean_text']))
        )

        # =====================================================
        # METRICS & SCORES
        # =====================================================
        pred_change = np.mean(preds_before != preds_after)
        conf_change = np.mean(np.abs(probs_before - probs_after))
        forgetting_score = 0.5 * pred_change + 0.5 * min(conf_change, 1)

        consistency_before = np.var(probs_before)
        consistency_after = np.var(probs_after)
        trust_score = min(max(consistency_after - consistency_before + 0.5, 0), 1)

        st.divider()
        st.header("📊 Results Dashboard")

        # ------------------ ACCURACY TABLE ------------------
        st.subheader("Model Accuracy Comparison")
        st.table(pd.DataFrame({
            "Stage": ["Before Unlearning", "After Unlearning"],
            "Accuracy": [round(acc_before, 4), round(acc_after, 4)]
        }))

        # ------------------ GAUGES ------------------
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(draw_gauge(forgetting_score, "Forgetting Strength", "Higher is Better"))
        with c2:
            st.pyplot(draw_gauge(trust_score, "Privacy / Trust Level", "Resistance to Re-ID"))

        # ------------------ VISUAL VERIFICATION ------------------
        st.subheader("Behavioral Verification")

        st.line_chart(pd.DataFrame({
            "Prediction Before": preds_before,
            "Prediction After": preds_after
        }))

        st.subheader("Confidence Change (URRT)")
        st.line_chart(pd.DataFrame({
            "Confidence Before": probs_before,
            "Confidence After": probs_after
        }))

        # ------------------ DATA SUMMARY ------------------
        st.subheader("Forgetting Summary")
        st.table(pd.DataFrame({
            "Metric": ["Total Samples", "Remaining Samples", "Forgotten Samples"],
            "Count": [df.shape[0], df_after.shape[0], df.shape[0] - df_after.shape[0]]
        }))
