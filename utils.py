# utils.py

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -------------------------------------------------
# Text Cleaning
# -------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# -------------------------------------------------
# Train Logistic Regression Model
# -------------------------------------------------
def train_model(df, text_col, label_col):
    tfidf = TfidfVectorizer(
        max_features=4000,
        stop_words="english"
    )
    X = tfidf.fit_transform(df[text_col])
    y = df[label_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, tfidf


# -------------------------------------------------
# Predict Labels & Confidence
# -------------------------------------------------
def predict(model, tfidf, texts):
    X = tfidf.transform(texts)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs


# -------------------------------------------------
# Layer 1: Unlearning Engine (Rule-based)
# -------------------------------------------------
def apply_forgetting_rule(
    df,
    rule,
    user_col,
    target_users,
    text_col,
    label_col=None,
    keyword=None
):
    if rule == "User-based":
        return df[~df[user_col].isin(target_users)]

    if rule == "Label-based (Selective)":
        return df[~(
            (df[user_col].isin(target_users)) &
            (df[label_col] == 0)
        )]

    if rule == "Keyword-based":
        return df[~(
            (df[user_col].isin(target_users)) &
            (df[text_col].str.contains(keyword, na=False))
        )]

    return df


# -------------------------------------------------
# Layer 4: Explainability
# Feature importance extraction
# -------------------------------------------------
def get_top_features(model, tfidf, top_n=10):
    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = model.coef_[0]

    top_positive = feature_names[np.argsort(coefficients)[-top_n:]]
    top_negative = feature_names[np.argsort(coefficients)[:top_n]]

    return top_positive, top_negative


# -------------------------------------------------
# Layer 5: Trust / Re-identification Resistance
# Confidence overlap score
# -------------------------------------------------
def confidence_overlap(before_probs, after_probs):
    min_len = min(len(before_probs), len(after_probs))
    diff = np.abs(before_probs[:min_len] - after_probs[:min_len])
    overlap_score = 1 - np.mean(diff)
    return max(0, min(overlap_score, 1))
