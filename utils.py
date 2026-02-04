# utils.py

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# --------------------------------------------------
# Train Model
# --------------------------------------------------
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


# --------------------------------------------------
# Predict
# --------------------------------------------------
def predict(model, tfidf, texts):
    X = tfidf.transform(texts)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs


# --------------------------------------------------
# Rule-Based Forgetting (NO hardcoded columns)
# --------------------------------------------------
def apply_forgetting_rule(
    df,
    rule,
    user_col,
    target_users,
    text_col,
    label_col=None,
    keyword=None
):
    # User-based: remove ALL data of selected users
    if rule == "User-based":
        return df[~df[user_col].isin(target_users)]

    # Label-based (Selective): remove ONLY matching label rows of selected users
    if rule == "Label-based (Selective)":
        return df[~(
            (df[user_col].isin(target_users)) &
            (df[label_col] == 0)
        )]

    # Keyword-based: remove ONLY rows containing keyword for selected users
    if rule == "Keyword-based":
        return df[~(
            (df[user_col].isin(target_users)) &
            (df[text_col].str.contains(keyword, na=False))
        )]

    return df
