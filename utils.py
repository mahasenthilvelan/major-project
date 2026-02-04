# utils.py

import re
import numpy as np
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
# Train Model
# -------------------------------------------------
def train_model(df, text_col, label_col):
    tfidf = TfidfVectorizer(
        max_features=4000,
        stop_words='english'
    )
    X = tfidf.fit_transform(df[text_col])
    y = df[label_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, tfidf


# -------------------------------------------------
# Predict
# -------------------------------------------------
def predict(model, tfidf, texts):
    X = tfidf.transform(texts)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs


# -------------------------------------------------
# Rule-Based Forgetting Engine
# -------------------------------------------------
def apply_forgetting_rule(
    df,
    rule,
    user_ids,
    user_col,
    label_col=None,
    keyword=None
):
    if rule == "User-based":
        return df[~df[user_col].isin(user_ids)]

    if rule == "Label-based":
        return df[~((df[user_col].isin(user_ids)) &
                    (df[label_col] == 0))]

    if rule == "Keyword-based":
        return df[~((df[user_col].isin(user_ids)) &
                    (df['clean_text'].str.contains(keyword, na=False)))]

    return df
