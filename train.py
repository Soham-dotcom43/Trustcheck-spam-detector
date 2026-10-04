"""Train the TrustCheck spam model.

Run:  python train.py

Data used:
  1. data/spam.csv            the UCI SMS Spam Collection (columns v1, v2)
  2. data/indian_scams.csv    modern Indian scam and normal messages (columns label, text)
  3. corrections you saved with the feedback buttons in the web app (trustcheck.db)

Items 2 and 3 get a higher weight because the UCI data is old (UK/US, ~2012).
"""
import argparse
import json
import os
import sqlite3
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

from features import extract_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
CHART_PATH = os.path.join(BASE_DIR, "static", "confusion_matrix.png")

INDIAN_WEIGHT = 3.0     # each Indian example counts as 3 normal ones
FEEDBACK_WEIGHT = 4.0   # corrections from real users count the most


def build_pipeline():
    """TF-IDF words + hand-made scam signals -> Logistic Regression."""
    words = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    signals = Pipeline([
        ("extract", FunctionTransformer(extract_matrix)),
        ("scale", MaxAbsScaler()),
    ])
    features = FeatureUnion([("tfidf", words), ("signals", signals)])
    classifier = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", classifier)])


def load_uci(path):
    if not os.path.exists(path):
        sys.exit(
            f"Cannot find {path}\n"
            "Put spam.csv from the Kaggle 'SMS Spam Collection Dataset' inside the data folder."
        )
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["weight"] = 1.0
    df["source"] = "uci"
    return df


def load_indian(path):
    df = pd.read_csv(path)
    df["weight"] = INDIAN_WEIGHT
    df["source"] = "indian"
    return df[["label", "text", "weight", "source"]]


def load_feedback(db_path):
    """Messages that a user marked as spam / not spam in the web app."""
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["label", "text", "weight", "source"])
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT user_label, text FROM predictions WHERE user_label IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:  # table does not exist yet
        rows = []
    finally:
        con.close()
    df = pd.DataFrame(rows, columns=["label", "text"])
    df["weight"] = FEEDBACK_WEIGHT
    df["source"] = "feedback"
    return df


def load_all(spam_csv, indian_csv, db_path):
    parts = [load_uci(spam_csv)]
    if os.path.exists(indian_csv):
        parts.append(load_indian(indian_csv))
    parts.append(load_feedback(db_path))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    df = df.dropna(subset=["text"])
    # If the same text appears twice, keep the last one (feedback beats older data).
    df = df.drop_duplicates(subset="text", keep="last").reset_index(drop=True)
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def save_confusion_matrix(y_true, y_pred):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Spam"], yticklabels=["Normal", "Spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
    fig.savefig(CHART_PATH, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spam-csv", default=os.path.join(BASE_DIR, "data", "spam.csv"))
    parser.add_argument("--indian-csv", default=os.path.join(BASE_DIR, "data", "indian_scams.csv"))
    parser.add_argument("--db", default=os.path.join(BASE_DIR, "trustcheck.db"))
    args = parser.parse_args()

    df = load_all(args.spam_csv, args.indian_csv, args.db)
    print("Training data by source and label:")
    print(df.groupby(["source", "label"]).size().to_string(), "\n")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["y"]
    )

    # Baseline for comparison: Naive Bayes on words only.
    tfidf_only = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    nb = Pipeline([("tfidf", tfidf_only), ("nb", MultinomialNB())])
    nb.fit(train_df["text"], train_df["y"])
    nb_f1 = f1_score(test_df["y"], nb.predict(test_df["text"]))

    model = build_pipeline()
    model.fit(train_df["text"].tolist(), train_df["y"], clf__sample_weight=train_df["weight"].values)
    pred = model.predict(test_df["text"].tolist())
    lr_f1 = f1_score(test_df["y"], pred)

    print("Naive Bayes (words only)    F1 =", round(nb_f1, 4))
    print("Logistic Regression + signals F1 =", round(lr_f1, 4), "\n")
    print(classification_report(test_df["y"], pred, target_names=["Normal", "Spam"], digits=3))

    indian_test = test_df[test_df["source"] == "indian"]
    indian_f1 = None
    if len(indian_test) >= 5 and indian_test["y"].nunique() == 2:
        indian_f1 = f1_score(indian_test["y"], model.predict(indian_test["text"].tolist()))
        print("F1 on the Indian-message part of the test set:", round(indian_f1, 4))

    save_confusion_matrix(test_df["y"], pred)

    # Final model: learn from ALL the data (the test split was only for measuring).
    final = build_pipeline()
    final.fit(df["text"].tolist(), df["y"], clf__sample_weight=df["weight"].values)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    metrics = {
        "f1_logistic": round(float(lr_f1), 4),
        "f1_naive_bayes": round(float(nb_f1), 4),
        "f1_indian_subset": None if indian_f1 is None else round(float(indian_f1), 4),
        "training_messages": int(len(df)),
        "feedback_messages": int((df["source"] == "feedback").sum()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model to", MODEL_PATH)
    print("Saved metrics to", METRICS_PATH)


if __name__ == "__main__":
    main()
