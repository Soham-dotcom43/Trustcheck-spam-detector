"""Loads the trained model, predicts, and explains WHY.

Explanation method (easy to describe in an interview):
  Logistic Regression score = sum over features of (coefficient x feature value).
  So each word's push toward spam or toward normal is simply
  coefficient x its TF-IDF value in this message. We sort those pushes.
"""
import re

import joblib
import numpy as np
from scipy import sparse

from features import FEATURE_LABELS, FEATURE_NAMES, extract_one

# Change these two numbers to make the checker stricter or more relaxed.
SPAM_THRESHOLD = 0.65         # at or above this -> "Likely spam"
SUSPICIOUS_THRESHOLD = 0.35   # between the two numbers -> "Suspicious"

LEVEL_TITLES = {
    "spam": "Likely spam",
    "suspicious": "Suspicious - be careful",
    "safe": "Looks safe",
}

_SEGMENT_RE = re.compile(r"\w+|\W+", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def level_for(prob):
    if prob >= SPAM_THRESHOLD:
        return "spam"
    if prob >= SUSPICIOUS_THRESHOLD:
        return "suspicious"
    return "safe"


class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.features = self.model.named_steps["features"]
        self.clf = self.model.named_steps["clf"]
        tfidf = dict(self.features.transformer_list)["tfidf"]
        self.word_names = list(tfidf.get_feature_names_out())
        self.n_words = len(self.word_names)

    def probability(self, text):
        return float(self.model.predict_proba([text])[0][1])

    # ---- explanation -------------------------------------------------
    def _contributions(self, text):
        """Return a list of dicts: every feature that is active in this message."""
        X = sparse.csr_matrix(self.features.transform([text])).tocoo()
        coefs = self.clf.coef_[0]
        items = []
        for col, value in zip(X.col, X.data):
            push = float(value * coefs[col])
            if col < self.n_words:
                name = self.word_names[col]
                items.append({"name": name, "label": name, "kind": "word", "push": push})
            else:
                key = FEATURE_NAMES[col - self.n_words]
                items.append({"name": key, "label": FEATURE_LABELS[key], "kind": "signal", "push": push})
        return items

    def _segments(self, text, items):
        """Split the message into pieces so the page can colour the important words."""
        unigram = {i["name"]: i["push"] for i in items if i["kind"] == "word" and " " not in i["name"]}
        shown = [abs(v) for v in unigram.values() if abs(v) >= 0.03]
        biggest = max(shown) if shown else 0
        cutoff = max(0.03, 0.2 * biggest)  # skip very weak words so only the important ones are marked
        segments = []
        for m in _SEGMENT_RE.finditer(text):
            piece = m.group()
            seg = {"text": piece, "direction": None, "strength": 0}
            if _WORD_RE.fullmatch(piece):
                value = unigram.get(piece.lower())
                if value is not None and abs(value) >= cutoff and biggest:
                    ratio = abs(value) / biggest
                    seg["direction"] = "spam" if value > 0 else "safe"
                    seg["strength"] = 3 if ratio > 0.6 else 2 if ratio > 0.3 else 1
            segments.append(seg)
        return segments

    @staticmethod
    def warning_signs(text):
        """Simple rule-based red flags, shown as chips."""
        v = dict(zip(FEATURE_NAMES, extract_one(text)))
        signs = []
        if v["has_link"]:
            signs.append(FEATURE_LABELS["has_link"])
        if v["has_phone_number"]:
            signs.append(FEATURE_LABELS["has_phone_number"])
        if v["mentions_money"]:
            signs.append(FEATURE_LABELS["mentions_money"])
        if v["urgent_words"] >= 1:
            signs.append(FEATURE_LABELS["urgent_words"])
        if v["sensitive_words"] >= 1:
            signs.append(FEATURE_LABELS["sensitive_words"])
        if v["caps_ratio"] > 0.5 and len(text) >= 10:
            signs.append(FEATURE_LABELS["caps_ratio"])
        if v["digit_ratio"] > 0.3:
            signs.append(FEATURE_LABELS["digit_ratio"])
        return signs

    def analyze(self, text, top_n=6):
        prob = self.probability(text)
        level = level_for(prob)
        items = self._contributions(text)
        spam_reasons = sorted((i for i in items if i["push"] > 0.02), key=lambda i: -i["push"])[:top_n]
        safe_reasons = sorted((i for i in items if i["push"] < -0.02), key=lambda i: i["push"])[:top_n]
        return {
            "prob": prob,
            "percent": int(round(prob * 100)),
            "level": level,
            "title": LEVEL_TITLES[level],
            "spam_reasons": spam_reasons,
            "safe_reasons": [dict(i, push=-i["push"]) for i in safe_reasons],
            "segments": self._segments(text, items),
            "signs": self.warning_signs(text),
        }
