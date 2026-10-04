"""Hand-made features that catch scam patterns the word model cannot see.

Used by both train.py (training) and app.py (explaining a result).
Keep this file free of Flask code so the model can be loaded anywhere.
"""
import math
import re

import numpy as np

URL_RE = re.compile(r"(https?://|www\.|bit\.ly|tinyurl|t\.co/|\b[a-z0-9-]+\.(?:in|com|net|org|xyz|top|info|co|link)\b)", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)|(?<!\d)\d{10,12}(?!\d)")
MONEY_RE = re.compile(r"(rs\.?\s?\d+|inr\s?\d+|₹\s?\d+|\$\s?\d+|£\s?\d+|\d+\s?(?:rs|rupees|lakh|crore)\b)", re.I)

URGENT_WORDS = [
    "urgent", "immediately", "now", "today", "tonight", "expire", "expires", "expired",
    "blocked", "block", "suspended", "suspend", "last chance", "final notice", "hurry",
    "act now", "limited time", "within 24", "avoid", "disconnection", "deactivated",
]
SENSITIVE_WORDS = [
    "kyc", "otp", "pan", "aadhaar", "aadhar", "password", "pin", "cvv", "bank account",
    "verify", "update your", "refund", "customs", "delivery fee", "claim", "prize", "lottery",
    "winner", "won", "selected", "work from home", "earn", "free", "cashback", "loan",
]

FEATURE_NAMES = [
    "has_link",
    "has_phone_number",
    "mentions_money",
    "urgent_words",
    "sensitive_words",
    "digit_ratio",
    "caps_ratio",
    "exclamations",
    "length_log",
]

# Plain-English labels used when explaining a result in the web page.
FEATURE_LABELS = {
    "has_link": "Contains a link or website",
    "has_phone_number": "Contains a phone number",
    "mentions_money": "Mentions an amount of money",
    "urgent_words": "Uses urgent or threatening words",
    "sensitive_words": "Asks about KYC, OTP, prizes, jobs or similar",
    "digit_ratio": "Has many digits",
    "caps_ratio": "Has many CAPITAL letters",
    "exclamations": "Uses exclamation marks",
    "length_log": "Message length",
}


def _count_words(text_lower, words):
    return sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", text_lower))


def extract_one(text):
    """Return a list of numbers, one per name in FEATURE_NAMES."""
    text = text or ""
    lower = text.lower()
    letters = [c for c in text if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper())
    digits = sum(1 for c in text if c.isdigit())
    n = max(len(text), 1)
    return [
        1.0 if URL_RE.search(text) else 0.0,
        1.0 if PHONE_RE.search(text) else 0.0,
        1.0 if MONEY_RE.search(text) else 0.0,
        float(min(_count_words(lower, URGENT_WORDS), 4)),
        float(min(_count_words(lower, SENSITIVE_WORDS), 4)),
        digits / n,
        caps / max(len(letters), 1),
        float(min(text.count("!"), 4)),
        math.log1p(len(text)),
    ]


def extract_matrix(texts):
    """Used inside the sklearn pipeline (FunctionTransformer)."""
    return np.array([extract_one(t) for t in texts], dtype=float)
