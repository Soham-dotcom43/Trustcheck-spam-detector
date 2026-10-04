# TrustCheck - Spam and Scam Message Detector

A Flask web app that scores SMS/WhatsApp/email messages as **Likely spam**, **Suspicious** or **Looks safe**,
and explains *why* by highlighting the words and warning signs that pushed the score.

## Features
- TF-IDF + hand-made scam signals (links, phone numbers, money, urgent words, KYC/OTP words) -> Logistic Regression
- Word-level explanation: coefficient x TF-IDF value for each word
- Three levels (Likely spam / Suspicious / Looks safe) with adjustable thresholds in `predictor.py`
- Training data: UCI SMS Spam Collection + Indian scam examples (KYC, parcel fee, fake jobs, UPI)
- Feedback buttons save corrections to SQLite; `python train.py` retrains with them
- History page: totals, daily chart, most common flagged words, model agreement with users
- JSON API: `POST /api/predict` with `{"text": "..."}`
- Tests with pytest

## Run it
```
pip install -r requirements.txt
python train.py        # needs data/spam.csv from Kaggle (SMS Spam Collection Dataset)
python app.py          # open http://127.0.0.1:5000
pytest                 # run the tests
```

## How the explanation works
Logistic Regression adds up `coefficient x feature value` for every feature. A positive
number pushes toward spam, a negative one toward normal. The page sorts these numbers and
colours the words in your message.

## Limitations
- It learns writing patterns, not truth. It cannot fact-check a message.
- The UCI data is old (UK/US), so new scam styles can be missed. Feedback and the Indian examples reduce this.
- English (and Hinglish) only.
