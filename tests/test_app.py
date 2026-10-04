import sqlite3

SPAM = "Dear customer your SBI account will be blocked today. Update your KYC now at http://sbi-kyc-update.in"
HAM = "Can you send me the notes of DBMS? Exam is on Monday"


def check(client, text):
    return client.post("/check", data={"text": text}, follow_redirects=True)


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Check message" in r.data


def test_empty_message_is_rejected(client):
    r = client.post("/check", data={"text": "   "})
    assert r.status_code == 400
    assert b"Please type or paste a message" in r.data


def test_too_long_message_is_rejected(client):
    r = client.post("/check", data={"text": "a" * 2001})
    assert r.status_code == 400


def test_spam_scores_higher_than_normal(client):
    spam = client.post("/api/predict", json={"text": SPAM}).get_json()
    ham = client.post("/api/predict", json={"text": HAM}).get_json()
    assert spam["spam_probability"] > ham["spam_probability"]
    assert spam["level"] in ("spam", "suspicious")
    assert "Contains a link or website" in spam["warning_signs"]


def test_result_page_shows_explanation(client):
    r = check(client, SPAM)
    assert r.status_code == 200
    assert b"Pushed toward spam" in r.data
    assert b"Warning signs found" in r.data


def test_html_in_message_is_escaped(client):
    r = check(client, "<script>alert(1)</script> win a free prize")
    assert b"<script>alert(1)</script>" not in r.data


def test_feedback_is_saved(client, tmp_path):
    check(client, HAM)
    r = client.post("/feedback/1", data={"label": "spam"}, follow_redirects=True)
    assert r.status_code == 200
    assert b"Your answer was saved" in r.data
    con = sqlite3.connect(str(tmp_path / "test.db"))
    assert con.execute("SELECT user_label FROM predictions WHERE id = 1").fetchone()[0] == "spam"


def test_bad_feedback_value_is_rejected(client):
    check(client, HAM)
    assert client.post("/feedback/1", data={"label": "banana"}).status_code == 400
    assert client.post("/feedback/999", data={"label": "spam"}).status_code == 404


def test_history_page_works_empty_and_filled(client):
    assert client.get("/history").status_code == 200
    check(client, SPAM)
    check(client, HAM)
    r = client.get("/history")
    assert r.status_code == 200
    assert b"Messages checked" in r.data


def test_sql_injection_text_is_stored_safely(client):
    check(client, "'; DROP TABLE predictions; -- free prize")
    assert client.get("/history").status_code == 200


def test_unknown_result_is_404(client):
    assert client.get("/result/12345").status_code == 404


def test_api_requires_text(client):
    assert client.post("/api/predict", json={}).status_code == 400
