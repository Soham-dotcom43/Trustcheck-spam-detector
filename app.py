"""TrustCheck - Flask web app.

Run:  python app.py      then open http://127.0.0.1:5000
"""
import json
import os
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from predictor import Predictor, level_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_LENGTH = 2000
IST = timezone(timedelta(hours=5, minutes=30))
MODEL_MISSING = "The model file was not found. Run  python train.py  once, then restart the app."

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    spam_prob   REAL NOT NULL,
    level       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    user_label  TEXT CHECK (user_label IN ('spam', 'ham'))
)
"""


def to_ist(iso_text):
    return datetime.fromisoformat(iso_text).astimezone(IST)


def create_app(model_path=None, db_path=None):
    app = Flask(__name__)
    app.config["MODEL_PATH"] = model_path or os.path.join(BASE_DIR, "model", "spam_model.joblib")
    app.config["DB_PATH"] = db_path or os.environ.get("TRUSTCHECK_DB", os.path.join(BASE_DIR, "trustcheck.db"))
    app.config["METRICS_PATH"] = os.path.join(os.path.dirname(app.config["MODEL_PATH"]), "metrics.json")

    with closing(sqlite3.connect(app.config["DB_PATH"])) as con:
        con.execute(SCHEMA)
        con.commit()

    predictor = None
    if os.path.exists(app.config["MODEL_PATH"]):
        predictor = Predictor(app.config["MODEL_PATH"])

    def db():
        con = sqlite3.connect(app.config["DB_PATH"])
        con.row_factory = sqlite3.Row
        return con

    @app.template_filter("ist")
    def ist_filter(iso_text):
        return to_ist(iso_text).strftime("%d %b, %I:%M %p")

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html", result=None, text="", error=None if predictor else MODEL_MISSING)

    @app.post("/check")
    def check():
        text = request.form.get("text", "").strip()
        if predictor is None:
            return render_template("index.html", result=None, text=text, error=MODEL_MISSING), 503
        if not text:
            return render_template("index.html", result=None, text="", error="Please type or paste a message first."), 400
        if len(text) > MAX_LENGTH:
            return render_template(
                "index.html", result=None, text=text[:MAX_LENGTH],
                error=f"That message is too long. Please keep it under {MAX_LENGTH} characters."), 400

        prob = predictor.probability(text)
        with closing(db()) as con:
            cur = con.execute(
                "INSERT INTO predictions (text, spam_prob, level, created_at) VALUES (?, ?, ?, ?)",
                (text, prob, level_for(prob), datetime.now(timezone.utc).isoformat()),
            )
            con.commit()
            pred_id = cur.lastrowid
        return redirect(url_for("result", pred_id=pred_id))

    @app.get("/result/<int:pred_id>")
    def result(pred_id):
        if predictor is None:
            return render_template("index.html", result=None, text="", error=MODEL_MISSING), 503
        with closing(db()) as con:
            row = con.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
        if row is None:
            abort(404)
        return render_template(
            "index.html", result=predictor.analyze(row["text"]), row=row, text=row["text"],
            error=None, saved=request.args.get("saved"),
        )

    @app.post("/feedback/<int:pred_id>")
    def feedback(pred_id):
        label = request.form.get("label")
        if label not in ("spam", "ham"):
            abort(400)
        with closing(db()) as con:
            cur = con.execute("UPDATE predictions SET user_label = ? WHERE id = ?", (label, pred_id))
            con.commit()
        if cur.rowcount == 0:
            abort(404)
        return redirect(url_for("result", pred_id=pred_id, saved=1))

    @app.get("/history")
    def history():
        with closing(db()) as con:
            recent = con.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 20").fetchall()
            counts = {r["level"]: r["n"] for r in con.execute(
                "SELECT level, COUNT(*) AS n FROM predictions GROUP BY level")}
            since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            timeline = con.execute(
                "SELECT created_at, level FROM predictions WHERE created_at >= ?", (since,)).fetchall()
            flagged = con.execute(
                "SELECT text FROM predictions WHERE level IN ('spam', 'suspicious') ORDER BY id DESC LIMIT 200"
            ).fetchall()
            judged = con.execute(
                "SELECT spam_prob, user_label FROM predictions WHERE user_label IS NOT NULL").fetchall()

        total = sum(counts.values())

        # Daily counts for the chart (last 14 days that have data), grouped by Indian date.
        per_day = defaultdict(lambda: {"spam": 0, "suspicious": 0, "safe": 0})
        for r in timeline:
            per_day[to_ist(r["created_at"]).strftime("%Y-%m-%d")][r["level"]] += 1
        days = sorted(per_day)[-14:]
        chart = {
            "labels": days,
            "spam": [per_day[d]["spam"] for d in days],
            "suspicious": [per_day[d]["suspicious"] for d in days],
            "safe": [per_day[d]["safe"] for d in days],
        }

        # Words that most often pushed a message toward spam.
        word_counter = Counter()
        if predictor is not None:
            for r in flagged:
                for reason in predictor.analyze(r["text"], top_n=3)["spam_reasons"]:
                    if reason["kind"] == "word":
                        word_counter[reason["name"]] += 1
        top_words = word_counter.most_common(10)

        agree = sum(1 for r in judged if (r["spam_prob"] >= 0.5) == (r["user_label"] == "spam"))
        metrics = None
        if os.path.exists(app.config["METRICS_PATH"]):
            with open(app.config["METRICS_PATH"]) as f:
                metrics = json.load(f)

        return render_template(
            "history.html", recent=recent, counts=counts, total=total, chart=chart,
            top_words=top_words, judged=len(judged), agree=agree, metrics=metrics,
        )

    # -------------------------------------------------------------------- API
    @app.post("/api/predict")
    def api_predict():
        if predictor is None:
            return jsonify(error=MODEL_MISSING), 503
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text or len(text) > MAX_LENGTH:
            return jsonify(error=f"'text' is required and must be under {MAX_LENGTH} characters"), 400
        a = predictor.analyze(text)
        return jsonify(
            spam_probability=round(a["prob"], 4),
            level=a["level"],
            warning_signs=a["signs"],
            pushed_toward_spam=[r["label"] for r in a["spam_reasons"]],
            pushed_toward_safe=[r["label"] for r in a["safe_reasons"]],
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
