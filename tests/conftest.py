import os
import sys

import joblib
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import create_app  # noqa: E402
from train import build_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def model_file(tmp_path_factory):
    """Train a small model from the Indian examples only, so tests do not need spam.csv."""
    df = pd.read_csv(os.path.join(ROOT, "data", "indian_scams.csv"))
    model = build_pipeline()
    model.fit(df["text"].tolist(), (df["label"] == "spam").astype(int))
    path = tmp_path_factory.mktemp("model") / "spam_model.joblib"
    joblib.dump(model, path)
    return str(path)


@pytest.fixture()
def client(model_file, tmp_path):
    app = create_app(model_path=model_file, db_path=str(tmp_path / "test.db"))
    app.config["TESTING"] = True
    return app.test_client()
