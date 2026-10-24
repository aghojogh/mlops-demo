import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.path.exists("model.pt"), reason="model.pt not found"
)


def get_client():
    from src.serve import app
    return TestClient(app)


def test_health():
    assert get_client().get("/health").json() == {"status": "ok"}


def test_predict():
    r = get_client().post("/predict", json={"features": [0.0] * 64})
    assert r.status_code == 200
    assert 0 <= r.json()["prediction"] <= 9


def test_wrong_size():
    r = get_client().post("/predict", json={"features": [0.0] * 3})
    assert r.status_code == 422
