"""API: train a throwaway bundle in tmp, then hit /health and /predict."""

from fastapi.testclient import TestClient

from titanic_prod.api import create_app
from titanic_prod.config import REPO_ROOT
from titanic_prod.train import run

PAX = {
    "pclass": 1, "sex": "female", "age": 29,
    "sibsp": 0, "parch": 0, "fare": 80.0, "embarked": "C",
}


def _client(tmp_path):
    artifacts = tmp_path / "artifacts"
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    result = run(REPO_ROOT / "configs" / "logreg.yaml",
                 artifacts_dir=artifacts, mlflow_uri=uri)
    assert result["accuracy"] > 0.70  # sanity: better than chance, near class ~0.79
    return TestClient(create_app(artifacts))


def test_health(tmp_path):
    resp = _client(tmp_path).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_survival(tmp_path):
    resp = _client(tmp_path).post("/predict", json=PAX)
    assert resp.status_code == 200
    body = resp.json()
    assert body["survived"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_bad_input(tmp_path):
    bad = dict(PAX, sex="unknown")
    resp = _client(tmp_path).post("/predict", json=bad)
    assert resp.status_code == 422
