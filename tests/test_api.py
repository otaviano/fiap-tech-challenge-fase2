"""Testes do serviço de inferência (FastAPI). Puláveis se fastapi não instalado."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from diag_opt.serving.api import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_identifica_o_modelo_servido(client):
    """O /health precisa dizer QUAL configuração do GA está no ar."""
    body = client.get("/health").json()
    assert body["model"]["estimator"] == "SVM"
    params = body["model"]["params"]
    assert set(params) == {"C", "gamma", "kernel"}
    assert params["kernel"] == "rbf"
    assert body["llm"]["model"]  # identifica o LLM configurado


def test_predict(client, dataset):
    row = dataset.X_test.iloc[0].to_dict()
    resp = client.post("/predict", json={"values": row})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in ("maligno", "benigno")
    assert 0.0 <= body["probability_malignant"] <= 1.0


def test_predict_features_ausentes(client):
    resp = client.post("/predict", json={"values": {"mean radius": 1.0}})
    assert resp.status_code == 422


def test_interpret_tem_qualidade(client, dataset):
    row = dataset.X_test.iloc[0].to_dict()
    resp = client.post("/interpret", json={"values": row, "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "interpretation" in body
    assert body["source"] in ("llm", "fallback")
    assert 0.0 <= body["quality_score"] <= 1.0
