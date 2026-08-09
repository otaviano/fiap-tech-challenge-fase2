"""Serviço de inferência + interpretação (FastAPI).

Serviço *stateless* que expõe o modelo de diagnóstico e a interpretação clínica
por LLM. É o container escalado horizontalmente pela IaC em [`infra/`](../../../infra).

Endpoints:
    GET  /health      -> checagem de saúde (usada pelo load balancer)
    POST /predict     -> predição a partir das 30 features
    POST /interpret   -> predição + interpretação clínica em linguagem natural

Execução local:
    uvicorn diag_opt.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import pandas as pd

from diag_opt.data import POSITIVE_LABEL, load_dataset
from diag_opt.evaluation import fit_serving_model
from diag_opt.llm.client import LLMClient
from diag_opt.llm.interpreter import _fallback_text
from diag_opt.llm.prompts import build_messages
from diag_opt.llm.quality import evaluate_interpretation

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # dependência opcional
    raise ImportError(
        "Instale as dependências de serviço: pip install fastapi uvicorn"
    ) from exc

# Hiperparâmetros otimizados pelo GA (melhor SVM encontrado nos experimentos).
_DEFAULT_SVM_PARAMS = {
    "C": float(os.getenv("MODEL_C", "5.31")),
    "gamma": float(os.getenv("MODEL_GAMMA", "0.0336")),
    "kernel": os.getenv("MODEL_KERNEL", "rbf"),
}

app = FastAPI(title="Diag-Opt — Diagnóstico Oncológico", version="0.1.0")


@lru_cache(maxsize=1)
def _bootstrap() -> tuple[Any, Any, list[str]]:
    """Carrega dados e treina o modelo de serviço uma única vez (cache)."""
    ds = load_dataset()
    pipeline = fit_serving_model("SVM", _DEFAULT_SVM_PARAMS, ds)
    return ds, pipeline, ds.feature_names


class Features(BaseModel):
    values: dict[str, float]  # {nome_da_feature: valor} para as 30 features


class InterpretRequest(BaseModel):
    values: dict[str, float]
    top_k: int = 4


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _to_frame(values: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    missing = [f for f in feature_names if f not in values]
    if missing:
        raise HTTPException(status_code=422, detail=f"Features ausentes: {missing[:5]}...")
    return pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)


@app.post("/predict")
def predict(features: Features) -> dict[str, Any]:
    _ds, pipeline, names = _bootstrap()
    frame = _to_frame(features.values, names)
    proba = pipeline.predict_proba(frame)[0]
    classes = list(pipeline.classes_)
    p_malig = float(proba[classes.index(POSITIVE_LABEL)])
    label = int(pipeline.predict(frame)[0])
    return {
        "prediction": "maligno" if label == POSITIVE_LABEL else "benigno",
        "probability_malignant": p_malig,
    }


@app.post("/interpret")
def interpret(req: InterpretRequest) -> dict[str, Any]:
    ds, pipeline, names = _bootstrap()
    frame = _to_frame(req.values, names)

    # Constrói o contexto do paciente para uma amostra arbitrária (z-score das
    # features em relação à distribuição de treino).
    proba = pipeline.predict_proba(frame)[0]
    classes = list(pipeline.classes_)
    p_malig = float(proba[classes.index(POSITIVE_LABEL)])
    label = int(pipeline.predict(frame)[0])
    mean, std = ds.X_train.mean(), ds.X_train.std().replace(0, 1e-9)
    z = ((frame.iloc[0] - mean) / std)
    ranked = z.abs().sort_values(ascending=False).head(req.top_k).index
    top_features = [
        {
            "name": n,
            "value": float(frame.iloc[0][n]),
            "zscore": float(z[n]),
            "direction": "maligno" if z[n] > 0 else "benigno",
        }
        for n in ranked
    ]
    ctx = {
        "predicted_label": label,
        "probability_malignant": p_malig,
        "top_features": top_features,
    }

    client = LLMClient()
    try:
        text = client.chat(build_messages(ctx))
        source = "llm"
    except Exception:
        text = _fallback_text(ctx)
        source = "fallback"

    quality = evaluate_interpretation(text, ctx)
    return {
        "prediction": "maligno" if label == POSITIVE_LABEL else "benigno",
        "probability_malignant": p_malig,
        "interpretation": text,
        "source": source,
        "quality_score": quality.score,
    }
