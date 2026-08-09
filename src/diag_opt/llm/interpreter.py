"""Interpretação em linguagem natural dos diagnósticos.

Transforma a saída numérica do modelo em um texto acionável para o médico:

1. constrói o *contexto do paciente* (predição, confiança e características mais
   relevantes, medidas em desvios-padrão em relação à distribuição de treino);
2. gera a explicação via LLM local;
3. em caso de indisponibilidade do LLM, usa um *fallback* determinístico baseado
   em template — garantindo que a demonstração e os testes rodem sempre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from diag_opt.data import POSITIVE_LABEL, Dataset
from diag_opt.llm.client import LLMClient, LLMUnavailableError
from diag_opt.llm.prompts import build_messages

@dataclass
class Interpretation:
    context: dict[str, Any]
    text: str
    source: str  # "llm" | "fallback"


def build_patient_context(
    pipeline: Pipeline, ds: Dataset, index: int, top_k: int = 4
) -> dict[str, Any]:
    """Extrai predição, confiança e principais características de um paciente."""
    patient = ds.X_test.iloc[[index]]
    proba = pipeline.predict_proba(patient)[0]
    classes = list(pipeline.classes_)
    p_malig = float(proba[classes.index(POSITIVE_LABEL)])
    predicted = int(pipeline.predict(patient)[0])

    # z-score de cada feature em relação à distribuição de TREINO.
    mean = ds.X_train.mean()
    std = ds.X_train.std().replace(0, 1e-9)
    z = ((patient.iloc[0] - mean) / std)

    ranked = z.abs().sort_values(ascending=False).head(top_k).index
    top_features: list[dict[str, Any]] = []
    for name in ranked:
        zval = float(z[name])
        # No Wisconsin Breast Cancer, praticamente todas as features (tamanho,
        # concavidade, irregularidade do núcleo) crescem com a malignidade — logo
        # valor acima da média puxa para maligno, abaixo para benigno.
        direction = "maligno" if zval > 0 else "benigno"
        top_features.append(
            {
                "name": name,
                "value": float(patient.iloc[0][name]),
                "zscore": zval,
                "direction": direction,
            }
        )

    return {
        "index": int(index),
        "predicted_label": predicted,
        "probability_malignant": p_malig,
        "top_features": top_features,
    }


def _fallback_text(context: dict[str, Any]) -> str:
    """Explicação determinística (sem LLM), no mesmo formato de seções."""
    predicao = "maligno" if context["predicted_label"] == POSITIVE_LABEL else "benigno"
    conf = context["probability_malignant"] * 100
    linhas_feat = [
        f"- **{f['name']}** (valor {f['value']:.2f}): {abs(f['zscore']):.1f} desvios-padrão "
        f"{'acima' if f['zscore'] > 0 else 'abaixo'} da média, padrão associado a tecido "
        f"{f['direction']}."
        for f in context["top_features"]
    ]
    return (
        "## Resumo\n"
        f"O modelo sugere um resultado **{predicao}**, com confiança de malignidade de "
        f"{conf:.1f}%. Este é um apoio estatístico, não um diagnóstico.\n\n"
        "## Fatores que mais influenciaram\n" + "\n".join(linhas_feat) + "\n\n"
        "## Recomendação de conduta\n"
        "Recomenda-se correlacionar o resultado com o exame clínico e de imagem. "
        "Em caso de suspeita de malignidade, considerar confirmação histopatológica.\n\n"
        "## Aviso\n"
        "Esta interpretação é uma ferramenta de apoio à decisão. A palavra final é "
        "sempre do médico responsável."
    )


def interpret_patient(
    pipeline: Pipeline,
    ds: Dataset,
    index: int,
    client: LLMClient | None = None,
    top_k: int = 4,
) -> Interpretation:
    """Gera a interpretação de um caso, com fallback se o LLM estiver indisponível."""
    context = build_patient_context(pipeline, ds, index, top_k=top_k)
    client = client or LLMClient()

    try:
        text = client.chat(build_messages(context))
        source = "llm"
    except LLMUnavailableError:
        text = _fallback_text(context)
        source = "fallback"

    return Interpretation(context=context, text=text, source=source)
