"""Avaliação no conjunto de teste e comparação original vs. otimizado.

Reúne as métricas exigidas no enunciado (accuracy, recall, F1) mais as
clinicamente relevantes (falsos negativos) para comparar o modelo com os
hiperparâmetros originais (Fase 1) contra o modelo otimizado pelo GA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from diag_opt.data import POSITIVE_LABEL, Dataset
from diag_opt.models import build_pipeline, get_model_spec


@dataclass
class EvaluationResult:
    model_name: str
    params: dict[str, Any]
    metrics: dict[str, float]
    confusion: list[list[int]]  # [[TN, FP], [FN, TP]] na convenção sklearn (labels 0,1)
    false_negatives: int
    false_positives: int

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params": self.params,
            "metrics": self.metrics,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
        }


def _malignant_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Score contínuo para a classe MALIGNA (pos_label=0), usado no AUC.

    Usa ``predict_proba`` quando disponível; caso contrário, ``decision_function``.
    """
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return proba[:, list(pipeline.classes_).index(POSITIVE_LABEL)]
    scores = pipeline.decision_function(X)
    # decision_function positivo favorece a classe 1 (benigno); invertemos p/ maligno.
    return -scores


def evaluate_on_test(model_name: str, params: dict[str, Any], ds: Dataset) -> EvaluationResult:
    """Treina no train e avalia no test set, retornando as métricas completas."""
    pipeline = build_pipeline(model_name, dict(params))
    pipeline.fit(ds.X_train, ds.y_train)

    y_pred = pipeline.predict(ds.X_test)
    y_score_malig = _malignant_scores(pipeline, ds.X_test)

    cm = confusion_matrix(ds.y_test, y_pred, labels=[1, 0])  # [[benigno...],[maligno...]]
    # cm com labels=[1,0]: linha0=benigno(real), linha1=maligno(real)
    # Falso negativo = maligno real previsto como benigno = cm[1][0]
    fn = int(cm[1][0])
    fp = int(cm[0][1])

    metrics = {
        "accuracy": float(accuracy_score(ds.y_test, y_pred)),
        "recall_maligno": float(recall_score(ds.y_test, y_pred, pos_label=POSITIVE_LABEL)),
        "precision_maligno": float(precision_score(ds.y_test, y_pred, pos_label=POSITIVE_LABEL)),
        "f1_maligno": float(f1_score(ds.y_test, y_pred, pos_label=POSITIVE_LABEL)),
        # roc_auc_score espera score da classe positiva do sklearn (label 1);
        # medimos com o score de maligno vs. o alvo "é maligno".
        "roc_auc": float(roc_auc_score((ds.y_test == POSITIVE_LABEL).astype(int), y_score_malig)),
    }

    return EvaluationResult(
        model_name=model_name,
        params=dict(params),
        metrics=metrics,
        confusion=cm.tolist(),
        false_negatives=fn,
        false_positives=fp,
    )


def compare_baseline_vs_optimized(
    model_name: str, optimized_params: dict[str, Any], ds: Dataset
) -> dict[str, EvaluationResult]:
    """Avalia lado a lado o modelo original (Fase 1) e o otimizado pelo GA."""
    spec = get_model_spec(model_name)
    return {
        "baseline": evaluate_on_test(model_name, dict(spec.baseline), ds),
        "optimized": evaluate_on_test(model_name, optimized_params, ds),
    }


def fit_serving_model(model_name: str, params: dict[str, Any], ds: Dataset) -> Pipeline:
    """Treina o modelo final garantindo ``predict_proba`` para a inferência clínica.

    Para o SVM (que treinamos sem ``probability=True`` por performance), envolve o
    estimador em ``CalibratedClassifierCV`` para produzir probabilidades calibradas,
    úteis à interpretação em linguagem natural ("confiança de 92%").
    """
    spec = get_model_spec(model_name)
    pipeline = build_pipeline(model_name, dict(params))
    model = pipeline.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        calibrated = CalibratedClassifierCV(model, cv=3)
        steps = list(pipeline.steps[:-1]) + [("model", calibrated)]
        pipeline = Pipeline(steps)
    pipeline.fit(ds.X_train, ds.y_train)
    return pipeline
