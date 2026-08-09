"""Função de fitness baseada nas métricas de desempenho dos modelos.

A avaliação usa **validação cruzada estratificada** (mesma estratégia da Fase 1)
para estimar o desempenho de forma robusta, evitando que o GA sobreajuste a uma
única partição.

O fitness é uma combinação ponderada de métricas, com peso dominante no
**recall da classe maligna** (``pos_label=0``) — coerente com o objetivo
clínico de minimizar falsos negativos. Os pesos são configuráveis para permitir
os experimentos exigidos no enunciado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from diag_opt.data import POSITIVE_LABEL
from diag_opt.models import build_pipeline
from diag_opt.ga.encoding import Chromosome


@dataclass(frozen=True)
class FitnessConfig:
    """Pesos e parâmetros da avaliação por validação cruzada."""

    weights: dict[str, float] = field(
        default_factory=lambda: {"recall": 0.6, "f1": 0.3, "roc_auc": 0.1}
    )
    cv_folds: int = 5
    random_state: int = 42

    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("A soma dos pesos do fitness deve ser positiva")
        return {k: v / total for k, v in self.weights.items()}


@dataclass
class FitnessResult:
    """Resultado detalhado de uma avaliação (fitness + métricas componentes)."""

    fitness: float
    metrics: dict[str, float]


def _scorers() -> dict:
    # Recall e F1 são medidos sobre a classe MALIGNA (pos_label=0).
    return {
        "recall": make_scorer(recall_score, pos_label=POSITIVE_LABEL),
        "f1": make_scorer(f1_score, pos_label=POSITIVE_LABEL),
        "roc_auc": "roc_auc",
    }


class FitnessEvaluator:
    """Avalia cromossomos via CV, com cache para evitar reavaliações."""

    def __init__(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        config: FitnessConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.X = X
        self.y = y
        self.config = config or FitnessConfig()
        self._cv = StratifiedKFold(
            n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state
        )
        self._weights = self.config.normalized_weights()
        self._scorers = _scorers()
        self._cache: dict[tuple, FitnessResult] = {}
        self.n_evaluations = 0

    @staticmethod
    def _key(chromosome: Chromosome) -> tuple:
        return tuple(sorted(chromosome.items()))

    def evaluate(self, chromosome: Chromosome) -> FitnessResult:
        key = self._key(chromosome)
        if key in self._cache:
            return self._cache[key]

        pipeline = build_pipeline(self.model_name, dict(chromosome))
        try:
            scores = cross_validate(
                pipeline,
                self.X,
                self.y,
                cv=self._cv,
                scoring=self._scorers,
                n_jobs=-1,
                error_score="raise",
            )
            metrics = {name: float(np.mean(scores[f"test_{name}"])) for name in self._scorers}
            fitness = sum(self._weights[name] * metrics[name] for name in self._weights)
        except Exception:
            # Combinação inválida de hiperparâmetros -> fitness mínimo (penalização).
            metrics = {name: 0.0 for name in self._scorers}
            fitness = 0.0

        result = FitnessResult(fitness=fitness, metrics=metrics)
        self._cache[key] = result
        self.n_evaluations += 1
        return result
