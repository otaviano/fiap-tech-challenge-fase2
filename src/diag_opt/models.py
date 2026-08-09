"""Fábrica de modelos e definição dos espaços de busca de hiperparâmetros.

Cada modelo expõe:

* ``BASELINE_PARAMS`` — os hiperparâmetros usados na Fase 1 (ponto de
  comparação "original" vs "otimizado");
* ``SEARCH_SPACES``   — a codificação dos genes: quais hiperparâmetros o
  algoritmo genético pode variar e em que faixa/domínio.

A construção do estimador é sempre feita via :func:`build_pipeline`, que
encapsula o ``StandardScaler`` para os modelos sensíveis à escala (SVM, KNN,
Regressão Logística), exatamente como na Fase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


@dataclass(frozen=True)
class GeneSpec:
    """Descreve um gene = um hiperparâmetro otimizável.

    kind:
        * ``"float"`` — real em ``[low, high]`` (use ``log=True`` para amostrar
          em escala logarítmica, adequado a parâmetros como C e gamma do SVM);
        * ``"int"``   — inteiro em ``[low, high]``;
        * ``"cat"``   — categórico, escolhido dentre ``choices``.
    """

    name: str
    kind: str  # "float" | "int" | "cat"
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind in ("float", "int"):
            if self.low is None or self.high is None:
                raise ValueError(f"Gene {self.name}: 'low' e 'high' obrigatórios para {self.kind}")
            if self.low > self.high:
                raise ValueError(f"Gene {self.name}: low > high")
            if self.log and self.low <= 0:
                raise ValueError(f"Gene {self.name}: escala log exige low > 0")
        elif self.kind == "cat":
            if not self.choices:
                raise ValueError(f"Gene {self.name}: 'choices' obrigatório para categórico")
        else:
            raise ValueError(f"Gene {self.name}: kind inválido '{self.kind}'")


# --- Construtores de estimador por modelo -------------------------------------

def _svc(**params: Any) -> SVC:
    # Sem probability=True: o scorer roc_auc usa decision_function (mais rápido e
    # sem a calibração interna que foi deprecada no sklearn 1.9). Probabilidades
    # calibradas para a interpretação clínica são geradas na camada de inferência.
    return SVC(random_state=RANDOM_STATE, **params)


def _rf(**params: Any) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)


def _gb(**params: Any) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)


def _logreg(**params: Any) -> LogisticRegression:
    return LogisticRegression(random_state=RANDOM_STATE, max_iter=5000, **params)


def _knn(**params: Any) -> KNeighborsClassifier:
    return KNeighborsClassifier(**params)


@dataclass(frozen=True)
class ModelSpec:
    """Agrega tudo que o GA precisa saber sobre um modelo."""

    name: str
    estimator_factory: Callable[..., Any]
    needs_scaling: bool
    baseline: dict[str, Any]
    genes: tuple[GeneSpec, ...]


MODELS: dict[str, ModelSpec] = {
    "SVM": ModelSpec(
        name="SVM",
        estimator_factory=_svc,
        needs_scaling=True,
        # Baseline Fase 1: SVC(probability=True) — defaults do sklearn (C=1, rbf, scale)
        baseline={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        genes=(
            GeneSpec("C", "float", low=1e-2, high=1e3, log=True),
            GeneSpec("gamma", "float", low=1e-4, high=1e1, log=True),
            GeneSpec("kernel", "cat", choices=("rbf", "poly", "sigmoid")),
        ),
    ),
    "RandomForest": ModelSpec(
        name="RandomForest",
        estimator_factory=_rf,
        needs_scaling=False,
        # Baseline Fase 1: RandomForestClassifier(n_estimators=100)
        baseline={
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        genes=(
            GeneSpec("n_estimators", "int", low=50, high=400),
            GeneSpec("max_depth", "int", low=2, high=30),
            GeneSpec("min_samples_split", "int", low=2, high=20),
            GeneSpec("min_samples_leaf", "int", low=1, high=10),
            GeneSpec("max_features", "cat", choices=("sqrt", "log2", 0.5, 0.8)),
        ),
    ),
    "GradientBoosting": ModelSpec(
        name="GradientBoosting",
        estimator_factory=_gb,
        needs_scaling=False,
        baseline={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0},
        genes=(
            GeneSpec("n_estimators", "int", low=50, high=400),
            GeneSpec("learning_rate", "float", low=1e-3, high=5e-1, log=True),
            GeneSpec("max_depth", "int", low=2, high=8),
            GeneSpec("subsample", "float", low=0.5, high=1.0),
        ),
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"Modelo '{name}' desconhecido. Opções: {list(MODELS)}")
    return MODELS[name]


def build_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    """Constrói o pipeline (scaler + estimador) com os hiperparâmetros dados."""
    spec = get_model_spec(model_name)
    estimator = spec.estimator_factory(**params)
    steps = []
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def baseline_pipeline(model_name: str) -> Pipeline:
    """Pipeline com os hiperparâmetros originais (Fase 1)."""
    spec = get_model_spec(model_name)
    return build_pipeline(model_name, dict(spec.baseline))
