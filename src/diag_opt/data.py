"""Carregamento e particionamento do dataset de diagnóstico.

Mantém a mesma base da Fase 1 (Wisconsin Breast Cancer Diagnostic), carregada
diretamente do scikit-learn — não exige download manual.

Convenção clínica (herdada da Fase 1):
    target == 0  -> MALIGNO   (classe positiva de interesse)
    target == 1  -> BENIGNO

O erro mais grave em oncologia é o falso negativo (maligno classificado como
benigno). Por isso a métrica principal de otimização é o *recall da classe
maligna* (``pos_label=0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Rótulo da classe clinicamente positiva (maligno) no dataset do sklearn.
POSITIVE_LABEL = 0
RANDOM_STATE = 42


@dataclass
class Dataset:
    """Container imutável com os dados já particionados."""

    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    target_names: list[str]  # index 0 = maligno, 1 = benigno


def load_dataset(test_size: float = 0.2, random_state: int = RANDOM_STATE) -> Dataset:
    """Carrega o Wisconsin Breast Cancer e faz split estratificado.

    O split estratificado preserva a proporção das classes em treino e teste,
    importante em datasets clinicamente desbalanceados.
    """
    raw = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = raw.data
    y: pd.Series = raw.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return Dataset(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
        # sklearn: target_names[0]='malignant', [1]='benign'
        target_names=list(raw.target_names),
    )


def class_distribution(y: pd.Series) -> dict[str, int]:
    """Retorna a contagem por classe rotulada (maligno/benigno)."""
    counts = y.value_counts().to_dict()
    return {
        "maligno": int(counts.get(0, 0)),
        "benigno": int(counts.get(1, 0)),
    }
