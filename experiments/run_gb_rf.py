"""Compara baseline vs. otimizado para Gradient Boosting e Random Forest.

Complementa ``run_experiments.py`` (que varre as configurações do GA sobre o
SVM) aplicando o GA aos dois modelos de árvore da Fase 1. Serve à §3.2/§3.3 do
relatório: mostrar o que o GA faz quando o baseline tem espaço para melhorar.

Nota metodológica: a Fase 1 avaliou no test set apenas o modelo vencedor (SVM);
para GB e RF publicou somente validação cruzada. As colunas "baseline" aqui são
recalculadas treinando cada modelo com os hiperparâmetros *default* do
scikit-learn, no mesmo split da Fase 1 (``random_state=42``).

Uso:
    PYTHONPATH=src python experiments/run_gb_rf.py

Gera:
    results/gb_rf_comparison.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from diag_opt.data import load_dataset
from diag_opt.evaluation import compare_baseline_vs_optimized
from diag_opt.ga.engine import GAConfig, GeneticOptimizer
from diag_opt.ga.fitness import FitnessEvaluator
from diag_opt.models import get_model_spec
from diag_opt.monitoring.logging_config import setup_logging

MODELOS = ("GradientBoosting", "RandomForest")
CONFIG = GAConfig(population_size=20, generations=15, seed=42)


def main() -> None:
    logger = setup_logging()
    ds = load_dataset()
    saida: dict[str, dict] = {}

    for nome in MODELOS:
        logger.info("Otimizando %s via GA...", nome)
        evaluator = FitnessEvaluator(nome, ds.X, ds.y)

        # Fitness de CV do baseline (params default), na mesma métrica composta.
        baseline_params = get_model_spec(nome).baseline
        base_cv = evaluator.evaluate(baseline_params)

        result = GeneticOptimizer(nome, evaluator, CONFIG).run()
        comp = compare_baseline_vs_optimized(nome, result.best_chromosome, ds)
        base, opt = comp["baseline"], comp["optimized"]

        saida[nome] = {
            "baseline_params": baseline_params,
            "baseline_cv_fitness": round(base_cv.fitness, 4),
            "baseline_cv_recall": round(base_cv.metrics["recall"], 4),
            "optimized_cv_fitness": round(result.best_fitness, 4),
            "best_params": result.best_chromosome,
            "n_evaluations": result.n_evaluations,
            "elapsed_seconds": round(result.elapsed_seconds, 1),
            "acc_base": base.metrics["accuracy"],
            "acc_opt": opt.metrics["accuracy"],
            "recall_base": base.metrics["recall_maligno"],
            "recall_opt": opt.metrics["recall_maligno"],
            "fn_base": base.false_negatives,
            "fn_opt": opt.false_negatives,
            "fp_base": base.false_positives,
            "fp_opt": opt.false_positives,
        }
        logger.info(
            "%s | fitness CV %.4f -> %.4f | FN %d -> %d | FP %d -> %d",
            nome,
            saida[nome]["baseline_cv_fitness"],
            saida[nome]["optimized_cv_fitness"],
            saida[nome]["fn_base"],
            saida[nome]["fn_opt"],
            saida[nome]["fp_base"],
            saida[nome]["fp_opt"],
        )

    out = Path("results/gb_rf_comparison.json")
    out.write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    logger.info("Comparativo salvo em %s", out)


if __name__ == "__main__":
    main()
