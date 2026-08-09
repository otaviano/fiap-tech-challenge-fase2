"""Definição e execução dos experimentos de algoritmo genético.

O enunciado exige ao menos 3 experimentos com configurações diferentes
(tamanho de população, taxa de mutação etc.). Definimos 4 configurações que
variam população, taxa de mutação, método de seleção e tipo de crossover, para
permitir uma análise comparativa rica no relatório.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from diag_opt.data import Dataset
from diag_opt.evaluation import compare_baseline_vs_optimized
from diag_opt.ga.engine import GAConfig, GeneticOptimizer
from diag_opt.ga.fitness import FitnessConfig, FitnessEvaluator
from diag_opt.monitoring.tracker import RunTracker

# --- As configurações dos experimentos ----------------------------------------

EXPERIMENTS: dict[str, GAConfig] = {
    # 1) Configuração de referência: torneio + crossover uniforme.
    "baseline_ga": GAConfig(
        population_size=20, generations=15, crossover_rate=0.8,
        mutation_rate=0.15, selection="tournament", crossover="uniform", seed=42,
    ),
    # 2) Mutação alta: mais exploração, testa fuga de ótimos locais.
    "high_mutation": GAConfig(
        population_size=20, generations=15, crossover_rate=0.8,
        mutation_rate=0.35, selection="tournament", crossover="uniform", seed=42,
    ),
    # 3) População grande / menos gerações: mais diversidade por geração.
    "large_population": GAConfig(
        population_size=40, generations=10, crossover_rate=0.8,
        mutation_rate=0.15, selection="tournament", crossover="uniform", seed=42,
    ),
    # 4) Seleção por roleta + crossover de 1 ponto: outro regime de busca.
    "roulette_onepoint": GAConfig(
        population_size=20, generations=15, crossover_rate=0.9,
        mutation_rate=0.15, selection="roulette", crossover="one_point", seed=42,
    ),
}


@dataclass
class ExperimentOutcome:
    name: str
    model_name: str
    ga_result: dict[str, Any]
    baseline_test: dict[str, Any]
    optimized_test: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "ga_result": self.ga_result,
            "baseline_test": self.baseline_test,
            "optimized_test": self.optimized_test,
        }


def run_experiment(
    name: str,
    config: GAConfig,
    model_name: str,
    ds: Dataset,
    fitness_config: FitnessConfig | None = None,
    logger: logging.Logger | None = None,
) -> ExperimentOutcome:
    """Roda um experimento: GA -> melhor indivíduo -> avaliação no test set."""
    evaluator = FitnessEvaluator(model_name, ds.X, ds.y, fitness_config)
    tracker = RunTracker(experiment=f"{model_name}_{name}", logger=logger)

    optimizer = GeneticOptimizer(model_name, evaluator, config, on_generation=tracker.on_generation)
    result = optimizer.run()

    comparison = compare_baseline_vs_optimized(model_name, result.best_chromosome, ds)
    tracker.finish(
        result,
        extra={
            "baseline_test": comparison["baseline"].summary(),
            "optimized_test": comparison["optimized"].summary(),
        },
    )

    return ExperimentOutcome(
        name=name,
        model_name=model_name,
        ga_result=result.to_dict(),
        baseline_test=comparison["baseline"].summary(),
        optimized_test=comparison["optimized"].summary(),
    )


def run_all_experiments(
    model_name: str = "SVM",
    ds: Dataset | None = None,
    fitness_config: FitnessConfig | None = None,
    logger: logging.Logger | None = None,
    only: list[str] | None = None,
) -> list[ExperimentOutcome]:
    """Executa todos os experimentos definidos em ``EXPERIMENTS``."""
    from diag_opt.data import load_dataset

    ds = ds or load_dataset()
    names = only or list(EXPERIMENTS)
    outcomes: list[ExperimentOutcome] = []
    for name in names:
        outcomes.append(
            run_experiment(name, EXPERIMENTS[name], model_name, ds, fitness_config, logger)
        )
    return outcomes
