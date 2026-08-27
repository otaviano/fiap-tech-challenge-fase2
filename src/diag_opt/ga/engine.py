"""Motor do Algoritmo Genético para otimização de hiperparâmetros.

Fluxo padrão de um GA geracional com elitismo:

    inicialização aleatória
        └─> [ avaliação -> seleção -> crossover -> mutação -> elitismo ] * G

O histórico por geração (melhor e média de fitness) é registrado para permitir
as análises de convergência exigidas no relatório.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from diag_opt.ga.encoding import Chromosome, random_chromosome
from diag_opt.ga.fitness import FitnessEvaluator, FitnessResult
from diag_opt.ga.operators import CROSSOVER, SELECTION, mutate
from diag_opt.models import get_model_spec


@dataclass
class GAConfig:
    """Configuração do algoritmo genético (variada nos experimentos)."""

    population_size: int = 20
    generations: int = 15
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elitism: int = 2
    tournament_size: int = 3
    selection: str = "tournament"  # "tournament" | "roulette" | "rank"
    crossover: str = "uniform"  # "uniform" | "one_point"
    seed: int = 42


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_chromosome: Chromosome
    best_metrics: dict[str, float]


@dataclass
class GAResult:
    best_chromosome: Chromosome
    best_fitness: float
    best_metrics: dict[str, float]
    history: list[GenerationStats]
    n_evaluations: int
    elapsed_seconds: float
    config: GAConfig
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": self.config.__dict__,
            "best_chromosome": self.best_chromosome,
            "best_fitness": self.best_fitness,
            "best_metrics": self.best_metrics,
            "n_evaluations": self.n_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "history": [
                {
                    "generation": g.generation,
                    "best_fitness": g.best_fitness,
                    "mean_fitness": g.mean_fitness,
                }
                for g in self.history
            ],
        }


class GeneticOptimizer:
    """Otimiza os hiperparâmetros de um modelo via algoritmo genético."""

    def __init__(
        self,
        model_name: str,
        evaluator: FitnessEvaluator,
        config: GAConfig | None = None,
        on_generation: Callable[[GenerationStats], None] | None = None,
    ) -> None:
        self.model_spec = get_model_spec(model_name)
        self.model_name = model_name
        self.evaluator = evaluator
        self.config = config or GAConfig()
        self.on_generation = on_generation
        self.rng = random.Random(self.config.seed)

    def _select(self, population: list[Chromosome], fitnesses: list[float]) -> Chromosome:
        selector = SELECTION[self.config.selection]
        if self.config.selection == "tournament":
            return selector(population, fitnesses, self.rng, k=self.config.tournament_size)
        return selector(population, fitnesses, self.rng)

    def _reproduce(self, p_a: Chromosome, p_b: Chromosome) -> tuple[Chromosome, Chromosome]:
        genes = self.model_spec.genes
        if self.rng.random() < self.config.crossover_rate:
            child_a, child_b = CROSSOVER[self.config.crossover](p_a, p_b, genes, self.rng)
        else:
            child_a, child_b = dict(p_a), dict(p_b)
        child_a = mutate(child_a, genes, self.rng, self.config.mutation_rate)
        child_b = mutate(child_b, genes, self.rng, self.config.mutation_rate)
        return child_a, child_b

    def run(self) -> GAResult:
        start = time.perf_counter()
        genes = self.model_spec.genes

        population: list[Chromosome] = [
            random_chromosome(genes, self.rng) for _ in range(self.config.population_size)
        ]

        best_chromo: Chromosome | None = None
        best_result: FitnessResult | None = None
        history: list[GenerationStats] = []

        for gen in range(self.config.generations):
            results = [self.evaluator.evaluate(ch) for ch in population]
            fitnesses = [r.fitness for r in results]

            gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
            if best_result is None or fitnesses[gen_best_idx] > best_result.fitness:
                best_result = results[gen_best_idx]
                best_chromo = dict(population[gen_best_idx])

            stats = GenerationStats(
                generation=gen,
                best_fitness=max(fitnesses),
                mean_fitness=sum(fitnesses) / len(fitnesses),
                best_chromosome=dict(population[gen_best_idx]),
                best_metrics=results[gen_best_idx].metrics,
            )
            history.append(stats)
            if self.on_generation:
                self.on_generation(stats)

            # Elitismo: preserva os melhores indivíduos intactos.
            ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            next_pop: list[Chromosome] = [dict(population[i]) for i in ranked[: self.config.elitism]]

            while len(next_pop) < self.config.population_size:
                parent_a = self._select(population, fitnesses)
                parent_b = self._select(population, fitnesses)
                child_a, child_b = self._reproduce(parent_a, parent_b)
                next_pop.append(child_a)
                if len(next_pop) < self.config.population_size:
                    next_pop.append(child_b)

            population = next_pop

        assert best_chromo is not None and best_result is not None
        return GAResult(
            best_chromosome=best_chromo,
            best_fitness=best_result.fitness,
            best_metrics=best_result.metrics,
            history=history,
            n_evaluations=self.evaluator.n_evaluations,
            elapsed_seconds=time.perf_counter() - start,
            config=self.config,
            model_name=self.model_name,
        )
