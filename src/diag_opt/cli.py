"""Interface de linha de comando do sistema de otimização de diagnóstico.

Exemplos:
    diag-opt experiments --model SVM        # roda os 4 experimentos do GA
    diag-opt optimize --model RandomForest  # roda 1 GA e mostra o melhor indivíduo
    diag-opt interpret --index 0            # interpreta um caso do test set via LLM
"""

from __future__ import annotations

import argparse
import json
import sys

from diag_opt.data import load_dataset
from diag_opt.evaluation import compare_baseline_vs_optimized, fit_serving_model
from diag_opt.experiments import EXPERIMENTS, run_all_experiments, run_experiment
from diag_opt.ga.engine import GAConfig, GeneticOptimizer
from diag_opt.ga.fitness import FitnessEvaluator
from diag_opt.llm.client import LLMClient
from diag_opt.llm.interpreter import interpret_patient
from diag_opt.llm.quality import evaluate_interpretation
from diag_opt.monitoring.logging_config import setup_logging


def _cmd_optimize(args: argparse.Namespace) -> int:
    logger = setup_logging()
    ds = load_dataset()
    evaluator = FitnessEvaluator(args.model, ds.X, ds.y)
    config = GAConfig(population_size=args.population, generations=args.generations)
    optimizer = GeneticOptimizer(
        args.model, evaluator, config,
        on_generation=lambda s: logger.info(
            "gen %02d best=%.4f mean=%.4f", s.generation, s.best_fitness, s.mean_fitness
        ),
    )
    result = optimizer.run()
    comp = compare_baseline_vs_optimized(args.model, result.best_chromosome, ds)
    print(json.dumps(
        {
            "best_chromosome": result.best_chromosome,
            "best_fitness": result.best_fitness,
            "baseline_test": comp["baseline"].metrics,
            "optimized_test": comp["optimized"].metrics,
        },
        indent=2, ensure_ascii=False, default=str,
    ))
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    logger = setup_logging()
    outcomes = run_all_experiments(model_name=args.model, logger=logger)
    summary = [
        {
            "experimento": o.name,
            "best_fitness": round(o.ga_result["best_fitness"], 4),
            "recall_baseline": round(o.baseline_test["metrics"]["recall_maligno"], 4),
            "recall_otimizado": round(o.optimized_test["metrics"]["recall_maligno"], 4),
            "fn_baseline": o.baseline_test["false_negatives"],
            "fn_otimizado": o.optimized_test["false_negatives"],
        }
        for o in outcomes
    ]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_interpret(args: argparse.Namespace) -> int:
    ds = load_dataset()
    # Usa o melhor indivíduo conhecido do SVM como modelo de serviço.
    params = {"C": args.C, "gamma": args.gamma, "kernel": "rbf"}
    pipeline = fit_serving_model("SVM", params, ds)
    client = LLMClient()
    interp = interpret_patient(pipeline, ds, args.index, client=client)
    quality = evaluate_interpretation(interp.text, interp.context)
    print(f"[fonte: {interp.source} | qualidade: {quality.score:.2f}]\n")
    print(interp.text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diag-opt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_opt = sub.add_parser("optimize", help="Roda 1 GA e compara com o baseline")
    p_opt.add_argument("--model", default="SVM", choices=["SVM", "RandomForest", "GradientBoosting"])
    p_opt.add_argument("--population", type=int, default=20)
    p_opt.add_argument("--generations", type=int, default=15)
    p_opt.set_defaults(func=_cmd_optimize)

    p_exp = sub.add_parser("experiments", help=f"Roda os experimentos: {list(EXPERIMENTS)}")
    p_exp.add_argument("--model", default="SVM", choices=["SVM", "RandomForest", "GradientBoosting"])
    p_exp.set_defaults(func=_cmd_experiments)

    p_int = sub.add_parser("interpret", help="Interpreta um caso do test set via LLM")
    p_int.add_argument("--index", type=int, default=0)
    p_int.add_argument("--C", type=float, default=13.19)
    p_int.add_argument("--gamma", type=float, default=0.0689)
    p_int.set_defaults(func=_cmd_interpret)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
