"""Executa todos os experimentos de GA e consolida os resultados.

Uso:
    python experiments/run_experiments.py [--model SVM]

Gera:
    results/experiments_summary.json   (comparativo consolidado)
    results/run_<modelo>_<exp>.json     (um por experimento, via RunTracker)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from diag_opt.data import class_distribution, load_dataset
from diag_opt.experiments import run_all_experiments
from diag_opt.monitoring.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SVM")
    args = parser.parse_args()

    logger = setup_logging()
    ds = load_dataset()
    logger.info("Dataset carregado: %s | distribuição=%s", ds.X.shape, class_distribution(ds.y))

    outcomes = run_all_experiments(model_name=args.model, ds=ds, logger=logger)

    summary = {
        "model": args.model,
        "dataset": {
            "n_amostras": int(ds.X.shape[0]),
            "n_features": int(ds.X.shape[1]),
            "distribuicao": class_distribution(ds.y),
        },
        "experiments": [o.to_dict() for o in outcomes],
        "tabela": [
            {
                "experimento": o.name,
                "populacao": o.ga_result["config"]["population_size"],
                "geracoes": o.ga_result["config"]["generations"],
                "taxa_mutacao": o.ga_result["config"]["mutation_rate"],
                "selecao": o.ga_result["config"]["selection"],
                "crossover": o.ga_result["config"]["crossover"],
                "best_fitness": round(o.ga_result["best_fitness"], 4),
                "best_params": o.ga_result["best_chromosome"],
                "avaliacoes": o.ga_result["n_evaluations"],
                "tempo_s": round(o.ga_result["elapsed_seconds"], 1),
                "acc_baseline": round(o.baseline_test["metrics"]["accuracy"], 4),
                "acc_otimizado": round(o.optimized_test["metrics"]["accuracy"], 4),
                "recall_baseline": round(o.baseline_test["metrics"]["recall_maligno"], 4),
                "recall_otimizado": round(o.optimized_test["metrics"]["recall_maligno"], 4),
                "f1_baseline": round(o.baseline_test["metrics"]["f1_maligno"], 4),
                "f1_otimizado": round(o.optimized_test["metrics"]["f1_maligno"], 4),
                "fn_baseline": o.baseline_test["false_negatives"],
                "fn_otimizado": o.optimized_test["false_negatives"],
            }
            for o in outcomes
        ],
    }

    out = Path("results/experiments_summary.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Resumo consolidado salvo em %s", out)

    print("\n=== RESUMO DOS EXPERIMENTOS ===")
    for row in summary["tabela"]:
        print(
            f"{row['experimento']:>18} | fit={row['best_fitness']:.4f} "
            f"| recall {row['recall_baseline']:.3f}->{row['recall_otimizado']:.3f} "
            f"| FN {row['fn_baseline']}->{row['fn_otimizado']} | {row['tempo_s']}s"
        )


if __name__ == "__main__":
    main()
