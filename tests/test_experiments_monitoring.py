import json

from diag_opt.experiments import run_experiment
from diag_opt.ga.engine import GAConfig, GAResult, GenerationStats
from diag_opt.monitoring.logging_config import setup_logging
from diag_opt.monitoring.tracker import RunTracker


def test_setup_logging_cria_logger(tmp_path):
    logger = setup_logging("teste_log", log_dir=tmp_path)
    logger.info("mensagem de teste")
    assert (tmp_path / "teste_log.log").exists()


def test_run_tracker_persiste_json(tmp_path):
    logger = setup_logging("tracker_test", log_dir=tmp_path)
    tracker = RunTracker("exp_demo", logger=logger, out_dir=tmp_path)
    result = GAResult(
        best_chromosome={"C": 1.0},
        best_fitness=0.9,
        best_metrics={"recall": 0.95},
        history=[GenerationStats(0, 0.9, 0.5, {"C": 1.0}, {"recall": 0.95})],
        n_evaluations=3,
        elapsed_seconds=1.2,
        config=GAConfig(),
        model_name="SVM",
    )
    tracker.on_generation(result.history[0])
    out = tracker.finish(result)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["experiment"] == "exp_demo"
    assert data["result"]["best_fitness"] == 0.9


def test_run_experiment_gera_comparacao(dataset, tmp_path, monkeypatch):
    # GA minúsculo para manter o teste rápido
    monkeypatch.chdir(tmp_path)
    cfg = GAConfig(population_size=4, generations=2, seed=1)
    outcome = run_experiment("mini", cfg, "SVM", dataset)
    assert outcome.name == "mini"
    assert "metrics" in outcome.baseline_test
    assert "metrics" in outcome.optimized_test
    assert outcome.ga_result["best_fitness"] > 0.5
