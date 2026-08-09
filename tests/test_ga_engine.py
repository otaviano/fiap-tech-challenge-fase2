from diag_opt.ga.engine import GAConfig, GeneticOptimizer
from diag_opt.ga.fitness import FitnessConfig, FitnessEvaluator


def test_ga_converge_e_respeita_config(dataset):
    ev = FitnessEvaluator("SVM", dataset.X, dataset.y, FitnessConfig(cv_folds=3))
    cfg = GAConfig(population_size=6, generations=4, seed=42)
    res = GeneticOptimizer("SVM", ev, cfg).run()

    # histórico com uma entrada por geração
    assert len(res.history) == cfg.generations
    # o melhor fitness não piora ao longo das gerações (elitismo + tracking do global)
    best_per_gen = [g.best_fitness for g in res.history]
    assert res.best_fitness >= max(best_per_gen) - 1e-9
    # fitness plausível para o dataset (bem acima de acerto aleatório)
    assert res.best_fitness > 0.8
    # cromossomo final é válido
    assert set(res.best_chromosome) == {"C", "gamma", "kernel"}


def test_ga_reprodutivel_com_mesma_seed(dataset):
    def run():
        ev = FitnessEvaluator("SVM", dataset.X, dataset.y, FitnessConfig(cv_folds=3))
        cfg = GAConfig(population_size=6, generations=3, seed=7)
        return GeneticOptimizer("SVM", ev, cfg).run()

    r1, r2 = run(), run()
    assert r1.best_chromosome == r2.best_chromosome
    assert r1.best_fitness == r2.best_fitness


def test_fitness_cache_evita_reavaliacoes(dataset):
    ev = FitnessEvaluator("SVM", dataset.X, dataset.y, FitnessConfig(cv_folds=3))
    chromo = {"C": 1.0, "gamma": 0.1, "kernel": "rbf"}
    r1 = ev.evaluate(chromo)
    n_after_first = ev.n_evaluations
    r2 = ev.evaluate(dict(chromo))
    assert ev.n_evaluations == n_after_first  # cache: não reavaliou
    assert r1.fitness == r2.fitness
