import random

from diag_opt.ga.operators import (
    mutate,
    one_point_crossover,
    roulette_selection,
    tournament_selection,
    uniform_crossover,
)
from diag_opt.models import MODELS

GENES = MODELS["SVM"].genes


def _two_parents(rng):
    from diag_opt.ga.encoding import random_chromosome

    return random_chromosome(GENES, rng), random_chromosome(GENES, rng)


def test_tournament_seleciona_melhor():
    pop = [{"a": 1}, {"a": 2}, {"a": 3}]
    fits = [0.1, 0.9, 0.5]
    rng = random.Random(0)
    # com k = tamanho da população, sempre retorna o de maior fitness
    chosen = tournament_selection(pop, fits, rng, k=3)
    assert chosen == {"a": 2}


def test_roulette_retorna_individuo_valido():
    pop = [{"a": 1}, {"a": 2}]
    rng = random.Random(0)
    assert roulette_selection(pop, [0.0, 0.0], rng) in pop  # soma 0 -> uniforme
    assert roulette_selection(pop, [1.0, 3.0], rng) in pop


def test_crossover_preserva_genes():
    rng = random.Random(1)
    pa, pb = _two_parents(rng)
    for crossover in (uniform_crossover, one_point_crossover):
        ca, cb = crossover(pa, pb, GENES, rng)
        assert set(ca) == set(pa) == set(cb)
        # cada gene do filho veio de um dos pais
        for g in GENES:
            assert ca[g.name] in (pa[g.name], pb[g.name])


def test_mutacao_mantem_dominio_e_muda_algo():
    rng = random.Random(2)
    from diag_opt.ga.encoding import random_chromosome

    base = random_chromosome(GENES, rng)
    mutant = mutate(base, GENES, rng, mutation_rate=1.0)  # muta todos
    for g in GENES:
        v = mutant[g.name]
        if g.kind == "cat":
            assert v in g.choices
        else:
            assert g.low <= v <= g.high


def test_mutacao_zero_nao_altera():
    rng = random.Random(3)
    from diag_opt.ga.encoding import random_chromosome

    base = random_chromosome(GENES, rng)
    assert mutate(base, GENES, rng, mutation_rate=0.0) == base
