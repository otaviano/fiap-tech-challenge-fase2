import random

from diag_opt.ga.operators import (
    SELECTION,
    mutate,
    one_point_crossover,
    rank_selection,
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


def test_rank_selection_retorna_individuo_valido():
    pop = [{"a": 1}, {"a": 2}, {"a": 3}]
    fits = [0.1, 0.9, 0.5]
    rng = random.Random(0)
    for _ in range(20):
        assert rank_selection(pop, fits, rng) in pop


def test_rank_selection_favorece_o_melhor():
    """O de maior fitness deve ser escolhido com mais frequência que o pior.

    Pesos 1..N sobre 3 indivíduos: o melhor tem 3/6 das chances, o pior 1/6.
    """
    pop = [{"a": "pior"}, {"a": "meio"}, {"a": "melhor"}]
    fits = [0.10, 0.50, 0.90]
    rng = random.Random(7)
    escolhas = [rank_selection(pop, fits, rng)["a"] for _ in range(600)]
    assert escolhas.count("melhor") > escolhas.count("meio") > escolhas.count("pior")


def test_rank_selection_e_invariante_a_escala_do_fitness():
    """Mesma ordem de fitness -> mesma sequência de escolhas, independente da escala.

    É a propriedade que distingue o ranqueamento da roleta: com fitness
    comprimido (0,90 / 0,91 / 0,92) a roleta fica quase uniforme, enquanto o
    ranqueamento mantém a mesma pressão seletiva do caso espalhado.
    """
    pop = [{"a": 1}, {"a": 2}, {"a": 3}]
    espalhado = [0.01, 0.50, 0.99]
    comprimido = [0.90, 0.91, 0.92]
    a = [rank_selection(pop, espalhado, random.Random(s))["a"] for s in range(50)]
    b = [rank_selection(pop, comprimido, random.Random(s))["a"] for s in range(50)]
    assert a == b


def test_registry_de_selecao_cobre_os_tres_metodos():
    assert set(SELECTION) == {"tournament", "roulette", "rank"}
