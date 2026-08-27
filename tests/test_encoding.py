import random

import pytest

from diag_opt.ga.encoding import clip_value, random_chromosome, random_value
from diag_opt.models import GeneSpec, MODELS


def test_genespec_validations():
    with pytest.raises(ValueError):
        GeneSpec("x", "float", low=10, high=1)  # low > high
    with pytest.raises(ValueError):
        GeneSpec("x", "cat")  # sem choices
    with pytest.raises(ValueError):
        GeneSpec("x", "float", low=-1, high=1, log=True)  # log com low<=0


@pytest.mark.parametrize("model_name", list(MODELS))
def test_random_chromosome_dentro_do_dominio(model_name):
    rng = random.Random(1)
    genes = MODELS[model_name].genes
    chromo = random_chromosome(genes, rng)
    assert set(chromo) == {g.name for g in genes}
    for gene in genes:
        v = chromo[gene.name]
        if gene.kind == "cat":
            assert v in gene.choices
        else:
            assert gene.low <= v <= gene.high


def test_random_value_int_e_log():
    rng = random.Random(0)
    gint = GeneSpec("n", "int", low=1, high=5)
    for _ in range(50):
        v = random_value(gint, rng)
        assert isinstance(v, int) and 1 <= v <= 5

    glog = GeneSpec("c", "float", low=1e-3, high=1e3, log=True)
    for _ in range(50):
        v = random_value(glog, rng)
        assert 1e-3 <= v <= 1e3


def test_clip_value_respeita_limites():
    g = GeneSpec("n", "int", low=2, high=10)
    assert clip_value(g, 100) == 10
    assert clip_value(g, -5) == 2
    gc = GeneSpec("k", "cat", choices=("a", "b"))
    assert clip_value(gc, "z") == "a"  # inválido -> primeira opção


def test_random_forest_tem_gene_bootstrap_booleano():
    """O cromossomo do RF cobre os 6 hiperparâmetros do exemplo canônico."""
    genes = {g.name: g for g in MODELS["RandomForest"].genes}
    assert set(genes) == {
        "n_estimators",
        "max_features",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "bootstrap",
    }
    assert genes["bootstrap"].kind == "cat"
    assert set(genes["bootstrap"].choices) == {True, False}
