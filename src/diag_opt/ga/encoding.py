"""Codificação e amostragem de genes (representação dos indivíduos).

Um **indivíduo** (cromossomo) é representado como um dicionário
``{nome_do_hiperparâmetro: valor}`` — cada par é um **gene**. Essa representação
fenotípica direta torna a leitura do relatório mais clara: cada gene mapeia
1:1 para um hiperparâmetro do modelo.

A amostragem respeita o domínio declarado em cada :class:`GeneSpec`:

* ``float`` com ``log=True`` é amostrado em escala logarítmica (ideal para C,
  gamma, learning_rate, cujas ordens de grandeza importam);
* ``int`` é amostrado uniformemente e arredondado;
* ``cat`` escolhe uniformemente entre as opções.
"""

from __future__ import annotations

import math
import random
from typing import Any

from diag_opt.models import GeneSpec

Chromosome = dict[str, Any]


def random_value(gene: GeneSpec, rng: random.Random) -> Any:
    """Amostra um valor válido para um gene."""
    if gene.kind == "cat":
        return rng.choice(gene.choices)
    if gene.kind == "int":
        return rng.randint(int(gene.low), int(gene.high))
    # float
    if gene.log:
        lo, hi = math.log10(gene.low), math.log10(gene.high)
        return 10 ** rng.uniform(lo, hi)
    return rng.uniform(gene.low, gene.high)


def random_chromosome(genes: tuple[GeneSpec, ...], rng: random.Random) -> Chromosome:
    """Gera um indivíduo aleatório respeitando todos os domínios."""
    return {gene.name: random_value(gene, rng) for gene in genes}


def clip_value(gene: GeneSpec, value: Any) -> Any:
    """Garante que um valor perturbado continue dentro do domínio do gene."""
    if gene.kind == "cat":
        return value if value in gene.choices else gene.choices[0]
    if gene.kind == "int":
        return int(min(max(round(value), gene.low), gene.high))
    return float(min(max(value, gene.low), gene.high))
