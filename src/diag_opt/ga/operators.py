"""Operadores genéticos: seleção, cruzamento (crossover) e mutação.

Todos operam sobre a representação de cromossomo como dicionário de genes e
recebem um ``random.Random`` explícito, garantindo reprodutibilidade quando uma
semente é fixada.
"""

from __future__ import annotations

import random

from diag_opt.models import GeneSpec
from diag_opt.ga.encoding import Chromosome, clip_value, random_value


# --- Seleção ------------------------------------------------------------------

def tournament_selection(
    population: list[Chromosome],
    fitnesses: list[float],
    rng: random.Random,
    k: int = 3,
) -> Chromosome:
    """Seleção por torneio: sorteia ``k`` indivíduos e retorna o de maior fitness.

    Pressão seletiva controlável por ``k`` — quanto maior, mais elitista.
    """
    n = len(population)
    aspirants = rng.sample(range(n), min(k, n))
    best = max(aspirants, key=lambda i: fitnesses[i])
    return dict(population[best])


def roulette_selection(
    population: list[Chromosome],
    fitnesses: list[float],
    rng: random.Random,
) -> Chromosome:
    """Seleção proporcional ao fitness (roleta). Fallback uniforme se soma <= 0."""
    total = sum(fitnesses)
    if total <= 0:
        return dict(rng.choice(population))
    pick = rng.uniform(0, total)
    acc = 0.0
    for chromo, fit in zip(population, fitnesses):
        acc += fit
        if acc >= pick:
            return dict(chromo)
    return dict(population[-1])


def rank_selection(
    population: list[Chromosome],
    fitnesses: list[float],
    rng: random.Random,
) -> Chromosome:
    """Seleção por ranqueamento: probabilidade proporcional à **posição**.

    Os indivíduos são ordenados do pior para o melhor e recebem pesos
    ``1, 2, ..., N``; o sorteio é proporcional a esse peso, não ao valor bruto
    do fitness.

    Motivação prática neste projeto: nosso fitness é uma combinação de métricas
    de classificação, então quase toda a população vive num intervalo estreito
    (ex.: 0,90 a 0,97). Na roleta isso torna as probabilidades quase uniformes —
    a pressão seletiva praticamente desaparece. O ranqueamento é **invariante à
    escala** do fitness: o melhor indivíduo tem sempre peso ``N`` e o pior peso
    ``1``, independentemente de a diferença entre eles ser 0,001 ou 0,5.
    """
    n = len(population)
    if n == 0:
        raise ValueError("População vazia")
    order = sorted(range(n), key=lambda i: fitnesses[i])  # pior -> melhor
    total = n * (n + 1) / 2  # soma dos pesos 1..N
    pick = rng.uniform(0, total)
    acc = 0.0
    for rank, idx in enumerate(order, start=1):
        acc += rank
        if acc >= pick:
            return dict(population[idx])
    return dict(population[order[-1]])


SELECTION = {
    "tournament": tournament_selection,
    "roulette": roulette_selection,
    "rank": rank_selection,
}


# --- Cruzamento ---------------------------------------------------------------

def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    genes: tuple[GeneSpec, ...],
    rng: random.Random,
) -> tuple[Chromosome, Chromosome]:
    """Crossover uniforme: cada gene é herdado de um dos pais com prob. 0,5."""
    child_a: Chromosome = {}
    child_b: Chromosome = {}
    for gene in genes:
        if rng.random() < 0.5:
            child_a[gene.name] = parent_a[gene.name]
            child_b[gene.name] = parent_b[gene.name]
        else:
            child_a[gene.name] = parent_b[gene.name]
            child_b[gene.name] = parent_a[gene.name]
    return child_a, child_b


def one_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    genes: tuple[GeneSpec, ...],
    rng: random.Random,
) -> tuple[Chromosome, Chromosome]:
    """Crossover de um ponto: troca o segmento de genes após um corte aleatório."""
    names = [g.name for g in genes]
    if len(names) < 2:
        return dict(parent_a), dict(parent_b)
    point = rng.randint(1, len(names) - 1)
    child_a = {n: (parent_a if i < point else parent_b)[n] for i, n in enumerate(names)}
    child_b = {n: (parent_b if i < point else parent_a)[n] for i, n in enumerate(names)}
    return child_a, child_b


CROSSOVER = {
    "uniform": uniform_crossover,
    "one_point": one_point_crossover,
}


# --- Mutação ------------------------------------------------------------------

def mutate(
    chromosome: Chromosome,
    genes: tuple[GeneSpec, ...],
    rng: random.Random,
    mutation_rate: float,
) -> Chromosome:
    """Mutação gene a gene: com prob. ``mutation_rate`` reamostra o gene.

    Para genes numéricos aplica uma perturbação gaussiana em torno do valor
    atual (busca local), reamostrando totalmente quando o resultado sai do
    domínio; para categóricos, sorteia uma nova categoria.
    """
    mutant = dict(chromosome)
    for gene in genes:
        if rng.random() >= mutation_rate:
            continue
        if gene.kind == "cat":
            mutant[gene.name] = random_value(gene, rng)
        else:
            current = mutant[gene.name]
            span = gene.high - gene.low
            perturbed = current + rng.gauss(0, 0.1 * span)
            mutant[gene.name] = clip_value(gene, perturbed)
    return mutant
