# Roteiro de gravação — Bloco 3: O algoritmo genético por dentro

> Duração alvo: **3 a 4 minutos**. Arquivos na tela, na ordem:
> `src/diag_opt/models.py` → `src/diag_opt/ga/fitness.py` → `src/diag_opt/ga/operators.py` → `src/diag_opt/ga/engine.py`.

**Tese do bloco, em uma frase:** três arquivos, três responsabilidades — `models.py` transforma o problema em genoma, `fitness.py` transforma em número, `operators.py` faz a busca. Nada do GA sabe o que é scikit-learn, e nada do scikit-learn sabe o que é GA.

---

## 3.1 · `models.py` — onde o problema vira genoma (~60s)

**Na tela:** `models.py:30-60` (a classe `GeneSpec`), depois `models.py:99-147` (o dicionário `MODELS`).

- **`GeneSpec` é o contrato.** Um gene = um hiperparâmetro, com nome, tipo, faixa e escala. Três tipos: `float`, `int` e `cat`. Isso significa que nosso cromossomo é **heterogêneo** — não é uma cadeia de bits, é `{"C": 2.61, "gamma": 0.0336, "kernel": "rbf"}`. Representação fenotípica direta: dá pra olhar um indivíduo e entender o modelo que ele descreve, sem decodificar nada.

- **O detalhe que vale a nota — `log=True`** (`models.py:107-108`). O `C` do SVM varia de 0,01 a 1000. Se eu sorteasse uniformemente nessa faixa, mais de 99% dos sorteios cairiam acima de 1 — o GA nunca visitaria a região dos valores pequenos. Amostrando em escala logarítmica (`encoding.py:34-36`), **cada ordem de grandeza tem a mesma chance**. É isso que faz ele encontrar `gamma = 0,03` em 15 gerações.

- **`__post_init__` valida o espaço de busca** (`models.py:48-60`): `low > high`, escala log com `low <= 0`, categórico sem opções — tudo isso falha no carregamento, não no meio de 100 avaliações de validação cruzada.

- **`ModelSpec` agrega o que o GA precisa saber:** fábrica do estimador, se precisa de escala, o baseline da Fase 1 e a tupla de genes. **Adicionar um quarto modelo é uma entrada nesse dicionário — zero linha alterada no GA.** É o ponto de extensibilidade do projeto.

- **O baseline está versionado no código de propósito** — a comparação "original vs otimizado" é entregável do desafio, então os hiperparâmetros da Fase 1 ficam explícitos, não implícitos nos defaults do scikit-learn.

- **`build_pipeline`** (`models.py:156-164`): o `StandardScaler` entra **dentro** do `Pipeline`. Consequência: ele é reajustado dentro de cada fold da validação cruzada — **sem vazamento de dados do teste para o treino**.

- *Detalhe fino, se sobrar tempo:* o gene `bootstrap` do Random Forest (`models.py:132`) mostra que um booleano é só um categórico de duas opções — o mesmo mecanismo liga e desliga o bagging.

---

## 3.2 · `ga/fitness.py` — onde o problema vira número (~70s)

**Na tela:** `fitness.py:27-58`, depois o método `evaluate` em `fitness.py:87-113`.

- **Este é o único arquivo que sabe o que é um "bom" modelo.** O fitness é `0,6 · recall + 0,3 · F1 + 0,1 · ROC-AUC`, normalizado pela soma dos pesos.

- **Por que o recall domina:** recall e F1 são medidos sobre a **classe maligna** (`pos_label=0`, `fitness.py:53-56`). Falso negativo aqui é um câncer não detectado. Acurácia sozinha seria enganosa num dataset com 357 benignos contra 212 malignos — um modelo que chuta "benigno" sempre acerta 63%.

- **Os pesos são configuráveis** (`FitnessConfig`) — é exatamente isso que permite rodar o experimento que varia a função de fitness sem tocar no motor.

- **Validação cruzada estratificada, 5 folds:** cada indivíduo é treinado e avaliado 5 vezes, e o fitness é a média. Isso impede o GA de sobreajustar a uma partição sortuda — com um único split, ele otimizaria o split, não o modelo.

- **O cache** (`fitness.py:83-91`) — vale mostrar o número: com elitismo e convergência, a população repete indivíduos. No experimento baseline eram **300 avaliações potenciais** (20 × 15) e o GA rodou **99 validações cruzadas reais**. Um terço do custo, resultado idêntico.

- **O `except` que vira penalização** (`fitness.py:105-108`): combinação inválida de hiperparâmetros não derruba a execução — recebe fitness zero e morre naturalmente na seleção. É tratamento de restrição por penalização, feito no lugar certo.

- `n_evaluations` conta validações cruzadas de verdade — é a métrica de custo computacional que aparece no relatório.

---

## 3.3 · `ga/operators.py` — onde a busca acontece (~80s)

**Na tela:** `operators.py:52-81` (seleção por ranqueamento) e `operators.py:136-159` (mutação).

- **Três estratégias de seleção, e o interessante é *por que* a terceira existe.** Torneio (o `k` controla a pressão seletiva), roleta e ranqueamento.

- **O argumento forte do bloco — a roleta quase não funciona neste problema.** Nosso fitness é uma combinação de métricas de classificação, então a população inteira vive entre ~0,90 e 0,97. Na roleta, o melhor indivíduo tira 0,97 da soma e o pior 0,90 — probabilidades **quase idênticas**. A pressão seletiva evapora e o GA degenera em busca aleatória. O **ranqueamento é invariante à escala**: ordena do pior pro melhor e atribui pesos 1 a N pela *posição*. O melhor sempre pesa N, o pior sempre 1 — tanto faz se a diferença de fitness é 0,001 ou 0,5.

- **Dois crossovers:** uniforme (cada gene sorteado independentemente — mistura agressiva) e de um ponto (troca blocos contíguos — preserva combinações de genes vizinhos que estejam funcionando juntas).

- **A mutação é híbrida, e isso é intencional** (`operators.py:152-158`): categórico sorteia uma nova categoria; numérico recebe uma **perturbação gaussiana com desvio de 10% da amplitude do gene**. Ou seja, mutação numérica é **busca local** — refina em volta do valor atual em vez de jogar o indivíduo pro outro canto do espaço. O `clip_value` garante que o resultado continua dentro do domínio.

- **Todo operador recebe um `random.Random` explícito.** Semente fixa = execução reprodutível. É o que torna os cinco experimentos do relatório verificáveis.

---

## 3.4 · `ga/engine.py` — a cola, em 20 segundos

**Na tela:** o laço principal em `engine.py:125-157`.

- População aleatória → laço: **avaliar → elitismo → seleção → crossover (80%) → mutação**, por G gerações.
- **Elitismo = 2:** os dois melhores passam intactos. Consequência prática: o melhor fitness **nunca regride** entre gerações — a curva de convergência do relatório é monotônica por construção.
- Cada geração registra melhor e média de fitness — é o gráfico de convergência.
- O `GAResult` carrega config, número de avaliações e tempo, então **cada JSON em `results/` é autocontido e reexecutável**.

---

## 3.5 · Fechamento com números (~30s)

| Experimento | Seleção / Crossover | Fitness | Avaliações |
| --- | --- | --- | --- |
| `baseline_ga` | torneio / uniforme | 0,9726 | 99 |
| `roulette_onepoint` | roleta / um ponto | **0,9762** | 144 |
| `high_mutation` (0,35) | torneio / uniforme | 0,9655 | 154 |
| `large_population` (40×10) | torneio / uniforme | 0,9655 | 156 |
| `rank_selection` | ranqueamento / uniforme | 0,9655 | 113 |

**Fala de encerramento:** os cinco experimentos, com estratégias diferentes, convergiram todos para kernel `rbf` com gamma pequeno (0,003 a 0,034). Convergência independente da configuração é o sinal de que a região ótima é real, não sorte de semente. E no Gradient Boosting o ganho em fitness de validação cruzada foi de **0,9216 → 0,9655**.

---

## Na manga — se a banca perguntar

No hold-out de teste o SVM otimizado ficou com acurácia **0,9737** contra **0,9825** do baseline, com **o mesmo recall e o mesmo 1 falso negativo**.

A resposta honesta: o baseline do SVM já estava muito perto do teto neste dataset, e nossa função de fitness prioriza recall — ela trocou um pouco de acurácia por robustez na classe que importa, exatamente como foi projetada. O ganho real do GA aparece nos modelos com espaço de busca maior, como o Gradient Boosting.
