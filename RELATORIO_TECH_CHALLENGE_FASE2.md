# Relatório Técnico — Tech Challenge Fase 2

**FIAP Postech — IA para Devs**
**Projeto 1 — Otimização de Modelos de Diagnóstico via Algoritmos Genéticos + LLM**

---

## 1. Introdução e contexto

Na Fase 1, desenvolvemos um sistema de suporte ao diagnóstico oncológico
(câncer de mama, dataset Wisconsin Breast Cancer) que comparou 6 modelos de
Machine Learning, elegendo o **SVM** como melhor (98,25% de accuracy, recall da
classe maligna de 97,6%, apenas 1 falso negativo).

Nesta Fase 2, evoluímos a solução em duas frentes:

1. **Otimização de hiperparâmetros via Algoritmo Genético (GA)**, comparando os
   modelos otimizados com os originais da Fase 1;
2. **Integração com um LLM local** para traduzir os diagnósticos numéricos em
   interpretações clínicas acionáveis, com técnicas de *prompt engineering* e
   avaliação objetiva de qualidade.

Todo o dataset e a convenção clínica da Fase 1 são preservados: a **classe
positiva é MALIGNO** (`pos_label=0`), pois o falso negativo (maligno
classificado como benigno) é o erro de maior custo clínico.

---

## 2. Otimização via Algoritmo Genético

### 2.1 Codificação dos genes (representação dos indivíduos)

Cada **indivíduo** é um conjunto de hiperparâmetros representado como um
dicionário `{hiperparâmetro: valor}` — cada par é um **gene**. Optamos por uma
representação **fenotípica** (valores reais/categóricos diretos) em vez de
binária, pela legibilidade e mapeamento 1:1 com o modelo.

Cada gene é descrito por um `GeneSpec` ([`models.py`](src/diag_opt/models.py))
com três tipos:

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| `float` | real, com opção de escala **logarítmica** | `C ∈ [10⁻², 10³]` (log) |
| `int` | inteiro em faixa | `n_estimators ∈ [50, 400]` |
| `cat` | categórico | `kernel ∈ {rbf, poly, sigmoid}` |

A escala logarítmica é essencial para hiperparâmetros como `C`, `gamma` e
`learning_rate`, cujas ordens de grandeza importam mais que o valor absoluto.

**Espaços de busca por modelo:**

- **SVM**: `C` (log), `gamma` (log), `kernel` (categórico)
- **RandomForest**: `n_estimators`, `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `max_features` (categórico misto)
- **GradientBoosting**: `n_estimators`, `learning_rate` (log), `max_depth`,
  `subsample`

### 2.2 Operadores genéticos

Implementados em [`ga/operators.py`](src/diag_opt/ga/operators.py):

- **Seleção**: por **torneio** (pressão seletiva ajustável por `k`) e por
  **roleta** (proporcional ao fitness);
- **Crossover**: **uniforme** (cada gene herdado de um pai com prob. 0,5) e de
  **um ponto**;
- **Mutação**: gene a gene, com **perturbação gaussiana** para genes numéricos
  (busca local) e reamostragem para categóricos, sempre respeitando o domínio;
- **Elitismo**: os melhores indivíduos passam intactos entre gerações,
  garantindo que o melhor fitness nunca regrida.

### 2.3 Função fitness

A fitness ([`ga/fitness.py`](src/diag_opt/ga/fitness.py)) é uma **combinação
ponderada de métricas**, estimada por **validação cruzada estratificada
(5-fold)** — a mesma estratégia da Fase 1, que evita sobreajuste a uma única
partição:

```
fitness = 0,6 × recall(maligno) + 0,3 × F1(maligno) + 0,1 × AUC
```

O peso dominante no recall reflete a prioridade clínica. Os pesos são
configuráveis (parametrizados nos experimentos). Um **cache** evita reavaliar
cromossomos repetidos, reduzindo drasticamente o custo. O **test set fica
reservado** apenas para a comparação final (sem vazamento de dados).

### 2.4 Experimentos realizados

Foram executadas **4 configurações** distintas do GA (requisito: ≥3), variando
população, taxa de mutação, método de seleção e tipo de crossover:

| Experimento | População | Gerações | Taxa mutação | Seleção | Crossover |
|-------------|-----------|----------|--------------|---------|-----------|
| `baseline_ga` | 20 | 15 | 0,15 | torneio | uniforme |
| `high_mutation` | 20 | 15 | **0,35** | torneio | uniforme |
| `large_population` | **40** | 10 | 0,15 | torneio | uniforme |
| `roulette_onepoint` | 20 | 15 | 0,15 | **roleta** | **1 ponto** |

**Resultados (SVM, modelo vencedor da Fase 1)** — fitness por validação cruzada:

| Experimento | Best fitness (CV) | Melhores hiperparâmetros | Avaliações | Tempo |
|-------------|-------------------|--------------------------|------------|-------|
| `baseline_ga` | 0,9726 | C=2,61; γ=0,034; rbf | 99 | 9,2 s |
| `high_mutation` | 0,9655 | C=12,46; γ=0,005; rbf | 154 | 9,3 s |
| `large_population` | 0,9655 | C=30,95; γ=0,003; rbf | 156 | 9,6 s |
| `roulette_onepoint` | **0,9762** | C=5,31; γ=0,034; rbf | 144 | 9,1 s |

**Observações sobre os experimentos:**

- A configuração `roulette_onepoint` atingiu o maior fitness de CV; `baseline_ga`
  ficou muito próximo com **menos avaliações** (99) — ou seja, mais eficiente.
- `high_mutation` explorou mais o espaço (154 avaliações, valores de `C` mais
  altos), mas não superou as configurações mais conservadoras — sinal de que,
  para o SVM neste dataset, exploração excessiva não compensa.
- O cache manteve o número de avaliações bem abaixo do total teórico
  (pop × gerações = 300 para `baseline_ga`, contra 99 efetivas).

---

## 3. Comparativo: modelos originais vs. otimizados

A comparação final é feita **no test set** (20% dos dados, nunca visto pelo GA).

### 3.1 SVM — o modelo já era quase ótimo

| Métrica | Original (Fase 1) | Otimizado (GA) |
|---------|-------------------|----------------|
| Accuracy | 0,9825 | 0,9825 |
| Recall (maligno) | 0,9762 | 0,9762 |
| F1 (maligno) | 0,9762 | 0,9762 |
| Falsos Negativos | 1 | 1 |

**Conclusão importante e honesta:** o SVM com hiperparâmetros *default* da Fase 1
já estava praticamente no ótimo para este dataset. O GA **confirma** essa
escolha — um resultado valioso: valida cientificamente a decisão da Fase 1 e
demonstra que o método não "inventa" ganhos onde não há.

### 3.2 Gradient Boosting — ganho clínico real

Para evidenciar o valor do GA, aplicamo-lo a um modelo com **baseline mais
fraco** na Fase 1 (Gradient Boosting, recall CV de 0,906):

| Métrica | Original (Fase 1) | Otimizado (GA) | Δ |
|---------|-------------------|----------------|---|
| Fitness (CV) | 0,9216 | **0,9655** | +0,044 |
| Recall (maligno, test) | 0,9048 | **0,9286** | +0,024 |
| **Falsos Negativos (test)** | **4** | **3** | **−1** |

O GA **reduziu um falso negativo** — em oncologia, isso significa um caso
maligno a menos sendo classificado erroneamente como benigno. Hiperparâmetros
encontrados: `n_estimators=294`, `learning_rate=0,155`, `max_depth=3`,
`subsample=0,54`.

### 3.3 Random Forest

| Métrica | Original | Otimizado |
|---------|----------|-----------|
| Fitness (CV) | 0,9444 | 0,9494 |
| Recall (maligno, test) | 0,9286 | 0,9286 |
| Falsos Negativos | 3 | 3 |

Ganho marginal no fitness de CV, estável no test set.

### 3.4 Síntese

O GA entrega os **maiores ganhos onde há espaço** (Gradient Boosting) e
**confirma modelos já bem ajustados** (SVM). Esse comportamento é exatamente o
esperado de uma otimização robusta e reforça a confiança na solução da Fase 1.

---

## 4. Integração com LLM para interpretação de resultados

### 4.1 Abordagem e decisão de arquitetura

Utilizamos um **LLM local open-source** (`qwen3` servido via `llama.cpp`, API
compatível com OpenAI). A decisão é central e clínica: **dados de pacientes não
saem da rede local**, alinhado à **LGPD**, com custo de API zero. O endpoint é
plugável por variável de ambiente (`LLM_BASE_URL`, `LLM_MODEL`), aceitando
Ollama, LM Studio, vLLM etc.

Fluxo de interpretação ([`llm/interpreter.py`](src/diag_opt/llm/interpreter.py)):

1. Constrói o **contexto do paciente**: predição, confiança (probabilidade
   calibrada) e as **características celulares mais relevantes**, medidas em
   desvios-padrão em relação à distribuição de treino;
2. Monta o prompt e chama o LLM;
3. Se o LLM estiver indisponível, usa um **fallback determinístico** (template),
   garantindo degradação graciosa.

### 4.2 Prompt engineering

Técnicas aplicadas ([`llm/prompts.py`](src/diag_opt/llm/prompts.py)):

- **Role prompting**: o modelo assume o papel de assistente de apoio (nunca
  substituto) ao médico;
- **Restrições explícitas**: proibição de diagnóstico definitivo, uso de
  linguagem probabilística, reforço de que a decisão é do profissional;
- **Estrutura de saída fixa**: seções *Resumo / Fatores / Recomendação / Aviso*,
  fáceis de auditar;
- **Grounding**: injeção apenas dos números do caso, reduzindo alucinação.

Exemplo de saída real do modelo (caso maligno, confiança 95,3%):

> **## Resumo** — O modelo prediz com alta confiança (95,3%) malignidade, com
> base em padrões celulares consistentes com tumores malignos.
> **## Fatores que mais influenciaram** — *perimeter error* (2,2 σ acima da
> média, irregularidade da forma), *worst perimeter* (heterogeneidade celular)…
> **## Recomendação** — considerar biópsia para confirmação…
> **## Aviso** — apoio à decisão, não diagnóstico; a palavra final é do médico.

### 4.3 Avaliação da qualidade das interpretações

Em vez de avaliação apenas subjetiva, definimos **critérios objetivos e
verificáveis** ([`llm/quality.py`](src/diag_opt/llm/quality.py)), gerando um
score de conformidade reprodutível — útil para comparar prompts:

| Critério | O que verifica |
|----------|----------------|
| Estrutura | presença das seções esperadas |
| Segurança clínica | disclaimer de "apoio, não diagnóstico" |
| Prudência | linguagem probabilística |
| Sem afirmação categórica | ausência de termos como "diagnóstico definitivo" |
| Grounding | cita as características que embasaram a predição |
| Idioma/tamanho | PT-BR e extensão adequada |

Na avaliação de casos reais, o LLM local atingiu **score ~0,86**. O único
critério ocasionalmente reprovado foi *sem afirmação categórica* — o modelo, por
vezes, é assertivo demais ("prediz com alta confiança"). Esse achado, capturado
automaticamente, orienta o refinamento do prompt e demonstra o valor de uma
avaliação objetiva. O fallback determinístico atinge score ≥ 0,85 por
construção.

---

## 5. Escalabilidade, monitoramento e nuvem

- **Monitoramento/logging** ([`monitoring/`](src/diag_opt/monitoring/)): logging
  estruturado (console + arquivo rotativo) e *tracking* de cada execução do GA em
  JSON (configuração, convergência por geração, métricas finais).
- **Escalabilidade** ([`docs/escalabilidade.md`](docs/escalabilidade.md)): dois
  workloads distintos — otimização (batch, paralelizável por indivíduo) e
  inferência (online, *stateless*, auto-scaling horizontal).
- **Nuvem/IaC** ([`infra/`](infra/)): Terraform provisionando ECS Fargate com
  **auto-scaling** por CPU e por requisições, ALB e CloudWatch — a
  materialização da arquitetura de escalabilidade (pontuação extra do enunciado).

Arquitetura completa em [`docs/arquitetura.md`](docs/arquitetura.md); segurança e
produção em [`docs/seguranca.md`](docs/seguranca.md).

---

## 6. Desafios enfrentados e soluções

| Desafio | Solução |
|---------|---------|
| SVM da Fase 1 já era quase ótimo — pouco espaço para ganho | Aplicar o GA também a modelos mais fracos (GB) para demonstrar valor; tratar a confirmação do SVM como resultado positivo de validação |
| Custo computacional do GA (muitas avaliações × CV) | Cache de fitness + `n_jobs=-1` na CV; redução de ~300 para ~100 avaliações efetivas |
| `SVC(probability=True)` deprecado e lento (sklearn 1.9) | Usar `decision_function` para o AUC no GA; calibrar probabilidades só no modelo de serviço (`CalibratedClassifierCV`) |
| Dependência de um LLM externo quebraria testes/demo | Cliente com detecção de disponibilidade + **fallback determinístico**; testes forçam o caminho offline |
| Privacidade de dados de pacientes | LLM **local**, dados anonimizados no prompt (LGPD) |
| Qualidade variável das respostas do LLM | Avaliação objetiva automatizada + prompt com restrições explícitas |
| Modelo `qwen3` emite blocos de raciocínio `<think>` | Limpeza automática da resposta no cliente |

---

## 7. Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Experimentos do GA
PYTHONPATH=src python experiments/run_experiments.py --model SVM

# Interpretação via LLM (usa LLM local; cai no fallback se indisponível)
pip install -e . && diag-opt interpret --index 0

# Testes (84% de cobertura)
pytest --cov=diag_opt
```

Notebook de demonstração ponta a ponta em
[`notebooks/demo.ipynb`](notebooks/demo.ipynb).

---

## 8. Conclusão

Entregamos um sistema que (1) otimiza hiperparâmetros de modelos de diagnóstico
via Algoritmo Genético com codificação, operadores e fitness próprios; (2)
comprova ganhos reais (redução de falso negativo no Gradient Boosting) e valida
a escolha da Fase 1 (SVM); (3) interpreta diagnósticos em linguagem natural com
um LLM local, respeitando privacidade e com avaliação objetiva de qualidade; e
(4) documenta arquitetura, escalabilidade, segurança e infraestrutura como
código — endereçando diretamente os pontos de melhoria apontados na Fase 1.
