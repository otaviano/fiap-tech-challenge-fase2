# Documento de Entrega — Fase 2

Este documento reúne os entregáveis do Tech Challenge da Fase 2. O projeto
escolhido foi o **Projeto 1 — Otimização de Modelos de Diagnóstico**: otimização
dos hiperparâmetros dos modelos da Fase 1 por **algoritmo genético**, somada à
**integração com um LLM local** para interpretar os diagnósticos em linguagem
clínica.

## Links do projeto

| Recurso | URL |
|---|---|
| **Repositório GitHub** | https://github.com/otaviano/fiap-tech-challenge-fase2 |
| **Vídeo de demonstração** | https://youtu.be/z1GaoESr23E |
| **Relatório técnico** | https://github.com/otaviano/fiap-tech-challenge-fase2/blob/main/RELATORIO_TECH_CHALLENGE_FASE2.md |
| **Notebook de demonstração** | https://github.com/otaviano/fiap-tech-challenge-fase2/blob/main/notebooks/demo.ipynb |
| **Arquitetura** | https://github.com/otaviano/fiap-tech-challenge-fase2/blob/main/docs/arquitetura.md |
| **Escalabilidade e monitoramento** | https://github.com/otaviano/fiap-tech-challenge-fase2/blob/main/docs/escalabilidade.md |
| **Segurança e produção** | https://github.com/otaviano/fiap-tech-challenge-fase2/blob/main/docs/seguranca.md |
| **Infraestrutura como código** | https://github.com/otaviano/fiap-tech-challenge-fase2/tree/main/infra |
| **Projeto da Fase 1 (base)** | https://github.com/otaviano/fiap-tech-challenge-fase1 |

## Entregáveis do repositório

| Item exigido | Onde encontrar |
|---|---|
| Código-fonte completo | `src/diag_opt/` — pacote Python modular instalável |
| Documentação da API | `src/diag_opt/serving/api.py` + Swagger em `/docs` |
| Scripts e notebooks de demonstração | `notebooks/demo.ipynb`, `experiments/`, `scripts/demo_api.sh` |
| Arquivos de configuração para implantação | `Dockerfile`, `infra/` (Terraform) |
| Relatório técnico | `RELATORIO_TECH_CHALLENGE_FASE2.md` (reproduzido a seguir) |
| Vídeo de demonstração (≤ 15 min) | https://youtu.be/z1GaoESr23E |

## Mapeamento das entregas técnicas exigidas

### 1. Otimização via Algoritmos Genéticos

| Requisito do desafio | Atendido em |
|---|---|
| Codificação adequada dos hiperparâmetros (representação de genes) | `models.py` (`GeneSpec`) e `ga/encoding.py` — genes `float` (linear e log), `int` e categórico; relatório §2.1 |
| Operadores de seleção, cruzamento e mutação | `ga/operators.py` — torneio, roleta e ranqueamento; crossover uniforme e de um ponto; mutação gaussiana e categórica, com elitismo; relatório §2.2 |
| Função fitness baseada em métricas de desempenho | `ga/fitness.py` — 0,6 · recall + 0,3 · F1 + 0,1 · ROC-AUC sobre a classe maligna, em validação cruzada estratificada de 5 folds; relatório §2.3 |
| Comparação entre modelos otimizados e originais | `evaluation.py`, notebook células 3 e 4; relatório §3 |
| Ao menos 3 experimentos com configurações diferentes | `experiments.py` — **5 experimentos**; evidência em `results/experiments_summary.json`; relatório §2.4 |

### 2. Escalabilidade automática, monitoramento e logging

| Requisito do desafio | Atendido em |
|---|---|
| Monitoramento e logging para tracking de desempenho | `monitoring/` — logs estruturados por geração e tracking em JSON; evidência de execução completa em `results/logs/exemplo_execucao.log`; relatório §5 |
| Documentação de arquitetura e decisões | `docs/arquitetura.md` (com diagrama), `docs/escalabilidade.md`, `docs/seguranca.md` |
| Implementação em nuvem (opcional, pontuação extra) | `infra/` — Terraform para ECS Fargate com auto-scaling por CPU e por requisições, ALB e CloudWatch |

### 3. Integração com LLMs para interpretação de resultados

| Requisito do desafio | Atendido em |
|---|---|
| Explicações em linguagem natural dos diagnósticos | `llm/interpreter.py` — Resumo, Fatores, Recomendação e Aviso; relatório §4.1 |
| Transformar dados numéricos em insights acionáveis | `llm/interpreter.py` — as features são traduzidas em desvios-padrão em relação à média da base, com as `top_k` mais influentes do caso |
| Preparar a base para a integração da Fase 3 | LLM servido por endpoint OpenAI-compatível, plugável e configurável por ambiente |
| Prompt engineering | `llm/prompts.py` — papel de apoio (nunca substituto), restrições explícitas, estrutura de saída fixa e grounding nos dados do caso; relatório §4.2 |
| Avaliação da qualidade das interpretações | `llm/quality.py` — critérios objetivos de estrutura, segurança clínica, prudência e grounding; relatório §4.3 |

### 4. Código e organização

| Requisito do desafio | Atendido em |
|---|---|
| Projeto Python estruturado com ambiente virtual | `pyproject.toml`, `requirements.txt`, `venv`; pacote instalável com CLI `diag-opt` |
| Documentação detalhada, incluindo diagramas de arquitetura | `README.md` e `docs/arquitetura.md` (diagrama Mermaid) |
| Testes automatizados | `tests/` — 33 testes, 86% de cobertura |
| IaC para provisionamento (nuvem) | `infra/` — Terraform |

## Conteúdo do vídeo (≤ 15 minutos)

| Item exigido | Onde aparece |
|---|---|
| Demonstração do sistema em execução | Execução dos experimentos do GA no terminal e do notebook |
| Explicação dos diferentes componentes da solução | Estrutura do pacote, `docs/arquitetura.md` e o bloco dedicado ao GA por dentro |
| Resultados da otimização via algoritmos genéticos | Curva de convergência, tabela dos 5 experimentos e comparação com os modelos originais |
| Demonstração da integração com LLMs | `diag-opt interpret`, interpretação gerada e avaliação de qualidade |

## Como reproduzir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

diag-opt experiments --model SVM        # 5 experimentos do GA
diag-opt optimize --model GradientBoosting
diag-opt interpret --index 0            # interpretação via LLM local (com fallback)

uvicorn diag_opt.serving.api:app --port 8000    # API REST
pytest --cov=diag_opt                            # 33 testes
```

O algoritmo genético usa semente fixa: a mesma configuração reproduz o mesmo
cromossomo e as mesmas métricas entre máquinas e versões de Python.
