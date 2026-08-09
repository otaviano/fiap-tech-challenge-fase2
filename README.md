# Tech Challenge — Fase 2: Otimização de Modelos de Diagnóstico

> **FIAP Postech — IA para Devs**
> Projeto 1 — Otimização via Algoritmos Genéticos + Interpretação com LLM

Continuação do [Tech Challenge Fase 1](https://github.com/otaviano/fiap-tech-challenge-fase1)
(diagnóstico de câncer de mama, nota 85/90). Nesta fase, otimizamos os
**hiperparâmetros** dos modelos de diagnóstico via **Algoritmo Genético** e
integramos um **LLM local** para interpretar os diagnósticos em linguagem
natural para os profissionais de saúde.

---

## O que foi construído

| Requisito do enunciado | Onde está |
|------------------------|-----------|
| GA para otimização de hiperparâmetros | [`src/diag_opt/ga/`](src/diag_opt/ga/) |
| Codificação de genes | [`models.py`](src/diag_opt/models.py) (`GeneSpec`) + [`ga/encoding.py`](src/diag_opt/ga/encoding.py) |
| Seleção, crossover e mutação | [`ga/operators.py`](src/diag_opt/ga/operators.py) |
| Fitness por métricas (recall/F1/AUC) | [`ga/fitness.py`](src/diag_opt/ga/fitness.py) |
| Comparação otimizado vs original | [`evaluation.py`](src/diag_opt/evaluation.py) |
| ≥3 experimentos com configs diferentes | [`experiments.py`](src/diag_opt/experiments.py) (4 configs) |
| Escalabilidade + monitoramento/logging | [`monitoring/`](src/diag_opt/monitoring/) + [`docs/escalabilidade.md`](docs/escalabilidade.md) |
| Integração com LLM | [`src/diag_opt/llm/`](src/diag_opt/llm/) |
| Prompt engineering | [`llm/prompts.py`](src/diag_opt/llm/prompts.py) |
| Avaliação da qualidade das interpretações | [`llm/quality.py`](src/diag_opt/llm/quality.py) |
| Documentação de arquitetura | [`docs/arquitetura.md`](docs/arquitetura.md) |
| IaC (nuvem, pontuação extra) | [`infra/`](infra/) (Terraform) |
| Testes automatizados | [`tests/`](tests/) (84% de cobertura) |

---

## Arquitetura (resumo)

```
Dataset → [ Algoritmo Genético: encoding → fitness(CV) → seleção → crossover → mutação → elitismo ]
        → Modelo otimizado → Avaliação (baseline vs otimizado)
        → Interpretação clínica via LLM local (com fallback) → Avaliação de qualidade
        ↳ Monitoramento/logging em todas as etapas
```

Detalhes completos em [`docs/arquitetura.md`](docs/arquitetura.md).

---

## Como executar

### 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate            # Linux/Mac  (.venv\Scripts\activate no Windows)
pip install -r requirements.txt
```

### 2. Rodar os experimentos do Algoritmo Genético

```bash
PYTHONPATH=src python experiments/run_experiments.py --model SVM
# resultados em results/experiments_summary.json
```

Ou via CLI:

```bash
pip install -e .
diag-opt experiments --model SVM        # 4 experimentos
diag-opt optimize --model GradientBoosting
```

### 3. Interpretação de um diagnóstico via LLM

O sistema usa, por padrão, um **LLM local** compatível com a API da OpenAI
(ex.: `llama-server`/llama.cpp, Ollama, LM Studio). Configure via variáveis de
ambiente (ou deixe o padrão):

```bash
export LLM_BASE_URL=http://localhost:8080/v1   # padrão
export LLM_MODEL=qwen3                          # padrão

diag-opt interpret --index 0
```

> Se o LLM não estiver disponível, o sistema usa automaticamente um **fallback
> determinístico** — a demonstração e os testes rodam sempre.

### 4. Notebook de demonstração

```bash
jupyter notebook notebooks/demo.ipynb
```

### 5. Testes

```bash
pytest --cov=diag_opt
```

---

## Por que LLM local?

Contexto **hospitalar**: manter o LLM local evita enviar dados de pacientes a
serviços externos, alinhado à **LGPD**. Custo de API zero e endpoint plugável
para qualquer servidor OpenAI-compat. Detalhes em [`docs/seguranca.md`](docs/seguranca.md).

---

## Documentação

- [Relatório técnico](RELATORIO_TECH_CHALLENGE_FASE2.md)
- [Arquitetura](docs/arquitetura.md)
- [Escalabilidade e monitoramento](docs/escalabilidade.md)
- [Segurança e produção](docs/seguranca.md)
- [Infraestrutura como código](infra/README.md)

---

*FIAP Postech — IA para Devs | Tech Challenge Fase 2 | 2026*
