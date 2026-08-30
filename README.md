# Tech Challenge — Fase 2: Otimização de Modelos de Diagnóstico

> **FIAP Postech — IA para Devs**
> Projeto 1 — Otimização via Algoritmos Genéticos + Interpretação com LLM

**[▶ Vídeo de demonstração (≤ 15 min)](https://youtu.be/z1GaoESr23E)** ·
**[Documento de entrega (PDF)](ENTREGA_TECH_CHALLENGE_FASE2.pdf)** ·
**[Relatório técnico](RELATORIO_TECH_CHALLENGE_FASE2.md)**

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
| ≥3 experimentos com configs diferentes | [`experiments.py`](src/diag_opt/experiments.py) (5 configs) |
| Escalabilidade + monitoramento/logging | [`monitoring/`](src/diag_opt/monitoring/) + [`docs/escalabilidade.md`](docs/escalabilidade.md) |
| Integração com LLM | [`src/diag_opt/llm/`](src/diag_opt/llm/) |
| Prompt engineering | [`llm/prompts.py`](src/diag_opt/llm/prompts.py) |
| Avaliação da qualidade das interpretações | [`llm/quality.py`](src/diag_opt/llm/quality.py) |
| Documentação de arquitetura | [`docs/arquitetura.md`](docs/arquitetura.md) |
| IaC (nuvem, pontuação extra) | [`infra/`](infra/) (Terraform) |
| Testes automatizados | [`tests/`](tests/) (33 testes, 86% de cobertura) |
| Logging da evolução do GA | [`monitoring/`](src/diag_opt/monitoring/) — evidência em [`results/logs/exemplo_execucao.log`](results/logs/exemplo_execucao.log) |

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

Requer **Python ≥ 3.10** (validado até o 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate            # Linux/Mac  (.venv\Scripts\activate no Windows)
pip install -r requirements.txt
pip install -e .                     # habilita o comando `diag-opt`
```

Para a API REST e os testes, instale também:

```bash
pip install -r requirements-serving.txt   # fastapi + uvicorn
pip install pytest pytest-cov jupyter ipykernel
```

### 2. Rodar os experimentos do Algoritmo Genético

```bash
diag-opt experiments --model SVM        # 5 experimentos
diag-opt optimize --model GradientBoosting
diag-opt optimize --model SVM --population 8 --generations 3   # corrida curta, p/ demo
```

Sem instalar o pacote, use o script direto:

```bash
PYTHONPATH=src python experiments/run_experiments.py --model SVM
# resultados em results/experiments_summary.json

PYTHONPATH=src python experiments/run_gb_rf.py
# comparativo baseline vs. otimizado dos modelos de árvore
# resultados em results/gb_rf_comparison.json
```

> O GA usa seed fixa: a mesma configuração reproduz o mesmo cromossomo e as mesmas
> métricas entre máquinas e versões de Python.

### 3. Interpretação de um diagnóstico via LLM

O sistema usa, por padrão, um **LLM local** compatível com a API da OpenAI
(ex.: `llama-server`/llama.cpp, Ollama, LM Studio). O modelo usado no
desenvolvimento e na demonstração foi o **Qwen3 4B Instruct**
(`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, quantização Q4_K_M) servido por
llama.cpp — leve o bastante para rodar na máquina local, o que mantém os dados
de paciente fora da rede. Configure via variáveis de ambiente (ou deixe o
padrão):

```bash
export LLM_BASE_URL=http://localhost:8080/v1   # padrão
export LLM_MODEL=qwen3                          # padrão

diag-opt interpret --index 0
```

> Se o LLM não estiver disponível, o sistema usa automaticamente um **fallback
> determinístico** — a demonstração e os testes rodam sempre.

### 4. API REST de inferência

```bash
uvicorn diag_opt.serving.api:app --port 8000
```

Endpoints: `GET /health`, `POST /predict` e `POST /interpret`.
Documentação interativa em <http://localhost:8000/docs>.

O `/health` identifica qual configuração está servindo — útil para auditar em
produção qual cromossomo do GA está no ar:

```bash
curl -s localhost:8000/health
# {"status":"ok",
#  "model":{"estimator":"SVM","params":{"C":5.31,"gamma":0.0336,"kernel":"rbf"},
#           "origem":"GA — experimento roulette_onepoint (maior fitness de CV)"},
#  "llm":{"base_url":"http://localhost:8080/v1","model":"qwen3"}}
```

O corpo das requisições é `{"values": {<nome_da_feature>: valor, ...}}` com as 30
features do dataset. Para montar um caso real a partir do próprio dataset
(**com o venv ativado** — senão o `import` falha e gera um arquivo vazio):

```bash
python -c "
from diag_opt.data import load_dataset; import json
print(json.dumps({'values': load_dataset().X.iloc[0].to_dict()}))" > /tmp/caso.json

# confira que o arquivo não saiu vazio antes de chamar a API
test -s /tmp/caso.json && echo "caso gerado" || echo "FALHOU — o venv está ativado?"

curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @/tmp/caso.json
# {"prediction":"maligno","probability_malignant":0.9127308585272944}

curl -s -X POST localhost:8000/interpret -H 'Content-Type: application/json' -d @/tmp/caso.json
```

> Se o `curl` responder `option -d: error encountered when reading a file`, o
> arquivo não existe no caminho informado — gere-o antes, no mesmo caminho que
> o `-d @` aponta.

O `/interpret` aceita ainda `top_k` (padrão `4`), que controla quantas features
entram na explicação — ex.: `{"values": {...}, "top_k": 3}`.

Os hiperparâmetros do modelo servido vêm do GA e são configuráveis por ambiente
(`MODEL_C`, `MODEL_GAMMA`, `MODEL_KERNEL`).

Via Docker:

```bash
docker build -t diag-opt .
docker run -p 8000:8000 diag-opt
```

### 5. Notebook de demonstração

```bash
jupyter notebook notebooks/demo.ipynb
```

### 6. Testes

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

- [Documento de entrega](docs/entrega-fase2.md) — links, checklist e mapeamento dos requisitos
- [Relatório técnico](RELATORIO_TECH_CHALLENGE_FASE2.md)
- [Arquitetura](docs/arquitetura.md)
- [Escalabilidade e monitoramento](docs/escalabilidade.md)
- [Segurança e produção](docs/seguranca.md)
- [Infraestrutura como código](infra/README.md)

---

*FIAP Postech — IA para Devs | Tech Challenge Fase 2 | 2026*
