#!/usr/bin/env bash
# Roteiro de demonstração da API de inferência (Tech Challenge Fase 2).
#
# Sobe a API (se ainda não estiver no ar), aquece o modelo e percorre os
# endpoints com um caso maligno e um benigno reais do conjunto de teste.
# Cada etapa espera ENTER, para você controlar o ritmo na apresentação.
#
#   ./scripts/demo_api.sh
#
set -euo pipefail

PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
API_PID=""

cd "$ROOT"

# Usa o interpretador do venv se existir; senão, o do ambiente ativo.
PY="${ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Formatação: jq se disponível, senão o json.tool do Python.
pretty() { if command -v jq >/dev/null 2>&1; then jq .; else "$PY" -m json.tool; fi; }

titulo() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }
pausa()  { printf '\n\033[2m[ENTER para continuar]\033[0m'; read -r _; }

limpar() {
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
    rm -rf "$TMP"
}
trap limpar EXIT

# ---------------------------------------------------------------- API no ar --
if curl -sf -m 2 "${BASE}/health" >/dev/null 2>&1; then
    echo "API já está rodando em ${BASE}"
else
    echo "Subindo a API em ${BASE} ..."
    PYTHONPATH=src "$PY" -m uvicorn diag_opt.serving.api:app \
        --host 127.0.0.1 --port "$PORT" >"${TMP}/api.log" 2>&1 &
    API_PID=$!
    for _ in $(seq 1 40); do
        curl -sf -m 2 "${BASE}/health" >/dev/null 2>&1 && break
        sleep 1
    done
    if ! curl -sf -m 2 "${BASE}/health" >/dev/null 2>&1; then
        echo "ERRO: a API não subiu. Log:" >&2
        cat "${TMP}/api.log" >&2
        exit 1
    fi
fi

# --------------------------------------------------- casos reais do dataset --
# Pega do conjunto de teste um caso comprovadamente maligno e um benigno.
PYTHONPATH=src "$PY" - "$TMP" <<'PYEOF'
import json, sys
from diag_opt.data import load_dataset, POSITIVE_LABEL

tmp = sys.argv[1]
ds = load_dataset()
maligno = ds.y_test[ds.y_test == POSITIVE_LABEL].index[0]
benigno = ds.y_test[ds.y_test != POSITIVE_LABEL].index[0]

for nome, idx in (("maligno", maligno), ("benigno", benigno)):
    payload = {"values": ds.X_test.loc[idx].to_dict()}
    with open(f"{tmp}/caso_{nome}.json", "w") as fh:
        json.dump(payload, fh)
    print(f"caso {nome}: paciente #{idx} (rótulo real confirmado no test set)")
PYEOF

# O primeiro request treina o modelo (lru_cache). Aquece fora da vista da plateia.
echo "Aquecendo o modelo ..."
curl -s -X POST "${BASE}/predict" -H 'Content-Type: application/json' \
    -d @"${TMP}/caso_maligno.json" >/dev/null

echo
echo "Pronto. Demo carregada — Swagger em ${BASE}/docs"
pausa

# --------------------------------------------------------------- 1. health --
titulo "1. Health check (usado pelo target group do ALB)"
echo "\$ curl ${BASE}/health"
curl -s "${BASE}/health" | pretty
pausa

# -------------------------------------------------------------- 2. predict --
titulo "2. Predição — caso MALIGNO"
echo "\$ curl -X POST ${BASE}/predict -d @caso_maligno.json"
curl -s -X POST "${BASE}/predict" -H 'Content-Type: application/json' \
    -d @"${TMP}/caso_maligno.json" | pretty
pausa

titulo "3. Predição — caso BENIGNO"
echo "\$ curl -X POST ${BASE}/predict -d @caso_benigno.json"
curl -s -X POST "${BASE}/predict" -H 'Content-Type: application/json' \
    -d @"${TMP}/caso_benigno.json" | pretty
pausa

# ------------------------------------------------------------ 4. interpret --
titulo "4. Interpretação clínica via LLM (pode levar alguns segundos)"
echo "\$ curl -X POST ${BASE}/interpret -d @caso_maligno.json"
curl -s -X POST "${BASE}/interpret" -H 'Content-Type: application/json' \
    -d @"${TMP}/caso_maligno.json" >"${TMP}/resp.json"

"$PY" - "${TMP}/resp.json" <<'PYEOF'
import json, sys, textwrap

with open(sys.argv[1]) as fh:
    r = json.load(fh)

print(f'predição........: {r["prediction"]}')
print(f'prob. maligno...: {r["probability_malignant"]:.4f}')
print(f'fonte do texto..: {r["source"]}   (llm = servidor local; fallback = determinístico)')
print(f'qualidade.......: {r["quality_score"]:.2f}')
print()
for par in r["interpretation"].split("\n"):
    print(textwrap.fill(par, 88) if par.strip() else "")
PYEOF
pausa

titulo "Fim da demonstração"
echo "Documentação interativa: ${BASE}/docs"
