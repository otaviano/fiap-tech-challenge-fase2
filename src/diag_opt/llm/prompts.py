"""Engenharia de prompts para interpretação clínica dos diagnósticos.

As técnicas de prompt engineering aplicadas aqui:

* **Papel (role prompting)**: o modelo assume o papel de assistente que apoia —
  nunca substitui — o médico;
* **Restrições explícitas**: proibição de diagnóstico definitivo, linguagem
  probabilística, reforço de que a decisão é do profissional;
* **Estrutura de saída fixa**: seções previsíveis, fáceis de auditar;
* **Grounding nos dados**: o prompt injeta apenas os números do caso (predição,
  confiança, features mais relevantes), reduzindo alucinação;
* **Few-shot leve**: o formato-alvo é demonstrado no próprio pedido.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "Você é um assistente de apoio à decisão clínica em oncologia mamária. "
    "Seu papel é traduzir a saída de um modelo de machine learning em uma "
    "explicação clara para o médico. Regras invioláveis:\n"
    "1. NUNCA forneça diagnóstico definitivo — o modelo é uma ferramenta de apoio.\n"
    "2. Use linguagem probabilística e prudente ('sugere', 'indica', 'compatível com').\n"
    "3. Sempre reforce que a decisão final é do profissional de saúde.\n"
    "4. Baseie-se estritamente nos dados fornecidos; não invente valores.\n"
    "5. Escreva em português do Brasil, tom técnico-acessível, sem jargão excessivo."
)

_OUTPUT_FORMAT = (
    "Estruture a resposta EXATAMENTE nas seções:\n"
    "## Resumo\n(1-2 frases com a predição e o nível de confiança)\n"
    "## Fatores que mais influenciaram\n(lista das características citadas e o que "
    "cada uma sugere clinicamente)\n"
    "## Recomendação de conduta\n(próximos passos prudentes de investigação)\n"
    "## Aviso\n(frase reforçando que é apoio à decisão, não diagnóstico)"
)


def _format_features(features: list[dict[str, Any]]) -> str:
    linhas = []
    for f in features:
        direcao = "acima" if f["direction"] == "maligno" else "abaixo"
        linhas.append(
            f"- {f['name']}: valor {f['value']:.2f} "
            f"({abs(f['zscore']):.1f} desvios-padrão {direcao} da média, "
            f"associado a tecido {'maligno' if f['direction']=='maligno' else 'benigno'})"
        )
    return "\n".join(linhas)


def build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """Monta as mensagens (system + user) a partir do contexto do paciente."""
    predicao = "MALIGNO" if context["predicted_label"] == 0 else "BENIGNO"
    confianca = context["probability_malignant"] * 100

    user = (
        f"Caso clínico (dados anonimizados). Saída do modelo de diagnóstico:\n\n"
        f"- Predição: {predicao}\n"
        f"- Confiança de malignidade: {confianca:.1f}%\n"
        f"- Características celulares mais relevantes para esta predição:\n"
        f"{_format_features(context['top_features'])}\n\n"
        f"Gere a interpretação para o médico responsável.\n\n{_OUTPUT_FORMAT}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
