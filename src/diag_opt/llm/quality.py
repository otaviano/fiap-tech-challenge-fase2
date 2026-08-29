"""Avaliação automática da qualidade das interpretações geradas pela LLM.

O enunciado pede "avaliar a qualidade das interpretações geradas". Em vez de uma
avaliação apenas subjetiva, definimos um conjunto de critérios objetivos e
verificáveis, gerando um score de conformidade. Isso torna a avaliação
reprodutível e comparável entre prompts diferentes (útil para prompt engineering).

Critérios avaliados:
* **Estrutura**: presença das seções esperadas;
* **Segurança clínica**: contém disclaimer de que é apoio, não diagnóstico;
* **Prudência**: usa linguagem probabilística, sem afirmações definitivas;
* **Grounding**: cita as características que embasaram a predição;
* **Idioma/tamanho**: resposta em português e com extensão adequada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_EXPECTED_SECTIONS = ("resumo", "influenciaram", "conduta", "aviso")
_SAFETY_TERMS = ("apoio", "não substitui", "decisão", "médico", "profissional")
_PRUDENT_TERMS = ("sugere", "indica", "compatível", "probab", "%")
# Afirmações categóricas indesejadas em contexto de apoio ao diagnóstico.
_FORBIDDEN = ("diagnóstico definitivo", "com certeza", "definitivamente", "garantido")
# Negadores que INVERTEM o sentido de um termo proibido na mesma sentença:
# "não constitui diagnóstico definitivo" é prudência, não afirmação categórica.
_NEGATIONS = ("não", "nao", "nunca", "jamais")
_SENTENCE_ENDS = ".;!?\n"


@dataclass
class QualityReport:
    score: float  # 0..1
    checks: dict[str, bool]
    details: dict[str, Any]

    def passed(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold


def _has_categorical_claim(low: str) -> bool:
    """Detecta afirmação categórica de fato, respeitando negações.

    Uma busca por substring simples reprova o texto correto: o disclaimer
    recomendado — "não constitui diagnóstico definitivo" — contém o termo
    proibido, mas exprime exatamente a prudência que queremos. Aqui, cada
    ocorrência só conta como violação se **não** houver negador entre o início
    da sentença e o termo.
    """
    for term in _FORBIDDEN:
        for match in re.finditer(re.escape(term), low):
            start = max(low.rfind(end, 0, match.start()) for end in _SENTENCE_ENDS)
            sentence_head = low[start + 1 : match.start()]
            if not any(neg in sentence_head for neg in _NEGATIONS):
                return True
    return False


def evaluate_interpretation(text: str, context: dict[str, Any]) -> QualityReport:
    """Aplica os critérios objetivos e retorna um score de conformidade."""
    low = text.lower()

    has_structure = sum(sec in low for sec in _EXPECTED_SECTIONS) >= 3
    has_safety = any(term in low for term in _SAFETY_TERMS)
    is_prudent = any(term in low for term in _PRUDENT_TERMS)
    no_forbidden = not _has_categorical_claim(low)

    cited = sum(1 for f in context.get("top_features", []) if f["name"].lower() in low)
    grounded = cited >= max(1, len(context.get("top_features", [])) // 2)

    # Heurística simples de idioma PT-BR + tamanho mínimo informativo.
    pt_markers = len(re.findall(r"\b(de|que|com|para|não|é|do|da)\b", low))
    is_portuguese = pt_markers >= 3
    reasonable_length = 200 <= len(text) <= 4000

    checks = {
        "estrutura": has_structure,
        "seguranca_clinica": has_safety,
        "prudencia": is_prudent,
        "sem_afirmacao_categorica": no_forbidden,
        "grounding_nos_dados": grounded,
        "idioma_ptbr": is_portuguese,
        "tamanho_adequado": reasonable_length,
    }
    score = sum(checks.values()) / len(checks)

    return QualityReport(
        score=score,
        checks=checks,
        details={"features_citadas": cited, "tamanho_chars": len(text)},
    )
