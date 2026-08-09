"""Testes da camada LLM.

Não dependem de um servidor LLM real: forçamos o caminho de *fallback* injetando
um cliente que sempre falha, garantindo que a demonstração e o CI rodem offline.
"""

from diag_opt.evaluation import fit_serving_model
from diag_opt.llm.client import LLMClient, LLMUnavailableError
from diag_opt.llm.interpreter import build_patient_context, interpret_patient
from diag_opt.llm.prompts import build_messages
from diag_opt.llm.quality import evaluate_interpretation


class _BrokenClient(LLMClient):
    def chat(self, *a, **k):
        raise LLMUnavailableError("offline")


def test_build_patient_context(dataset):
    pipe = fit_serving_model("SVM", {"C": 10, "gamma": 0.05, "kernel": "rbf"}, dataset)
    ctx = build_patient_context(pipe, dataset, index=0, top_k=4)
    assert 0.0 <= ctx["probability_malignant"] <= 1.0
    assert len(ctx["top_features"]) == 4
    assert ctx["predicted_label"] in (0, 1)


def test_prompt_contem_secoes_e_dados(dataset):
    pipe = fit_serving_model("SVM", {"C": 10, "gamma": 0.05, "kernel": "rbf"}, dataset)
    ctx = build_patient_context(pipe, dataset, index=0)
    messages = build_messages(ctx)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "## Resumo" in user
    assert "Confiança de malignidade" in user


def test_interpret_usa_fallback_quando_llm_offline(dataset):
    pipe = fit_serving_model("SVM", {"C": 10, "gamma": 0.05, "kernel": "rbf"}, dataset)
    interp = interpret_patient(pipe, dataset, index=0, client=_BrokenClient())
    assert interp.source == "fallback"
    assert "## Resumo" in interp.text
    assert "## Aviso" in interp.text


def test_quality_report_do_fallback_e_alto(dataset):
    pipe = fit_serving_model("SVM", {"C": 10, "gamma": 0.05, "kernel": "rbf"}, dataset)
    interp = interpret_patient(pipe, dataset, index=0, client=_BrokenClient())
    report = evaluate_interpretation(interp.text, interp.context)
    # o template determinístico foi desenhado para satisfazer os critérios
    assert report.passed(threshold=0.8)
    assert report.checks["seguranca_clinica"]
    assert report.checks["sem_afirmacao_categorica"]
