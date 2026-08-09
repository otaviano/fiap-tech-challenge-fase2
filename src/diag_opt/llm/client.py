"""Cliente para LLM local via API compatível com OpenAI.

Aponta, por padrão, para o ``llama-server`` (llama.cpp) rodando localmente em
``http://localhost:8080/v1`` com o modelo ``qwen3``. Tudo é configurável por
variáveis de ambiente, de modo que a banca avaliadora possa apontar para o seu
próprio servidor (Ollama, LM Studio, vLLM etc.) sem alterar código:

    LLM_BASE_URL   (default: http://localhost:8080/v1)
    LLM_MODEL      (default: qwen3)
    LLM_TIMEOUT    (default: 120 segundos)

Motivação clínica: manter o LLM **local** evita enviar dados de pacientes a
serviços externos, alinhado à LGPD e a boas práticas de privacidade em saúde.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

# Remove o "raciocínio" interno que modelos do tipo qwen3 podem emitir.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
    model: str = os.getenv("LLM_MODEL", "qwen3")
    timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))
    temperature: float = 0.2
    max_tokens: int = 800


class LLMUnavailableError(RuntimeError):
    """Erro levantado quando o servidor LLM não está acessível."""


class LLMClient:
    """Wrapper mínimo sobre o endpoint ``/chat/completions``."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def is_available(self) -> bool:
        """Verifica rapidamente se o servidor responde em ``/models``."""
        try:
            resp = requests.get(f"{self.config.base_url}/models", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def _clean(text: str) -> str:
        return _THINK_RE.sub("", text).strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Envia mensagens e retorna o conteúdo textual da resposta."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        try:
            resp = requests.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # rede/servidor fora
            raise LLMUnavailableError(str(exc)) from exc

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return self._clean(content)
