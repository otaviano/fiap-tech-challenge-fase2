"""Tracking de execuções do GA (monitoramento de desempenho).

Registra, para cada execução/experimento, a configuração, o histórico de
convergência por geração e as métricas finais, persistindo tudo em JSON para
auditoria e para as análises do relatório. Também expõe um callback pronto para
o ``on_generation`` do :class:`GeneticOptimizer`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diag_opt.ga.engine import GAResult, GenerationStats


@dataclass
class RunRecord:
    experiment: str
    started_at: str
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RunTracker:
    """Coleta métricas de uma execução do GA e persiste em disco."""

    def __init__(
        self,
        experiment: str,
        logger: logging.Logger | None = None,
        out_dir: Path | str = "results",
    ) -> None:
        self.experiment = experiment
        self.logger = logger or logging.getLogger("diag_opt")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._start = time.perf_counter()
        self.record = RunRecord(
            experiment=experiment,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def on_generation(self, stats: GenerationStats) -> None:
        """Callback para o GA: loga a evolução de cada geração."""
        self.logger.info(
            "[%s] geração %02d | best=%.4f | mean=%.4f | recall=%.4f",
            self.experiment,
            stats.generation,
            stats.best_fitness,
            stats.mean_fitness,
            stats.best_metrics.get("recall", float("nan")),
        )

    def finish(self, result: GAResult, extra: dict[str, Any] | None = None) -> Path:
        """Fecha o registro, loga o resumo e grava o JSON da execução."""
        self.record.finished_at = datetime.now(timezone.utc).isoformat()
        self.record.result = result.to_dict()
        if extra:
            self.record.extra.update(extra)

        self.logger.info(
            "[%s] concluído | best_fitness=%.4f | evals=%d | %.1fs",
            self.experiment,
            result.best_fitness,
            result.n_evaluations,
            result.elapsed_seconds,
        )

        out_path = self.out_dir / f"run_{self.experiment}.json"
        out_path.write_text(json.dumps(asdict(self.record), indent=2, ensure_ascii=False))
        self.logger.debug("Registro salvo em %s", out_path)
        return out_path
