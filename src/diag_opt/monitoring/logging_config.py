"""Configuração centralizada de logging.

Logs vão para o console (nível INFO) e para arquivo rotativo em ``results/logs``
(nível DEBUG), permitindo o *tracking* de desempenho exigido no requisito de
escalabilidade/monitoramento.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_LOG_DIR = Path("results/logs")


def setup_logging(
    name: str = "diag_opt",
    log_dir: Path | str = _DEFAULT_LOG_DIR,
    level: int = logging.INFO,
) -> logging.Logger:
    """Cria (ou reaproveita) um logger com saída para console e arquivo."""
    logger = logging.getLogger(name)
    if logger.handlers:  # já configurado
        return logger

    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(console)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / f"{name}.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger
