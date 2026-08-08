"""Logging helpers: console logger + JSONL experiment tracker."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


def get_logger(name: str = "tgtp", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a console logger with a consistent format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class ExperimentLogger:
    """Append-only JSONL tracker for scalars/metrics (wandb-free, Colab-safe)."""

    def __init__(self, path: str, resume: bool = False) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not resume:
            open(path, "w", encoding="utf-8").close()  # truncate
        self.logger = get_logger("tgtp.experiment")

    def log(self, **kwargs: Any) -> None:
        record = {"ts": time.time()}
        record.update(kwargs)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self.logger.info(json.dumps({k: v for k, v in kwargs.items()}, default=str))

    def log_metrics(self, step: int, metrics: dict[str, float], tag: str = "train") -> None:
        self.log(step=step, tag=tag, **metrics)

    def load(self) -> list:
        rows = []
        if not os.path.exists(self.path):
            return rows
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def format_duration(seconds: float) -> str:
    """Human-readable duration."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
