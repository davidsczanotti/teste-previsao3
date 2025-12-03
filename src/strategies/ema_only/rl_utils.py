from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any


class MetricsLogger:
    """Logger simples em CSV para métricas de treino."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = self.path.exists()

    def log(self, row: Dict[str, Any]) -> None:
        keys = list(row.keys())
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)
