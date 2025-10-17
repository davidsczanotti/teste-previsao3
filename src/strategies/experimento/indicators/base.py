from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class Indicator:
    name: str
    params: Dict[str, Any]
    role: str | None = None  # 'gate' | 'score' | 'stop'

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

