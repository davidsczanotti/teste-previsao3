from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class SignalGenerator:
    name: str
    params: Dict[str, Any]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

