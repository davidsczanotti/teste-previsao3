#!/usr/bin/env python3
"""
CLI limpa para executar backtest EMA-only baseado em config.json
"""

import json
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest import run_backtest

def main():
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"Erro: {config_path} não encontrado.")
        sys.exit(1)

    run_backtest(str(config_path))

if __name__ == "__main__":
    main()
