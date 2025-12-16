# teste-previsao3

Ferramentas de análise/backtest para ações (B3) e cripto usando dados do Yahoo Finance (`yfinance`) e uma implementação fiel do sinal **Trend Surfer v4.1** (TradingView/Pine).

## Requisitos
- Python 3.12+
- Poetry

## Instalação
- `poetry install`

## Comandos principais
- Scanner B3 (lista fixa de tickers): `poetry run python run_scanner.py`
- Scanner cripto (Yahoo Finance): `poetry run python run_scanner_crypto.py`
- Analisar 1 ticker: `poetry run python main.py WEGE3.SA`
- Modo interativo: `poetry run python main.py`

## Parâmetros (Trend Surfer v4.1)
- Os parâmetros estão no `config` dentro de `main.py` e `run_scanner*.py`.
- O cálculo de indicadores/sinais está em `src/core/indicators.py` e `src/core/signals.py`.
- O backtest com semântica “Pine-like” está em `src/core/backtest.py`.

## Organização do repo
- `docs/`: notas e comparações (ex.: `docs/scanner/`)
- `examples/`: scripts de exemplo por ticker
- `data/`: caches/DBs locais (não versionados)
