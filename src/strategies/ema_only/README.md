EMA-only — baseline de cruzamento de EMA
=======================================

Visão geral
-----------

Esta estratégia implementa um backtest simples baseado em **uma única EMA**:

- Modo mean reversion (default): entra comprado quando o preço fecha **abaixo**
  da EMA e sai quando fecha **acima**.
- Modo cruzamento (`use_cross=true`): entra quando ocorre um **cruzamento
  descendente** (preço cruzando de cima para baixo a EMA) e sai em um
  **cruzamento ascendente**.

É uma baseline didática para comparar com experimentos mais complexos
(`exper_corr_pos`, `exper_hr_bg_rl`, etc.) sob os mesmos custos e dados.

Arquivos principais
-------------------

- `backtest.py`  
  Implementa:
  - `EmaOnlyParams`: parâmetros da estratégia (período da EMA, lot_size, fee, uso de cruzamento).
  - `backtest_ema_only(df, params, initial_capital)`: função pura que recebe
    um DataFrame OHLCV e retorna lista de trades, PnL total e estatísticas
    agregadas.

- `pipeline.py` (modo legacy via CLI com flags)  
  CLI original que aceita parâmetros via linha de comando (`--ema-period`,
  `--use-cross`, etc.) e executa backtests ad‑hoc.

- `config.json` (novo, config‑driven)  
  Arquivo de configuração padrão para rodar backtests determinísticos e
  reprodutíveis sem passar flags.

- `run.py` (novo)  
  CLI limpa que lê `config.json`, carrega dados de `data/klines_cache.db` e
  executa o backtest, salvando um JSON de resultados.

Dados e cache
-------------

Os dados de mercado são lidos do cache SQLite `data/klines_cache.db` via
`utils.data_loader.load_data`. Antes de rodar o experimento, garanta que o
cache está populado para o símbolo/timeframe desejado.

Exemplo (BTCUSDT 1h, desde 2017‑01‑01):

```bash
poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"
```

Configuração (`config.json`)
----------------------------

O arquivo `config.json` define os parâmetros do experimento:

- `data.symbol`: símbolo (ex.: `"BTCUSDT"`).
- `data.timeframe`: timeframe (ex.: `"1h"`).
- `data.days`: quantos dias de histórico carregar (ex.: `3650`).
- `strategy.ema_period`: período da EMA (ex.: `8`).
- `strategy.use_cross`: se `true`, usa eventos de cruzamento; se `false`,
  usa lógica simples acima/abaixo.
- `strategy.lot_size`: tamanho fixo da posição em BTC.
- `strategy.fee_pct`: taxa de corretagem por lado (decimal).
- `backtest.initial_capital`: capital inicial em USDT.
- `backtest.outdir`: pasta onde o JSON de resultados será salvo.

Exemplo (já presente no repo):

```json
{
  "data": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "days": 3650
  },
  "strategy": {
    "ema_period": 8,
    "use_cross": true,
    "lot_size": 0.001,
    "fee_pct": 0.001
  },
  "backtest": {
    "initial_capital": 1000.0,
    "outdir": "src/strategies/ema_only/reports/backtest"
  }
}
```

Como rodar o experimento (modo config‑driven)
--------------------------------------------

1) Popule o cache (se ainda não fez)  

```bash
poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"
```

2) Execute o backtest EMA‑only

```bash
BINANCE_OFFLINE=1 poetry run python -m src.strategies.ema_only.run
```

Isso vai:

- Ler `config.json`.
- Carregar BTCUSDT 1h do cache local (`use_cache_only=True`).
- Rodar `backtest_ema_only` com os parâmetros de `strategy`.
- Salvar um JSON em `src/strategies/ema_only/reports/backtest/ema_only_BTCUSDT_1h.json`
  contendo:
  - parâmetros usados,
  - estatísticas agregadas (PnL, retorno %, win rate, MDD, etc.),
  - trades individuais.

Modo legacy (pipeline com flags)
--------------------------------

Ainda é possível usar o pipeline original para testar rapidamente variações:

```bash
BINANCE_OFFLINE=1 poetry run python -m src.strategies.ema_only.pipeline \
  --symbol BTCUSDT \
  --interval 1h \
  --days 3650 \
  --ema-period 8 \
  --use-cross \
  --lot-size 0.001 \
  --fee-rate 0.001 \
  --cache-only
```

Este modo continua funcional, mas para pesquisa reprodutível recomenda‑se o
fluxo baseado em `config.json` e `run.py`.

