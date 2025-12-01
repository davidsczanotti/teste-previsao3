EMA-only — baseline de cruzamento de EMA
=======================================

Visão geral
-----------

Esta estratégia implementa um backtest simples baseado em **uma única EMA**:

- Modo mean reversion (default): entra comprado quando o preço fecha **abaixo**
  da EMA e sai quando fecha **acima**.
- Modo cruzamento de EMAs (`signal_mode="ema_cross"`): usa duas EMAs
  (`ema_period` + `slow_ema_period`) e fica comprado quando a EMA rápida cruza
  **para cima** da lenta e zera posição quando cruza **para baixo**.
- `use_cross=true` no modo price_reversion: exige que o preço recupere a EMA
  (cruzamento de baixo para cima) depois de um pullback para abrir a posição,
  e sai no cruzamento descendente.

É uma baseline didática para comparar com experimentos mais complexos
(`exper_corr_pos`, `exper_hr_bg_rl`, etc.) sob os mesmos custos e dados.

Arquivos principais
-------------------

- `backtest.py`  
  Implementa:
  - `EmaOnlyParams`: parâmetros da estratégia (período da EMA, modo de sinal,
    pullback, filtro de tendência, lot_size, fee, uso de cruzamento/reclaim).
  - `backtest_ema_only(df, params, initial_capital)`: função pura que recebe um
    DataFrame OHLCV e retorna lista de trades, PnL total e estatísticas
    agregadas. Suporta `signal_mode="price_reversion"` (legado) e
    `signal_mode="ema_cross"` (tendência via duas EMAs).

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
- `strategy.signal_mode`: `"price_reversion"` (preço vs EMA) ou `"ema_cross"`
  (cruzamento de EMAs).
- `strategy.ema_period`: período da EMA rápida (ex.: `34` ou `89`).
- `strategy.slow_ema_period`: período da EMA lenta (obrigatório no modo
  `ema_cross`).
- `strategy.pullback_pct`: distância mínima abaixo da EMA rápida para abrir
  posição no modo `price_reversion` (ex.: `0.002` = 0,2%).
- `strategy.use_cross`: no modo `price_reversion`, exige que o preço recupere a
  EMA (de baixo para cima) após o pullback para entrar; a saída ocorre no
  cruzamento descendente.
- `strategy.use_trend_filter` + `strategy.trend_filter_period`: ativa filtro de
  tendência para `price_reversion` (exige EMA rápida e preço acima da EMA de
  filtro).
- `strategy.lot_size`: tamanho fixo da posição em BTC.
- `strategy.fee_pct`: taxa de corretagem por lado (decimal).
- `backtest.initial_capital`: capital inicial em USDT.
- `backtest.outdir`: pasta onde o JSON de resultados será salvo.

Exemplo (já presente no repo):

```json
{
  "data": {
    "symbol": "BTCUSDT",
    "timeframe": "4h",
    "days": 3650
  },
  "strategy": {
    "ema_period": 55,
    "slow_ema_period": 200,
    "signal_mode": "ema_cross",
    "pullback_pct": 0.0,
    "use_trend_filter": false,
    "trend_filter_period": null,
    "use_cross": false,
    "lot_size": 0.001,
    "fee_pct": 0.0004
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
- Rodar `backtest_ema_only` com os parâmetros de `strategy` (modo
  `ema_cross` no exemplo).
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
