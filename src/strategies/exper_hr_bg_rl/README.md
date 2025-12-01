exper_hr_bg_rl — Hybrid Range‑Volatility RL (BTCUSDT 1h)
========================================================

Visão geral
-----------

Este experimento implementa um agente de Aprendizado por Reforço em BTCUSDT 1h,
usando sinais de **range/volatilidade** (high‑low, ATR, realized vol, etc.) como
parte do estado. A ideia é permitir que o agente aprenda:

- quando **não operar** (ficar flat);
- quando **entrar** comprado/vendido em regimes de volatilidade específicos;
- por quanto tempo **segurar** a posição, levando em conta custos reais.

O experimento é isolado nesta pasta, seguindo o mesmo padrão de `exper_corr_pos`:

- `config.json` — parâmetros de dados, ambiente, modelo e PPO;
- `data.py` — carregamento de OHLCV 1h a partir de `data/klines_cache.db` e
  construção do dataset de features;
- `features.py` — features de range/vol/realized vol;
- `env.py` — ambiente RL discreto (flat/long/short) com custos, PnL mark‑to‑market;
- `models.py` — política+crítico (MLP) para PPO;
- `trainer.py` — laço de treino PPO (coleta rollout, GAE, atualização);
- `train.py` — CLI de treino (`poetry run python -m src.strategies.exper_hr_bg_rl.train`);
- `backtest.py` — backtest greedy do modelo treinado.

Dados e timeframe
-----------------

- Símbolo: `BTCUSDT`
- Timeframe: `1h`
- Janela: até `data.lookback_days` dias (ver `config.json`)
- Fonte: cache local em `data/klines_cache.db` populado via:

  ```bash
  poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"
  ```

Pipeline do experimento
-----------------------

1. **Carregar dados**  
   `data.py` usa `utils.data_loader` para ler BTCUSDT 1h do cache local,
   garantindo índice datetime.

2. **Gerar features**  
   `features.py` constrói um vetor de features por candle, com:
   - range high‑low normalizado;
   - ATR e ATR relativo;
   - realized vol em múltiplas janelas;
   - Parkinson volatility;
   - z‑score de volume;
   - kurtosis/skew de retornos.

3. **Ambiente RL (env.py)**  
   - Ações discretas: `{0: flat, 1: long, 2: short}` (ou `{0: flat, 1: long}` se `allow_short=false`).
   - Cada passo aplica:
     - custos de abertura/fechamento (`fee_pct`, `slippage_pct`);
     - PnL mark‑to‑market entre `t` e `t+1`;
   - Episódios janelados (`window_bars`) com início aleatório (`random_start=true`).

4. **Modelo e PPO (models.py, trainer.py)**  
   - Política MLP sobre o vetor de features (obs) com head discreto para ações.
   - Crítico MLP separado para valor.
   - PPO com GAE, clipping, entropia e clipping de gradiente,
     parametrizado via `config.json::ppo`.

5. **Treino (train.py)**  
   - Carrega `config.json`, constrói dataset, ambiente, modelo e trainer.
   - Executa múltiplos episódios de treino com rollouts de tamanho fixo.
   - Loga métricas em `reports/train/metrics.csv` e salva:
     - `policy_best.pt` (melhor equity greedy);
     - `policy_final.pt` (último modelo).

6. **Backtest (backtest.py)**  
   - Carrega o melhor modelo (`policy_best.pt` ou `policy_final.pt`).
   - Roda uma trajetória greedy sobre uma janela determinística de dados 1h.
   - Calcula métricas agregadas via `utils.metrics.calculate_metrics`:
     PF, win_rate, PnL, etc.
   - Salva JSON em `reports/backtest/exper_hr_bg_rl_BTCUSDT_1h.json`.

Comandos principais
-------------------

1) Popular cache (Passo 0 obrigatório):

```bash
poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"
```

2) Treinar agente PPO:

```bash
BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
  poetry run python -m src.strategies.exper_hr_bg_rl.train
```

3) Backtest greedy:

```bash
BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_hr_bg_rl.backtest
```

Estado atual
------------

Este README descreve a arquitetura alvo e o workflow do experimento.  
Os módulos `env.py`, `models.py`, `trainer.py`, `train.py` e `backtest.py`
devem ser implementados seguindo este design, usando o mesmo padrão de
configuração de `exper_corr_pos` (blocos `data`, `env`, `model`, `ppo`, `train`).

