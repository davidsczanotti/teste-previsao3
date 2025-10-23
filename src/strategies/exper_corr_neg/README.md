# exper_corr_neg — Mixture of Experts (RL) para BTCUSDT 1h

Este experimento implementa um agente de Aprendizagem por Reforço com Mixture of Experts (MoE) para BTCUSDT (1h), seguindo o fluxo e as boas práticas descritas no `AGENTS.md`. A ideia de “correlação negativa” aqui não é entre tickers, e sim entre algoritmos/sinais: combinamos especialistas que tendem a performar bem em regimes opostos (tendência vs. reversão, direção vs. volatilidade, momentum de preço vs. reversão de volume) e deixamos um decodificador (gating) ponderar cada um dinamicamente.

## Decisões-chave
- Ativo/timeframe: BTCUSDT 1h, desde ~2017 (cache local `data/klines_cache.db`).
- Ambiente e ações: `{short, flat, long}` com tamanho fixo de 0.1 BTC; capital lógico de 1000; custos padrão (fee 0.1%, slippage 0.01%); stop móvel por ATR (trailing).
- MoE (PyTorch): 5 especialistas pequenos + gating com top‑2 (sparsidade), softmax com temperatura e regularização de balanceamento de carga.
  - Trend Expert — EMAs, Donchian, momentum
  - Mean‑Reversion Expert — RSI, Bollinger, z‑score
  - Volatility Expert — ATR, realized vol (range/rv)
  - Volume/Flow Expert — OBV, Chaikin, “volume spike”
  - Squeeze/Breakout Expert — BB width, ATR percentil
- Treino RL: PPO adaptado ao MoE (laço custom). O gating escolhe os experts (top‑2) a cada passo; a política mistura suas saídas para decidir a ação.
- Avaliação: Backtest, Monte Carlo e Walk‑Forward (WF) com relatórios reprodutíveis.

## Estrutura do projeto
```
src/strategies/exper_corr_neg/
  __init__.py
  config.json                 # parâmetros do experimento (sem flags)
  features.py                 # cálculo de features para experts e contexto
  env.py                      # ambiente de trading (Gym‑like, com stub se gym não existir)
  models.py                   # experts, gating network e política MoE (PyTorch)
  trainer.py                  # PPO trainer adaptado ao MoE
  data.py                     # carregamento do BTCUSDT 1h e alinhamento de features
  train.py                    # treino contínuo (config‑driven)
  walk_forward.py             # walk‑forward (treino por janela + validação)
  reports/                    # artefatos (checkpoints, resumos, etc.)
```

## Dados e modo offline
- Leitura: sempre do cache local (`data/klines_cache.db`) via `utils.data_loader`.
- Atualização (opcional, com rede):
  ```bash
  poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"
  ```
- Execução offline (recomendado p/ reprodutibilidade): use a variável `BINANCE_OFFLINE=1` e um cache Numba local para acelerar funções do pandas_ta:
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache ...
  ```

## Configuração (sem flags)
Todos os parâmetros ficam em `src/strategies/exper_corr_neg/config.json`:
- `env`: custos, tamanho de posição, multiplicadores de ATR (stop/trailing)
- `model`: camadas dos experts e do gating, número de experts, temperatura, top‑k
- `ppo`: hiperparâmetros do PPO (gamma, lambda, clip, lr, coeficientes etc.)
- `train`: episódios, passos por rollout, device, diretório de saída
- `walk_forward`: agenda (dias de treino/val/step), episódios por janela, device, diretório

Edite o JSON e rode os comandos “limpos” abaixo — não há necessidade de flags.

## Comandos
- Treino contínuo
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_neg.train
  ```
  Artefatos: `src/strategies/exper_corr_neg/reports/train/`

- Walk‑Forward (treina por janela e valida OOS)
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_neg.walk_forward
  ```
  Artefatos: `src/strategies/exper_corr_neg/reports/walk_forward/`

## Avaliação e Relatórios (AGENTS.md)
- Backtest com custos e stop móvel; export de trades (quando aplicável).
- Monte Carlo: perturbação por blocos na sequência de trades para distribuição de P&L/PF/MDD.
- Walk‑Forward: adaptativo (reotimização a cada janela) e/ou fixo (parâmetros ancorados) — comparar consistência.
- Relatório Markdown consolidando: MDD/Calmar/Ulcer, Sharpe/Sortino rolantes, tail risk, turnover, stress de custos/lag/perturbação de features, splits por regime (baixa/alta vol) e gráficos (equity+MDD/UI, Sharpe rolante, histogramas, heatmaps por hora, etc.).

### Checklist de robustez (resumo)
- Risco: MDD absoluto/relativo, Ulcer Index, Calmar/Mar, Time Under Water
- Qualidade: Sharpe/Sortino (rolantes), skew/kurtosis, PF, hit rate e payoff ratio
- Execução: turnover, cost/alpha, slippage efetivo, duração média e trades/dia
- Estabilidade: correlações (benchmark e entre sinais), Information Ratio por regime, estabilidade de Sharpe em janelas
- Calibração (se probabilístico): Brier/ACE, reliability plot, Q‑Q
- Testes: custos ±2–3×, lag (1 bar), partial fills, price perturbation, feature noise/dropout, grid pequena de hypers, bootstrap/permutation, Reality Check simples
- Gate prático: Sharpe OOS ≥ 1, MDD ≤ 15–20%, UI ≤ 5–8, PF ≥ 1.3, ≥ 60% das janelas OOS positivas, sensibilidade a custos < 30%

## Observações
- O ambiente funciona sem a dependência `gym` (há um stub leve), mas se quiser os wrappers e compatibilidade plena, instale `gym`.
- Para execuções longas, considere GPU (`device: "cuda"` no config) e aumente `episodes`/`rollout_steps` paulatinamente.
- Sinais adicionais (ex.: modelos de direção/volatilidade como LightGBM/GARCH) podem ser integrados como novos experts.

Qualquer alteração no comportamento deve ser feita editando o `config.json`. Os scripts já consomem esse arquivo e salvam os resultados dentro da pasta da estratégia para manter tudo organizado e reprodutível.

