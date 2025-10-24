# exper_corr_neg — Mixture of Experts (RL) para BTCUSDT 1h

Este experimento implementa um agente de Aprendizagem por Reforço com Mixture of Experts (MoE) para BTCUSDT (1h), seguindo o fluxo e as boas práticas descritas no `AGENTS.md`. A ideia de “correlação negativa” aqui não é entre tickers, e sim entre algoritmos/sinais: combinamos especialistas que tendem a performar bem em regimes opostos (tendência vs. reversão, direção vs. volatilidade, momentum de preço vs. reversão de volume) e deixamos um decodificador (gating) ponderar cada um dinamicamente.

## Decisões-chave
- Ativo/timeframe: BTCUSDT 1h, desde ~2017 (cache local `data/klines_cache.db`).
- Ambiente e ações: `{short, flat, long}` com capital lógico inicial de 1000, alavancagem configurável, stops móveis por ATR, penalidade de turnover e encerramento antecipado por piso de equity ou drawdown máximo. As janelas de treino podem iniciar em pontos aleatórios (`random_start`) para evitar viés de começo de série.
- MoE (PyTorch): 6 especialistas pequenos + gating com softmax (temperatura ajustável) e top‑k esparso. Regularização de balanceamento mantém o uso distribuído.
  - Trend — EMAs, Donchian, momentum
  - Mean‑Reversion — RSI, Bollinger, z‑score
  - Volatility — ATR, realized vol (range/rv)
  - Volume/Flow — OBV, Chaikin, spikes de volume
  - Squeeze/Breakout — largura de Bollinger, percentil de ATR
  - Pattern — forma de candles (corpo/pavio), padrões simples (doji, hammer, engulfing) e contagens rolling
- Treino RL: PPO adaptado ao MoE. O gating escolhe/pondera especialistas por passo; schedule de entropia reduz exploração ao longo dos episódios.
- Avaliação: Backtest, monitoramento contínuo (`metrics.csv/png`), visualização de ações (`visualize.py`) e análise do gating (`visualize_gating.py`). Walk‑Forward e Monte Carlo podem ser habilitados conforme necessidade.

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
- `env`: custos, tamanho/alavancagem da posição (fixo ou dinâmico), multiplicadores de ATR (stop/trailing), penalidade de turnover, pisos de equity/drawdown e janela/aleatoriedade de início
- `model`: camadas dos experts e do gating, número de experts, nomes didáticos, temperatura e top‑k
- `ppo`: hiperparâmetros do PPO (gamma, lambda, clip, lr, coeficientes etc.)
- `train`: episódios, passos por rollout, device, diretório de saída, schedule de entropia, espaçamento de logs/gráficos/avaliações
- `walk_forward`: agenda (dias de treino/val/step), episódios por janela, device, diretório

Edite o JSON e rode os comandos “limpos” abaixo — não há necessidade de flags.

## Comandos
- Treino contínuo
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_neg.train
  ```
  Artefatos: `src/strategies/exper_corr_neg/reports/train/`

- Visualização do backtest (preço + ações + equity)
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_neg.visualize
  ```
  Gera `src/strategies/exper_corr_neg/reports/train/visual_backtest.png` com legenda dos experts.

- Análise do gating (pesos/top‑k, drawdown/ruína)
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_neg.visualize_gating
  ```
  Artefatos: `gating_trace.csv`, `gating_heatmap.png` (com marcas de drawdown/ruína) e `gating_usage.png`.

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
- Para execuções longas, considere GPU (`device: "cuda"`) e ajuste `episodes`/`rollout_steps` paulatinamente.
- Monitorar risco: `metrics.csv/png` mostra entropia, drawdown (via ruína), equity greedy e uso dos experts. Acompanhe também `expert_usage.png` e `gating_heatmap.png`.
- Para expandir o MoE, basta acrescentar novas features e atualizar `model.num_experts`/`model.expert_names`; o gating decidirá quando usar cada especialista.

Qualquer alteração no comportamento deve ser feita editando o `config.json`. Os scripts já consomem esse arquivo e salvam os resultados dentro da pasta da estratégia para manter tudo organizado e reprodutível.
