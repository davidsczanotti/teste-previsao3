Estratégia RL — MA(7) x MA(40) x MA(120)
==============================

Visão Geral
- Ambiente `MaCrossoverEnv` fornece ao agente contexto de tendência baseado no cruzamento das médias móveis simples de 7, 40 e 120 períodos (configuráveis).
- Observação inclui 7 candles normalizados + métricas das MAs (diferença, slopes, relação com o preço) e flags de posição.
- Execução ocorre na abertura da barra seguinte (`--exec_next_open` por padrão), evitando lookahead.
- Custos modelados: fee, slippage, custo fixo por ação, penalização de troca rápida, penalidade opcional por ociosidade.
- A heurística `ma_signal` gera rótulos (e gate opcional) com base apenas nas MAs; o agente aprende timing/gestão.

Backtest (MA crossover)
```bash
poetry run python -m src.strategies.ma_crossover_rl.backtest \
  --ticker BTCUSDT --interval 15m --days 365 \
  --lot_size 0.001 --fee_rate 0.001 --min_hold_bars 2 --cooldown_bars 1 \
  --ma_short_window 7 --ma_mid_window 40 --ma_long_window 120 --ma_type ema
```

Otimização (Optuna)
```bash
poetry run python -m src.strategies.ma_crossover_rl.optimize \
  --ticker BTCUSDT --interval 15m --days 365 --trials 120
```
Gera `reports/ma_crossover_optuna/study_<...>.json` com a melhor combinação de MAs/hold/cooldown.

Treino RL (exemplo long-only, assertivo)
```bash
poetry run python -m src.strategies.ma_crossover_rl.train \
  --ticker BTCUSDT --interval 15m --days 3650 \
  --episodes 300 --hidden 64 --lr 0.003 --episode_len 8192 --long_only \
  --min_hold_bars 3 --reopen_cooldown_bars 1 \
  --action_cost_open 0.02 --action_cost_close 0.02 \
  --epsilon_start 0.3 --epsilon_end 0.02 --bc_weight 0.1 \
  --idle_penalty 0.005 --idle_grace 30 --idle_ramp 0.0 \
  --exec_next_open --switch_penalty 0.02 --switch_window_bars 5 \
  --ma_short_window 7 --ma_mid_window 40 --ma_long_window 120 \
  --ma_type ema \
  --gate_on_heuristic \
  --reward_atr_norm --atr_period 14
```

Treino PPO (NumPy, sem dependência externa)
```bash
poetry run python -m src.strategies.ma_crossover_rl.train_ppo \
  --ticker BTCUSDT --interval 15m --days 3650 \
  --episodes 200 --hidden 64 --lr 0.0003 --episode_len 4096 --long_only \
  --ma_short_window 7 --ma_mid_window 40 --ma_long_window 120 \
  --ma_type ema \
  --min_hold_bars 3 --reopen_cooldown_bars 1 \
  --fee_rate 0.0005 --action_cost_open 0.0 --action_cost_close 0.0 \
  --exec_next_open --switch_penalty 0.02 --switch_window_bars 5 \
  --gate_on_heuristic --reward_atr_norm --atr_period 14
```
Salva em `reports/agents/ma_crossover_rl/<TICKER>_<INTERVAL>.npz` e grava métricas em `metrics_ppo_*.jsonl`.
- Modelo salvo em `reports/agents/ma_crossover_rl/<TICKER>_<INTERVAL>.npz`.
- Métricas por episódio (reward, trades, hold médio, histograma de ações) em `reports/agents/ma_crossover_rl/metrics_<TICKER>_<INTERVAL>_<TIMESTAMP>.jsonl`.

Flags relevantes
- `--ma_*_window`: define as janelas das MAs (curta, intermediária, longa).
- `--gate_on_heuristic`: somente permite abrir posição quando o padrão MA_curta > MA_média > MA_longa (ou inverso) estiver alinhado.
- `--switch_penalty` + `--switch_window_bars`: custo extra para reabrir/virar lado rapidamente (anti-churn).
- `--reward_atr_norm` + `--atr_period`: normaliza a parte intrabar do PnL pelo ATR.
- `--idle_penalty`, `--idle_grace`, `--idle_ramp`: empurra contra inatividade prolongada (use moderado).
- `--min_hold_bars`, `--reopen_cooldown_bars`: evitam zigue-zague.
- `--model <arquivo>`: continua o treinamento a partir de um checkpoint.

Boas práticas
1. **Treinos longos + random start** para cobrir regimes distintos.
2. **Validar out-of-sample** (greedy) e comparar PF/MDD/trades antes de usar.
3. **Monitorar métricas JSONL** para detectar hiperatividade (trades excessivos) ou colapso em Hold.
4. **Ajustar custos gradualmente** após estabilizar (reduzir `action_cost_*`, liberar short removendo `--long_only`).
5. **Guardar checkpoints “best-so-far”** com base na avaliação out-of-sample.

Diagnóstico rápido
- 0 trades no greedy ⇒ aumente exploração (`epsilon_*`) ou desative gate temporariamente.
- Muitos trades ⇒ aumente `min_hold_bars`, `switch_penalty` e/ou mantenha gate ativo.
- Reward muito negativo com poucos trades ⇒ revise custos, allow_short ou inclua outros filtros de tendência.
