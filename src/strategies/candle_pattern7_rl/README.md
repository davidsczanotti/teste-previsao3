Estratégia RL com Padrão de 7 Velas
==================================

Visão Geral
- Ambiente `Candle7Env` observa sempre as últimas 7 velas fechadas e entrega sinais ao agente.
- Execução ocorre na abertura da próxima barra (`--exec_next_open` por padrão), evitando lookahead.
- Custos modelados: fees, slippage, custo de ação, penalização de ociosidade, penalização de troca rápida.
- Heurística do padrão fornece rótulo auxiliar (`heuristic_action`) para behavior cloning e, opcionalmente, gate de abertura (`--gate_on_heuristic`).
- Recompensa = PnL realizado nas saídas + marcação a mercado intrabar (normalizável por ATR) − custos/penalidades.

Instalação
- Python 3.12 + Poetry (já utilizados no projeto).
- Instale deps na raiz: `poetry install`.

Como Executar (Backtest clássico)
- `poetry run python -m src.strategies.candle_pattern7_rl.backtest --ticker BTCUSDT --interval 15m --days 90 --lot_size 0.001 --fee_rate 0.001`

Como Treinar (RL)
- Exemplo completo (ênfase em assertividade, long-only):
  ```bash
  poetry run python -m src.strategies.candle_pattern7_rl.train \
    --ticker BTCUSDT --interval 15m --days 3650 \
    --episodes 300 --hidden 64 --lr 0.003 --episode_len 8192 --long_only \
    --min_hold_bars 3 --reopen_cooldown_bars 1 \
    --action_cost_open 0.02 --action_cost_close 0.02 \
    --epsilon_start 0.3 --epsilon_end 0.02 --bc_weight 0.1 \
    --idle_penalty 0.005 --idle_grace 30 --idle_ramp 0.0 \
    --exec_next_open --switch_penalty 0.02 --switch_window_bars 5 \
    --gate_on_heuristic \
    --reward_atr_norm --atr_period 14
  ```
- Saída do modelo: `reports/agents/candle_pattern7_rl/<TICKER>_<INTERVAL>.npz`
- Métricas de episódio em `reports/agents/candle_pattern7_rl/metrics_<TICKER>_<INTERVAL>_<TIMESTAMP>.jsonl`.

Flags Importantes
- `--exec_next_open` (padrão): executa ordens na próxima abertura.
- `--switch_penalty` e `--switch_window_bars`: penaliza reabrir rápido (anti-churn).
- `--reward_atr_norm` + `--atr_period`: normaliza reward intrabar pelo ATR.
- `--idle_penalty`, `--idle_grace`, `--idle_ramp`: empurra contra ociosidade (use moderado).
- `--gate_on_heuristic`: só permite abrir posição quando o padrão 7 velas concorda (Hold/Close liberados).
- `--bc_weight`: perda auxiliar de imitação (decai no tempo) para destravar aprendizado inicial.
- `--min_hold_bars`, `--reopen_cooldown_bars`: evitam zigue-zague de barras.

Continuar Treinamento (Warm-start)
- `poetry run python -m src.strategies.candle_pattern7_rl.train --ticker BTCUSDT --interval 15m --days 3650 --episodes 150 --episode_len 8192 --long_only --gate_on_heuristic --model reports/agents/candle_pattern7_rl/BTCUSDT_15m.npz`

Boas Práticas
1. **Treinos longos + random start**: cobre regimes distintos do histórico.
2. **Validação temporal**: avalie greedy em janelas out-of-sample; compare PF/MDD/trades.
3. **Monitoramento**: use o JSONL para acompanhar trades por episódio, hold médio, invalids.
4. **Ajuste progressivo**: comece `long_only` com custos altos, depois reduza custos e libere short se o agente estiver estável.
5. **Salvar best-so-far**: mantenha checkpoints em separado quando a avaliação OOS melhorar.

Diagnóstico rápido
- 0 trades no greedy ⇒ política ainda colapsada em Hold; revisite exploração/BC/idle penalty.
- Milhares de trades ⇒ reduza `epsilon_*`, aumente `min_hold_*`, `switch_penalty`, gate.
- Reward muito negativo com poucos trades ⇒ revise custos vs. lot_size, ou permita short.

