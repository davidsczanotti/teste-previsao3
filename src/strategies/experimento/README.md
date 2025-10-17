# Experimento MTF (sem lookahead) — Especificação

## Objetivo
Projetar um experimento de estratégia com múltiplos indicadores e múltiplos timeframes (MTF) para BTC, com alinhamento sem lookahead, componentes plugáveis, configuração via JSON, auditoria completa (trades, sinais, custos), e métrica de otimização robusta (profit factor + penalização e Sharpe).

## Timeframes e MTF (sem lookahead)
- Base/timeframe principal: `30m` para execução e avaliação.
- Contexto de suporte: `5m` e `15m` para filtros e confirmação.
- Alinhamento sem lookahead: todos os dados de 5m e 15m são alinhados à barra base de 30m usando apenas informações disponíveis até o fechamento da barra de 30m (ou até sua abertura, conforme o papel do sinal).
- Construir features nos TF auxiliares (5m/15m) primeiro, sem usar barras futuras.
- Para cada barra de 30m (timestamp de fechamento `t_close`), sincronizar com a última barra concluída de 5m/15m com `close_time <= t_close` via `merge_asof(direction='backward')`.
- Para agregações (ex.: tendência 15m em janelas de 2 barras dentro de 30m), usar apenas as barras 15m que caberiam até `t_close`; nunca incluir barras parcialmente abertas ou futuras.

Exemplo de alinhamento (pseudocódigo Python):

```python
# df_30m, df_15m, df_5m com colunas 'close_time' ordenadas
# Features já calculadas em 5m/15m SEM lookahead.
df_30m = df_30m.sort_values('close_time')
df_15m = df_15m.sort_values('close_time')
df_5m  = df_5m.sort_values('close_time')

# Alinhar 15m -> 30m
df = pd.merge_asof(
    df_30m, df_15m,
    left_on='close_time', right_on='close_time',
    direction='backward', suffixes=('', '_15m')
)

# Alinhar 5m -> (30m já com 15m)
df = pd.merge_asof(
    df, df_5m,
    left_on='close_time', right_on='close_time',
    direction='backward', suffixes=('', '_5m')
)

# Observação: se preferir usar a abertura da barra base como cutoff,
# faça o merge usando um 'cutoff_time' = open_time_base ou aplique máscara.
```

## Papel dos Indicadores
- Filtro (gate): habilita/bloqueia a geração de sinais. Ex.: tendência MTF, ATR mínimo, volume mínimo.
- Confirmação (score): atribui pontuações aos possíveis sinais. Ex.: confluência de EMAs/RSI/ADX.
- Ajuste de saída/stop: modifica stop, alvo e trailing. Ex.: stop ATR, trailing por canal/ATR.

## Geradores de Sinal (base + variantes)
- Estratégia base: cruzamento de EMAs no 30m.
- Variante A: EMA(9/21) + filtro de tendência 15m.
- Variante B: EMA(20/50) + score por RSI/ADX em 5m/15m.
- Variante C: EMA(12/26) + ATR gate + trailing ATR.

## Filtros (exemplos)
- Tendência MTF: somente compra se EMA(15m, 50) > EMA(15m, 200).
- ATR limiar: ATR(30m) >= p_perc do range mediano dos últimos N dias.
- Volume mínimo: volume(30m) >= percentual do seu percentil 40/50.

## Risco/Execução
- Sizing: fração de capital fixa ou risco fixo por trade (em ATR).
- Stops: fixo em múltiplos de ATR; trailing por ATR/canal de volatilidade.
- Custos: fee por lado + slippage em ticks/pips; aplicar em cada execução (entrada/saída).

## Métrica de Otimização
Objetivo composto: priorizar Profit Factor com penalização por baixo número de trades e complementar com Sharpe.

```python
# Exemplo de objetivo (n_trades >= n_min, caso contrário descarta)
score = 0.0
if n_trades >= n_min:
    pf = total_profit / total_loss if total_loss > 0 else float('inf')
    trade_saturation = min(1.0, n_trades / n_target)  # penalização suave
    score = 0.7 * pf * trade_saturation + 0.3 * sharpe
else:
    score = -1e6  # rejeita soluções com pouca amostra
```

## Walk-Forward (janelas e passo)
Sugestões para BTC em 30m (ajuste conforme dados e custos):
- Rápido (exploração): 30 dias otimização / 10 dias validação, passo 10 dias.
- Equilíbrio: 60 dias otimização / 20 dias validação, passo 20 dias.
- Robusto: 90 dias otimização / 30 dias validação, passo 30 dias.

Recalibrar parâmetros a cada passo; acumular métricas out-of-sample para relatório final.

## Arquitetura por Componentes (plugáveis)
Manter tudo dentro de `src/strategies/experimento` (sem reaproveitar de outros módulos). Componentes:
- Dados/MTF: carrega múltiplos TF (5m/15m/30m), calcula features e alinha ao 30m sem lookahead.
- Indicadores: cada um adiciona colunas e, opcionalmente, fornece gate/score/ajuste de stops.
- Geradores de Sinal: EMA cross base e variantes (A/B/C) plugáveis.
- Filtros: tendência MTF, limiar de ATR, volume, horário, etc.
- Risco/Execução: sizing, stops (fixo/ATR), trailing, custos/slippage.
- Motor de Backtest: iteração por barra (30m), aplica sinais+filtros+risco, calcula PnL.

## Configuração em JSON (exemplo)
```json
{
  "symbol": "BTCUSDT",
  "base_timeframe": "30m",
  "context_timeframes": ["15m", "5m"],
  "data": {
    "source": "binance",
    "days": 180,
    "cache": true
  },
  "indicators": [
    { "name": "ema", "tf": "30m", "params": {"fast": 9, "slow": 21}, "role": "score" },
    { "name": "ema", "tf": "15m", "params": {"fast": 50, "slow": 200}, "role": "gate" },
    { "name": "atr", "tf": "30m", "params": {"length": 14}, "role": "stop" }
  ],
  "signal_generators": [
    { "name": "ema_cross", "variant": "A", "params": {"fast": 9, "slow": 21} }
  ],
  "filters": {
    "trend_tf": { "tf": "15m", "ema_fast": 50, "ema_slow": 200 },
    "atr_min_pct": 0.2,
    "volume_percentile": 0.4
  },
  "risk": {
    "sizing": { "type": "fixed_fraction", "fraction": 0.02 },
    "stop":   { "type": "atr", "mult": 2.0 },
    "trailing":{ "type": "atr", "mult": 1.5 },
    "costs":   { "fee_bp": 2.0, "slippage_ticks": 1 }
  },
  "optimization": {
    "metric": "pf_penalized_sharpe",
    "min_trades": 20,
    "target_trades": 60,
    "w_pf": 0.7,
    "w_sharpe": 0.3
  },
  "walk_forward": {
    "opt_days": 60,
    "val_days": 20,
    "step_days": 20
  },
  "storage": {
    "results_db": "src/strategies/experimento/experimento.sqlite",
    "artifacts_dir": "src/strategies/experimento/artifacts"
  }
}
```

## Interfaces e Auditoria
- Foco na simplicidade: persistir todos os resultados (trades, sinais por barra, parâmetros, custos) em SQLite dentro de `src/strategies/experimento/experimento.sqlite` para auditoria e reprodutibilidade.
- Opcional: interface minimalista Flask para listar backtests, abrir detalhes de trades e exportar CSV.
- CLI: `poetry run python -m src.strategies.experimento.<modulo>` para executar backtests/otimizações.
- DB: tabelas `runs`, `bars`, `signals`, `trades`, `fills`, `params`, `metrics`.
- Artefatos: salvar gráficos e JSONs de configuração por run em `artifacts/`.

## Observações finais
- Não reutilizar módulos externos do projeto para a estratégia; manter tudo em `experimento/`.
- Configuração exclusivamente via JSON; não usar flags em comandos.
- Evitar lookahead rigorosamente nos alinhamentos MTF e nos indicadores.

