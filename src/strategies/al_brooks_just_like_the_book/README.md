# Al Brooks (Book-Style) Strategy

Este módulo implementa uma aproximação programável das regras do Al Brooks, com foco em três conjuntos de setups frequentes e objetivos:

- Continuação em tendência via inside bars (ii/ioi) alinhadas à tendência
- H2/L2 (segunda tentativa) dentro da tendência
- Breakout + Pullback (rompimento de swing seguido de retorno à EMA20)

Nota importante: o método do Al Brooks é discricionário. Aqui traduzimos os conceitos em heurísticas objetivas e auditáveis, mantendo a filosofia do livro, mas respeitando as limitações de um sistema 100% algorítmico.

## Componentes

- `config.py`: dataclass `AlBrooksBookConfig` e helpers para salvar/carregar config ativa
- `indicators.py`: EMA20/50/200, ATR, classificação de barras (trend/doji/inside/outside) e swings simples
- `rules.py`: detecção de sinais (IB-Trend, H2/L2, BO-PB)
- `backtest.py`: motor de backtest com custos (slippage/taker fee) e trailing por ATR
- `optimize.py`: integração com Optuna via utilitário genérico do projeto
- `walk_forward.py`: integração com validação walk-forward do projeto

## Como usar

1) Otimização (BTCUSDT em 1m, por padrão):

```
python -m src.strategies.al_brooks_just_like_the_book.optimize --days 365 --trials 300
```

- Ao final, a melhor configuração será salva em `reports/active/ALBROOKS_BOOK_BTCUSDT_1m.json`.

2) Backtest usando config ativa:

```
python -m src.strategies.al_brooks_just_like_the_book.backtest --ticker BTCUSDT --interval 1m --days 365
```

3) Walk-Forward Validation:

```
python -m src.strategies.al_brooks_just_like_the_book.walk_forward --opt-window 30 --val-window 15 --step-size 15
```

## Parâmetros principais

- `ema_fast_period` (20), `ema_medium_period` (50), `ema_slow_period` (200)
- `swing_lookback` (3): pivôs simples de swing
- `bar_body_min_pct` (55%): mínimo p/ classificar trend bar (corpo/alcance)
- `near_extreme_frac` (0.25): quão “perto” do extremo o fechamento precisa estar
- `atr_period` (14), `atr_stop_multiplier` (opcional), `atr_trail_multiplier` (0.5)
- `enable_inside_trend`, `enable_h2_l2`, `enable_bo_pb`
- `bo_lookback` (20): janela p/ encontrar rompimento prévio de swing
- `max_ema_distance_atr` (1.0): PB próximo da EMA20 em múltiplos de ATR
- `use_trend_slope`, `min_ema_slope`: filtro opcional de força de tendência
- Custos: `taker_fee_pct` (0.0004), `slippage_pct` (0.0005)

## Observações

- Os “sinais” são avaliados sempre no último candle FECHADO; a confirmação do gatilho (stop entry) ocorre no candle atual.
- Stop padrão é no extremo da barra de sinal; se `atr_stop_multiplier>0`, o stop basado em ATR substitui o stop no extremo.
- Alvos em múltiplos de risco (`risk_reward_ratio`) e trailing opcional por ATR.
- O CSV de trades é salvo em `reports/live/ALBROOKS_BOOK_<TICKER>_<TF>_trades.csv`.

Se quiser, posso habilitar um modo `live.py` semelhante ao existente no `al_brooks_1m` para auditoria contínua (PNG + CSV) dessa versão book-style.

