Segue uma TODO list enxuta e priorizada para organizar o experimento. (Estado atualizado)

Fundação

- [x] Criar estrutura base em src/strategies/experimento: data/, indicators/, signals/, filters/, risk/, engine/, storage/, scripts/.
- [x] Definir arquivo config_active.json (fonte única de config, sem flags).
- [ ] Adicionar artifacts/ para relatórios e gráficos.
- [x] Confirmar remoção do .docx (feito manualmente) e manter README.md como referência.
 - [x] Forçar consumo do cache (data/klines_cache.db) com auto-update no início dos scripts.
Dados / MTF

 - [x] Implementar loader MTF (data/loader.py) para 5m/15m/30m (cache + binance).
- [x] Calcular features por TF sem lookahead (implementado direto em scripts/backtest.py com EMA/ATR mínimos).
- [x] Alinhar ao 30m via merge_asof(direction='backward') (data/align.py).
- [x] Validar com dataset sintético um caso de no-lookahead (data/synth.py, smoke test OK).
Indicadores (plugáveis)

- [ ] Interface base (indicators/base.py) com apply(df), role.
- [x] Implementar EMA, ATR mínimos (indicators/common.py).
- [ ] Registrar catálogo simples (indicators/registry.py) e RSI.
Geradores de Sinal

- [ ] Interface base (signals/base.py) com generate(df).
- [x] Implementar signals/ema_cross.py (variantes A/B/C configuráveis no futuro; hoje A long-only).
- [ ] Integração de scores (ponderação) e prioridade de sinais.
Filtros

- [x] filters/trend_mtf.py (EMA 15m 50>200).
- [x] filters/atr_threshold.py (ATR mínimo 30m).
- [x] filters/volume.py (percentil mínimo).
- [ ] Orquestrador de filtros (filters/apply.py).
Risco / Execução

- [x] risk/sizing.py (fixed_fraction).
- [x] risk/stops.py (fixo ATR, trailing ATR).
- [x] risk/costs.py (fee por lado, slippage em ticks).
- [x] Aplicar na simulação em pontos de entrada/saída (engine/backtest.py).
Motor de Backtest

- [x] engine/backtest.py: loop por barra 30m; sinais → filtros → risco; PnL.
- [ ] Suporte a side bidirecional; uma posição por vez (atual long-only).
- [ ] Logging de eventos por barra (detalhes adicionais) para auditoria.
Otimização

- [ ] scripts/optimize.py: carrega config_active.json, roda Optuna internamente.
- [ ] Objetivo: PF penalizado por nº trades + Sharpe; mínimo de trades.
- [ ] Atualiza config_active.json com melhores params (opcional: snapshot em artifacts/).
Walk-Forward

- [ ] scripts/walk_forward.py: janelas (60/20/20 sugeridas), rolling e acúmulo OOS.
- [ ] Relatório final consolidado (PF, Sharpe, drawdown, nº trades).
Armazenamento / Auditoria

- [x] storage/db.py: SQLite experimento.sqlite.
- [x] Esquema: runs, bars, signals, trades, fills, params, metrics.
- [x] Persistir cada execução, com config e timestamps.
Interfaces / Relatórios

- [ ] scripts/report.py: export CSV (signals/trades/metrics) e gerar gráficos.
- [ ] Opcional: scripts/app.py (Flask) para listar runs e detalhes.
Configuração (JSON)

- [ ] config/schema.json (rótulos/validação leve).
- [ ] Exemplo atualizado em config/example.json; ativo em config_active.json.
- [x] Respeitar: nenhum flag — apenas poetry run python -m src.strategies.experimento.scripts.<modulo>.
Validação / Qualidade

- [ ] Testes unitários críticos de alinhamento no-lookahead e custos.
- [ ] Sanidade do PnL com custos vs sem custos.
- [ ] Perf: cache/localização de merges, reduzir cópias de DF.
Próximos passos imediatos

- [x] Criar esqueleto de pastas/arquivos e config_active.json.
- [x] Implementar pipeline mínimo: dados sintéticos + EMA cross + backtest simples + persistência em SQLite.
- [x] Rodar um backtest de fumaça via poetry run python -m src.strategies.experimento.scripts.backtest.
 - [x] Adicionar Monte Carlo (scripts/monte_carlo.py) e Selftest (scripts/selftest.py) com dados tendenciosos.
