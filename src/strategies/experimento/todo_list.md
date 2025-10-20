Segue uma TODO list enxuta e priorizada para organizar o experimento. (Estado atualizado)

Fundação

- [x] Criar estrutura base em src/strategies/experimento: data/, indicators/, signals/, filters/, risk/, engine/, storage/, scripts/.
- [x] Definir arquivo config_active.json (fonte única de config, sem flags).
- [x] Adicionar artifacts/ para relatórios, snapshots e gráficos.
- [x] Confirmar remoção do .docx e manter README.md como referência.
- [x] Consumo do cache (data/klines_cache.db) com auto-update opcional via JSON.
Dados / MTF

 - [x] Implementar loader MTF (data/loader.py) para 5m/15m/30m (cache + binance).
- [x] Calcular features por TF sem lookahead (implementado direto em scripts/backtest.py com EMA/ATR mínimos).
- [x] Alinhar ao 30m via merge_asof(direction='backward') (data/align.py).
- [x] Validar com dataset sintético um caso de no-lookahead (data/synth.py, smoke test OK).
Indicadores (plugáveis)

- [ ] Interface base (indicators/base.py) com apply(df), role.
- [x] EMA/ATR (indicators/common.py) + SMA/WMA/HMA + VWAP diário (reset por dia).
- [ ] Registrar catálogo simples (indicators/registry.py) e RSI.
Geradores de Sinal

- [ ] Interface base (signals/base.py) com generate(df).
- [x] Implementar signals/ema_cross.py (variantes A/B/C configuráveis no futuro; hoje A long-only).
- [ ] Integração de scores (ponderação) e prioridade de sinais.
Filtros

- [x] trend_mtf (legacy) + ma_trend genérico (ma_type: ema/sma/wma/hma).
- [x] atr_threshold (ATR mínimo 30m) – parametrizado via JSON.
- [x] volume_min (percentil mínimo).
- [x] Orquestrador de filtros (filters/apply.py), incluindo vwap_bias.
Risco / Execução

- [x] risk/sizing.py (fixed_fraction).
- [x] risk/stops.py (fixo ATR, trailing ATR).
- [x] risk/costs.py (fee por lado, slippage em ticks).
- [x] Aplicar na simulação em pontos de entrada/saída (engine/backtest.py).
Motor de Backtest

- [x] engine/backtest.py: loop 30m; sinais → filtros (incl. vwap) → risco; PnL.
- [x] Suporte a side long/short/both e exit_on_cross (config).
- [ ] Logging de eventos por barra (detalhes adicionais) para auditoria.
Otimização

- [x] scripts/optimize.py: Optuna com objetivo PF penalizado + Sharpe; mínimo de trades.
- [x] Robustez: ATR dinâmico; update_cache opcional; otimiza também ma_trend (ma_fast/ma_slow) e volume_min.percentile.
- [x] apply_best.py: aplica "best.*" do último Optimize no config_active.json (+ backups em artifacts/<run_id>/).
Walk-Forward

- [x] scripts/walk_forward.py: janelas configuráveis (60/20/20, 90/30/30, 60/30/30 validadas), rolling e acúmulo OOS.
- [x] Relatório agregado (PF, trades) + equity OOS; tags wfo_group/window_index no DB.
- [x] report_wfo: reconstrói do DB, gera equity/ventanas/params e per-window equity+drawdown.
Armazenamento / Auditoria

- [x] storage/db.py: SQLite experimento.sqlite.
- [x] Esquema: runs, bars, signals, trades, fills, params, metrics.
- [x] Persistir cada execução, com config e timestamps.
Interfaces / Relatórios

- [x] scripts/report.py: export CSVs do run + gráfico preço/EMAs/setas + equity.
- [x] scripts/report_wfo.py: relatórios agregados + por janela (linha, EMAs, entradas/saídas, equity, drawdown).
- [x] scripts/app.py (Flask): navegação de runs/WFO + botões (pipelines, rebuild WFO, cleanup, aplicar best, snapshot).
- [x] scripts/compare_wfo.py: CSV histórico com PF/trades por grupo WFO.
Configuração (JSON)

- [ ] config/schema.json (rótulos/validação leve).
- [ ] Exemplo atualizado em config/example.json; ativo em config_active.json.
- [x] Respeitar: nenhum flag — apenas poetry run python -m src.strategies.experimento.scripts.<modulo>.
Validação / Qualidade

- [ ] Testes unitários críticos de alinhamento no-lookahead e custos.
- [ ] Sanidade do PnL com custos vs sem custos.
- [ ] Perf: cache/localização de merges, reduzir cópias de DF.
Pipelines e automações

- [x] pipeline.py (Backtest → MonteCarlo → Report).
- [x] pipeline_wfo.py (Walk-Forward → report_wfo).
- [x] apply_best.py (aplicar "best.*" do Optimize no config).
- [x] snapshot_config.py (snapshot do config em artifacts/snapshots/).
- [x] cleanup.py (purge de artifacts conforme JSON).
