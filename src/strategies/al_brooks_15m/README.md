# Estratégia Al Brooks - Inside Bar

Este documento descreve a implementação, o uso e os resultados de uma estratégia de trading baseada nos ensinamentos de Al Brooks, focada no padrão de velas "Inside Bar".

## 1. Princípios da Estratégia

A estratégia foi baseada na transcrição de um vídeo que detalha um setup de Al Brooks. O foco principal é operar a **reversão de um pullback dentro de uma tendência estabelecida**.

- **Contexto de Tendência**: A tendência é definida pelo alinhamento de três Médias Móveis Exponenciais (EMAs).
  - **Tendência de Alta**: Preço acima das médias, com `EMA rápida > EMA média > EMA lenta`.
  - **Tendência de Baixa**: Preço abaixo das médias, com `EMA rápida < EMA média < EMA lenta`.

- **Sinal de Entrada**:
  - Ocorre um recuo (*pullback*) do preço em direção às médias móveis.
  - Durante esse recuo, um **Inside Bar** é formado (um candle contido dentro dos limites do candle anterior).
  - **Gatilho**: A entrada ocorre no rompimento da máxima (para compra) ou da mínima (para venda) do Inside Bar.

- **Filtro Principal (A Cereja do Bolo)**:
  - A transcrição destaca o uso de um indicador de "Afastamento Médio" para filtrar os melhores sinais.
  - A estratégia só considera uma entrada se o preço estiver muito próximo de uma EMA longa (ex: EMA 50), garantindo que a operação não ocorra com o preço "esticado".

- **Stop e Alvo**:
  - **Stop Loss**: Posicionado abaixo da mínima de todo o movimento de pullback (para compras) ou acima da máxima do rally (para vendas).
  - **Take Profit (Alvo)**: Definido por uma relação Risco/Recompensa (ex: 2.0, significando um alvo de 2x o tamanho do risco).

## 2. O Que Foi Feito por Nós

1.  **Implementação do Backtest**: O arquivo `backtest.py` foi criado para simular a estratégia, incluindo as lógicas de compra e venda, o filtro de afastamento médio e o gerenciamento de risco.
2.  **Otimização com Optuna**: O script `optimize.py` foi desenvolvido para encontrar os melhores parâmetros (períodos das EMAs, relação Risco/Recompensa e o valor do filtro de afastamento) usando um processo de otimização robusto.
3.  **Validação Fora da Amostra (Out-of-Sample)**: Implementamos um teste rigoroso onde a estratégia é otimizada em um longo período de dados de treino e validada em um período mais recente que o otimizador nunca viu. Isso é crucial para verificar se a estratégia não está sobreajustada (*overfitting*).
4.  **Validação Walk-Forward**: O script `walk_forward.py` implementa um teste ainda mais robusto, simulando a re-otimização periódica da estratégia, que é como um trader real se adaptaria às mudanças de mercado.
5.  **Monitoramento ao Vivo**: O script `live.py` foi criado para monitorar o mercado em tempo real, aplicando os parâmetros otimizados e imprimindo os sinais no console **(sem executar ordens reais)**.
6.  **Filtros de Robustez Adicionais**: A versão atual incorpora filtro de tendência via ADX, controle de volatilidade por ATR (stops dinâmicos e trailing) e um viés de tendência em timeframe superior, além de exigir um volume mínimo de trades por janela na validação walk-forward.

## 3. Principais Resultados Alcançados

Os testes no timeframe de 15m revelaram um forte **overfitting** no método de otimização simples e falta de robustez na validação walk-forward.

### Teste Out-of-Sample (Otimização Simples)

A estratégia mostrou lucro no período de treino, mas falhou em se generalizar para o período de validação.

- **Resultado em Amostra (Treino)**: P&L **+$546.84**, Profit Factor **1.96**
- **Resultado Fora da Amostra (Validação)**: P&L **-$99.78**, Profit Factor **0.84**

### Validação Walk-Forward

Mesmo re-otimizando a estratégia periodicamente, o resultado agregado ao longo de 22 períodos foi negativo.

- **P&L Total Agregado**: **-$414.88**
- **Taxa de Sucesso dos Períodos**: 86.36% (períodos com otimização bem-sucedida)

**Conclusão Principal**: Para o timeframe de 15m, a estratégia **não se mostrou robusta ou lucrativa** sob testes rigorosos. Os parâmetros que funcionam para um período de mercado não se mantêm para os períodos seguintes, indicando que a lógica pode precisar de filtros adicionais ou uma abordagem diferente.

## 4. Passo a Passo de Como Usar

O fluxo de trabalho é simples: atualizar a base de dados, otimizar a estratégia e, em seguida, rodar o backtest ou o monitoramento ao vivo.

### Passo 1: Atualizar a Base de Dados

Garanta que seu cache local de dados (`data/klines_cache.db`) esteja atualizado.

```bash
poetry run python -m scripts.populate_cache BTCUSDT 15m
```

### Passo 2: Otimizar a Estratégia (Optuna)

Este passo é crucial. Ele executa o teste *out-of-sample* e salva a melhor configuração encontrada em `reports/active/`.

```bash
# Exemplo com 5 anos de dados e 300 tentativas de otimização
poetry run python -m src.strategies.al_brooks_15m.optimize --days 1825 --trials 300
```

Ao final, um arquivo como `reports/active/ALBROOKS_BTCUSDT_15m.json` será criado.

### Passo 3: Executar o Backtest

O script de backtest carrega automaticamente a configuração ativa gerada no passo anterior e executa a simulação, gerando um relatório e um gráfico.

```bash
poetry run python -m src.strategies.al_brooks.backtest
```

O gráfico com os trades será salvo em `reports/charts/`.

### Passo 4: Executar a Validação Walk-Forward

Este script executa uma validação walk-forward completa, testando a estratégia em múltiplos períodos de otimização e validação. É uma forma robusta de testar a consistência da estratégia ao longo do tempo.

```bash
# Exemplo com 30 dias de otimização, 15 dias de validação, passo de 15 dias e exigindo 15 trades mínimos
poetry run python -m src.strategies.al_brooks_15m.walk_forward --opt-window 30 --val-window 15 --step-size 15 --min-trades 15
```

O script gera um relatório detalhado em `reports/walk_forward/` com estatísticas agregadas e gráficos de performance por período.

### Passo 5: Executar o Modo Live (Monitoramento)

Este script também carrega a configuração ativa e começa a monitorar o mercado em tempo real, buscando por sinais de compra ou venda.

**Importante**: Este modo apenas imprime os sinais no console, ele **não** executa ordens reais.

```bash
poetry run python -m src.strategies.al_brooks.live
```

Você verá o preço atual e os sinais sendo atualizados a cada 10 segundos.

////////////////////////////////////////////////////////////////////////////
2025-10-09 17:53:38,994 - INFO - Melhor score: 3.0435
2025-10-09 17:53:38,995 - INFO - Melhores parâmetros: {'ema_fast_period': 7, 'ema_medium_period': 11, 'ema_slow_period': 22, 'risk_reward_ratio': 1.2, 'max_avg_deviation_pct': 0.75, 'adx_threshold': 20.0, 'atr_stop_multiplier': 1.5, 'atr_trail_multiplier': 2.3000000000000003, 'htf_lookback': 15, 'min_atr': 49.5}
2025-10-09 17:53:39,190 - INFO - Resultado validação: P&L $121.07, Win Rate 75.00%, Profit Factor 8.88
2025-10-09 17:53:39,190 - WARNING - Nenhum período atingiu o mínimo de 15 trades. Agregando resultados com 21 períodos que tiveram trades.
2025-10-09 17:53:39,191 - INFO - Estatísticas agregadas:
2025-10-09 17:53:39,191 - INFO -   - Períodos bem sucedidos: 0/22
2025-10-09 17:53:39,191 - INFO -   - P&L Total: $-912.76
2025-10-09 17:53:39,191 - INFO -   - P&L Médio: $-43.46
2025-10-09 17:53:39,191 - INFO -   - Win Rate Médio: 27.72%
2025-10-09 17:53:39,191 - INFO -   - Profit Factor Médio: 1.46
2025-10-09 17:53:39,193 - INFO - Relatório salvo em: reports/walk_forward/BTCUSDT_15m_report.json
2025-10-09 17:53:40,032 - INFO - Gráfico salvo em: reports/charts/walk_forward_BTCUSDT_15m.png
2025-10-09 17:53:40,032 - INFO - Validação walk-forward concluída!

============================================================
RESUMO DA VALIDAÇÃO WALK-FORWARD
============================================================
Estratégia: AL Brooks
Símbolo: BTCUSDT
Timeframe: 15m
Trades mínimos por janela: 15
Períodos totais: 22
Períodos bem sucedidos: 0
Períodos com trades: 21
Taxa de sucesso: 0.00%
P&L Total: $-912.76
P&L Médio: $-43.46
Win Rate Médio: 27.72%
Profit Factor Médio: 1.46
Profit Factor Agregado: 0.70
Obs: nenhuma janela atingiu o mínimo de trades; métricas acima usam todos os períodos com trades.
============================================================

--- Otimização Concluída ---
Melhor valor (Profit Factor): 10.25
Melhores parâmetros encontrados:
{'ema_fast_period': 15, 'ema_medium_period': 36, 'ema_slow_period': 42, 'risk_reward_ratio': 2.1, 'max_avg_deviation_pct': 1.2000000000000002, 'adx_threshold': 28.0, 'atr_stop_multiplier': 1.1, 'atr_trail_multiplier': 2.9000000000000004, 'htf_lookback': 38, 'min_atr': 12.0}

Configuração ativa salva em: reports/active/ALBROOKS_BTCUSDT_15m.json

--- Resultado em Amostra (In-Sample / Treino) ---
P&L Final: $ 1801.58 | Trades: 560 | Win Rate: 38.04% | Profit Factor: 1.24

--- Resultado Fora da Amostra (Out-of-Sample / Validação) ---
P&L Final: $ -51.24 | Trades: 112 | Win Rate: 36.61% | Profit Factor: 0.98

Gerando gráfico do período de validação...

Gráfico do backtest salvo em: reports/charts/al_brooks_backtest_BTCUSDT_validation.png
[zanotti@automacaosrv1 teste-previsao3]$ docker compose exec app python -m src.strategies.al_brooks_15m.backtest
Usando configuração ativa para BTCUSDT@15m
Carregando dados: BTCUSDT @ 15m dos últimos 1825 dias...
Total de 175106 candles carregados.
Executando backtest...

--- Resultados do Backtest ---
Período Analisado: 2020-10-10 19:30:00 a 2025-10-09 19:15:00
Resultado Final (P&L): $ -5721.77
Total de Operações Fechadas: 1206
Taxa de Acerto: 35.66%
Profit Factor: 0.45
Média de Ganho: $ 10.82
Média de Perda: $ 13.37
Duração Média do Trade: 0 days 00:31:24

Gráfico do backtest salvo em: reports/charts/al_brooks_backtest_BTCUSDT.png

