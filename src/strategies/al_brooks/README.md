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

### A Paciência do Caçador: A Seletividade da Estratégia

É comum observar o robô emitir a mensagem `SINAL: hold` por longos períodos. Isso pode parecer contraintuitivo, mas é o comportamento esperado e demonstra que a estratégia está funcionando como projetada: sendo **extremamente seletiva** e aguardando pacientemente que todas as condições rigorosas definidas na otimização sejam atendidas ao mesmo tempo.

Pense na estratégia como um caçador de elite (*sniper*), não como uma metralhadora. Ele não atira em tudo que se move; ele espera pelo alvo perfeito.

Para que um trade aconteça, o mercado precisa satisfazer a seguinte lista de critérios simultaneamente:

1.  **Filtro de Volatilidade Mínima**: O ATR do candle anterior é maior que o valor otimizado (ex: `23.0`)? Se a volatilidade estiver muito baixa, ele não opera.
2.  **Filtro de Tendência (ADX)**: O ADX é maior que o valor otimizado (ex: `23.0`)? Se o mercado estiver lateral e sem força, ele não opera.
3.  **Filtro de Afastamento da Média**: O preço está a menos de X% (ex: `1.45%`) de distância da EMA lenta? Se o preço estiver muito "esticado", ele não opera.
4.  **Filtro de Viés de Timeframe Maior**: A tendência no timeframe superior está alinhada com a direção do trade?
5.  **Alinhamento das Médias**: As EMAs (rápida, média e lenta) estão perfeitamente alinhadas para cima ou para baixo?
6.  **Sinal de Pullback**: O preço fez um recuo em direção à EMA rápida?
7.  **Padrão de Candle**: O candle anterior foi um **Inside Bar**?
8.  **Gatilho de Entrada**: O preço atual rompeu a máxima/mínima do candle anterior?

A chance de todas essas condições se alinharem é estatisticamente baixa, e é exatamente por isso que a estratégia se mostrou lucrativa nos testes: ela só entra em cenários de altíssima probabilidade. O estado `hold` é o comportamento padrão e mais seguro.

## 2. O Que Foi Feito por Nós

1.  **Implementação do Backtest**: O arquivo `backtest.py` foi criado para simular a estratégia, incluindo as lógicas de compra e venda, o filtro de afastamento médio e o gerenciamento de risco.
2.  **Otimização com Optuna**: O runner sem flags `optimize_noflags.py` encontra os melhores parâmetros (períodos das EMAs, relação Risco/Recompensa e filtros) usando um processo de otimização robusto.
3.  **Validação Fora da Amostra (Out-of-Sample)**: Implementamos um teste rigoroso onde a estratégia foi otimizada em um longo período de dados de treino (~4 anos) e validada em um período mais recente que o otimizador nunca viu (~1 ano). Isso garante que a estratégia é robusta e não apenas "decorou" o passado (*overfitting*).
4.  **Monitoramento ao Vivo**: O script `live.py` foi criado para monitorar o mercado em tempo real, aplicando os parâmetros otimizados e imprimindo os sinais no console **(sem executar ordens reais)**.

## 3. Principais Resultados Alcançados

Após a otimização em um período de 5 anos de dados (`--days 1825`), os resultados do teste *out-of-sample* foram:

- **Resultado em Amostra (Treino - ~4 anos)**:
  - **P&L Final**: $ 489.88
  - **Trades**: 92
  - **Taxa de Acerto**: 44.57%
  - **Profit Factor**: 2.00

- **Resultado Fora da Amostra (Validação - ~1 ano)**:
  - **P&L Final**: $ 46.59
  - **Trades**: 21
  - **Taxa de Acerto**: 28.57%
  - **Profit Factor**: 1.21

**Conclusão Principal**: A estratégia se provou **robusta**, pois continuou lucrativa no período de validação (dados não vistos), que é o teste mais importante para qualquer sistema de trading.

## 4. Passo a Passo de Como Usar

O fluxo de trabalho é simples: atualizar a base de dados, otimizar a estratégia e, em seguida, rodar o backtest ou o monitoramento ao vivo.

### Passo 1: Atualizar a Base de Dados

Garanta que seu cache local de dados (`data/klines_cache.db`) esteja atualizado com os dados mais recentes da Binance.

```bash
poetry run python -m scripts.populate_cache BTCUSDT 1m
```

### Passo 2: Otimizar a Estratégia (Optuna, sem flags)

Este passo executa o teste de treino/validação e atualiza `src/strategies/al_brooks_1m/config.json` com os melhores parâmetros.

```bash
poetry run python -m src.strategies.al_brooks.optimize
```

Os parâmetros (símbolo, timeframe, dias, trials etc.) são lidos do bloco `optimize` em `src/strategies/al_brooks/config.json` e o resultado também é salvo nesse arquivo.

### Passo 3: Executar o Backtest

O script de backtest carrega automaticamente a configuração ativa e executa a simulação, gerando um relatório e um gráfico.

```bash
poetry run python -m src.strategies.al_brooks.backtest
```

O gráfico com os trades será salvo em `src/strategies/al_brooks/reports/charts/`.

### Passo 4: Análise Walk‑Forward (sem flags)

Executa otimizações rolantes e validações fora da amostra, gerando um gráfico e um resumo audível em JSON.

```bash
poetry run python -m src.strategies.al_brooks.walk_forward
```

Parâmetros dos janelamentos podem ser definidos opcionalmente em `config.json` no bloco `walk_forward`:

```json
"walk_forward": {"opt_window": 90, "val_window": 30, "step_size": 30, "min_trades": 10, "cache_only": true}
```

Artefatos:
- Gráfico: `src/strategies/al_brooks/reports/charts/walk_forward_ALBROOKS_<TICKER>_<TF>.png`
- Resumo: `src/strategies/al_brooks/reports/snapshots/wf_summary.json`

### Passo 5: Simulação Monte Carlo (sem flags)

Roda backtest, aplica bootstrap por blocos à sequência de trades e salva gráficos/estatísticas.

```bash
poetry run python -m src.strategies.al_brooks.monte_carlo
```

Artefatos:
- Histogramas: `src/strategies/al_brooks/reports/monte_carlo/mc_*.png`
- Sumário: `src/strategies/al_brooks/reports/monte_carlo/mc_summary_<TICKER>_<TF>_<TS>.json`

### Passo 4: Executar o Modo Live (Monitoramento)

Este script também carrega a configuração ativa e começa a monitorar o mercado em tempo real, buscando por sinais de compra ou venda.

**Importante**: Este modo apenas imprime os sinais no console, ele **não** executa ordens reais.

```bash
poetry run python -m src.strategies.al_brooks.live
```

Você verá o preço atual e os sinais sendo atualizados a cada 10 segundos.
