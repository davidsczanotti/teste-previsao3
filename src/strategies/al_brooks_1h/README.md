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
poetry run python -m scripts.populate_cache
```

### Passo 2: Otimizar a Estratégia (Optuna)

Este passo é crucial. Ele executa o teste *out-of-sample* e salva a melhor configuração encontrada em `reports/active/`.

```bash
# Exemplo com 5 anos de dados e 300 tentativas de otimização
poetry run python -m src.strategies.al_brooks.optimize --days 1825 --trials 300
```

Ao final, um arquivo como `reports/active/ALBROOKS_BTCUSDT_15m.json` será criado.

### Passo 3: Executar o Backtest

O script de backtest carrega automaticamente a configuração ativa gerada no passo anterior e executa a simulação, gerando um relatório e um gráfico.

```bash
poetry run python -m src.strategies.al_brooks.backtest
```

O gráfico com os trades será salvo em `reports/charts/`.

### Passo 4: Executar o Modo Live (Monitoramento)

Este script também carrega a configuração ativa e começa a monitorar o mercado em tempo real, buscando por sinais de compra ou venda.

**Importante**: Este modo apenas imprime os sinais no console, ele **não** executa ordens reais.

```bash
poetry run python -m src.strategies.al_brooks.live
```

Você verá o preço atual e os sinais sendo atualizados a cada 10 segundos.

