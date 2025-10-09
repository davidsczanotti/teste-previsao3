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
