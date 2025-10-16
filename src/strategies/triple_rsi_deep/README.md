# Estratégia: Triple RSI com Filtros de Regime

Este documento descreve a implementação, uso e filosofia de uma estratégia de trading quantitativa baseada na confluência de três Indicadores de Força Relativa (RSI) em diferentes períodos, enriquecida com filtros de regime de mercado.

## 1. Princípios da Estratégia

O núcleo da estratégia é usar diferentes "lentes" (RSIs) para analisar o mercado, garantindo que operamos apenas em condições de alta probabilidade.

- **Contexto de Regime (RSI Lento)**:
  - Um RSI de longo prazo (ex: 200 períodos) atua como nosso "mapa do tempo".
  - **Regime de Alta**: `RSI_Lento > 50`. Apenas operações de compra são permitidas.
  - **Regime de Baixa**: `RSI_Lento < 50`. Apenas operações de venda são permitidas.

- **Sinal de Pullback (RSI Médio)**:
  - Um RSI de médio prazo (ex: 50 períodos) identifica recuos contra a tendência principal.
  - Em um regime de alta, esperamos o `RSI_Médio` cair para uma zona de "oportunidade" (ex: abaixo de 40), sinalizando um pullback.
  - Em um regime de baixa, esperamos o `RSI_Médio` subir para uma zona de "fraqueza" (ex: acima de 60).

- **Gatilho de Entrada (RSI Rápido)**:
  - Um RSI de curto prazo (ex: 14 períodos) confirma o fim do pullback e a retomada da tendência.
  - **Gatilho de Compra**: Após um sinal de pullback, a entrada ocorre quando o `RSI_Rápido` cruza para cima de 50.
  - **Gatilho de Venda**: Após um sinal de pullback, a entrada ocorre quando o `RSI_Rápido` cruza para baixo de 50.

- **Filtros de Qualidade**:
  - **ADX (Average Directional Index)**: Garante que o mercado está em tendência (`ADX > threshold`), evitando "ruído" de mercados laterais.
  - **ATR (Average True Range)**: Garante que há volatilidade mínima para o trade se desenvolver (`ATR > threshold`), evitando mercados "parados".

- **Gerenciamento de Risco**:
  - **Stop Loss**: Dinâmico, baseado em um múltiplo do ATR para se adaptar à volatilidade atual.
  - **Take Profit**: Definido por uma relação Risco/Recompensa fixa (ex: 1.5x o risco inicial).

## 2. Estrutura e Metodologia

1.  **Implementação Vetorizada (`strategy.py`)**: A lógica foi implementada usando `vectorbt`, uma biblioteca de alta performance que permite testar milhões de combinações de parâmetros em segundos, em vez de horas.
2.  **Otimização Robusta (`optimize.py`)**: Usamos `Optuna` para buscar os melhores parâmetros. O objetivo da otimização não é o lucro bruto, mas uma métrica mais robusta como o **Sortino Ratio**, que penaliza a volatilidade negativa.
3.  **Validação Fora da Amostra**: O processo de otimização divide os dados em "treino" e "validação". A estratégia é otimizada nos dados de treino e, em seguida, seu desempenho real é medido nos dados de validação, que o otimizador nunca viu. Isso previne o *overfitting*.

## 3. Como Usar

O fluxo de trabalho é projetado para ser simples e robusto.

### Passo 1: Otimizar a Estratégia

Este é o passo mais importante. Ele executa a otimização e salva a melhor configuração encontrada.

```bash
# Otimiza usando 365 dias de dados e executa 100 tentativas
docker compose exec app python -m src.strategies.triple_rsi_deep.optimize --days 365 --trials 100
```

Um arquivo `reports/active/TRIPLE_RSI_DEEP_BTCUSDT_1m.json` será criado com os melhores parâmetros.

### Passo 2: Executar o Backtest de Validação

Este script carrega os parâmetros otimizados e executa um backtest detalhado no período de validação, gerando um relatório completo e um gráfico.

```bash
docker compose exec app python -m src.strategies.triple_rsi_deep.backtest
```

O relatório será salvo em `reports/charts/`.

### Próximos Passos (Avançado)

- **Walk-Forward Validation**: Implementar um script `walk_forward.py` para simular a re-otimização periódica da estratégia, o teste mais rigoroso de robustez.
- **Live Trading**: Criar um script `live.py` que carrega os parâmetros e monitora o mercado em tempo real para gerar sinais.