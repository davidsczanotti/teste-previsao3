# Análise Profunda: O Colapso de 2021

## Resumo Estatístico (BTCUSDT 4h)
- **Total de Trades:** 127
- **PnL Total:** -$759.51 (aprox. -38% do capital naquele momento)
- **Win Rate Geral:** 28.35% (Extremamente baixo)

## Onde o dinheiro foi perdido?
A maior parte do prejuízo veio de operações **LONG (Compra)**:
- **Longs:** 89 trades, Prejuízo de -$739.70 (Win Rate: 25%)
- **Shorts:** 38 trades, Prejuízo de -$19.81 (Win Rate: 34%)

Isso é paradoxal para um ano de Bull Market (BTC subiu de 29k para 69k), indicando que a estratégia comprava sistematicamente **topos locais**.

## Diagnóstico dos Problemas

### 1. Compra de Topo (FOMO Estrutural)
Exemplo Crítico: `2021-01-08`.
- O Bitcoin atingiu ~42k (topo histórico na época).
- A estratégia entrou Long em **41.2k** (quase na máxima exata).
- Stopado no mesmo dia em 39.3k.
- **Causa:** A lógica de entrada exige "confirmação" (médias se separando). Em 2021, os movimentos eram explosivos e verticais. Quando as médias confirmavam a tendência, o preço já estava esticado e pronto para corrigir.

### 2. "Whipsaw" em Mercados Laterais (Março 2021)
Mês com maior prejuízo (-$380).
- O Bitcoin ficou lateralizando entre 50k e 60k.
- A estratégia tentava comprar rompimentos de 55k/58k, o preço falhava e voltava, ativando o Stop Loss.
- Como o `Win Rate` foi < 30%, a estratégia sangrou lentamente tentando acertar uma tendência que não se sustentava.

### 3. Volatilidade Extrema em Shorts (Maio 2021)
Exemplo: `2021-05-19` (O Crash dos 30k).
- A estratégia entrou Short em **37.3k** (no meio do pânico).
- O preço teve um repique violento (volatilidade) para 40k no mesmo candle/dia.
- A estratégia foi stopada na volatilidade antes que a tendência de baixa se firmasse.

## Conclusão
O ano de 2021 foi caracterizado por **"Fakeouts" (Falsos Rompimentos)** e **Reversões em V**.
A estratégia, sendo seguidora de tendência (Trend Following), precisa de movimentos contínuos. Em 2021, o mercado dava o sinal de entrada e revertia imediatamente.

**Por que a estratégia funciona no longo prazo (+97%)?**
Porque em anos como 2017 (Bull Run limpa) ou 2022 (Bear Market limpo), as tendências duram semanas sem essas violações violentas, permitindo que os lucros (agora com alvo 2.0x) cubram os prejuízos de anos "sujos" como 2021.
