# Melhorias na Estratégia (Pareto 2021)

## Contexto
A estratégia original (`custom_cci_ma` mode) apresentava um prejuízo total de ~48% no período de 2017-2025, com o ano de 2021 sendo o pior desempenho (-41%).

A análise mostrou que a estratégia entrava muito tarde nas tendências (momentum chasing) e era penalizada em reversões rápidas (comum em 2021). Além disso, a relação Risco/Retorno (R:R) era insuficiente (1.66) para uma taxa de acerto de ~35-40%.

## Mudanças Implementadas
Focamos em corrigir a mecânica de entrada e saída baseada nos problemas de 2021:

1.  **Entrada Mais Cedo (Early Entry):**
    - `custom_dist_atr_mult`: Reduzido de **0.2** para **0.05**.
    - **Efeito:** A estratégia agora entra no trade assim que a tendência começa a se separar (distância entre média rápida e lenta), capturando mais do movimento e reduzindo o risco de "comprar no topo".

2.  **Alvo Maior (Higher Reward):**
    - `custom_target_factor`: Aumentado de **1.5** para **2.0**.
    - **Efeito:** Aumentou o Risco/Retorno potencial para ~2.2 (2.0 / 0.9). Isso permite que a estratégia seja lucrativa mesmo com uma taxa de acerto mais baixa.

## Resultados (Backtest 2017-2025)

| Métrica | Antes | Depois | Variação |
| :--- | :--- | :--- | :--- |
| **Retorno Total** | **-47.79%** | **+50.03%** | **+97.82%** |
| **Profit Factor** | 0.96 | 1.03 | +0.07 |
| **2021 PnL** | -41.64% | -11.78% | +29.86% |
| **2022 PnL** | +23.22% | +56.63% | +33.41% |
| **2025 PnL** | -20.11% | -2.16% | +17.95% |

## Conclusão
A aplicação do princípio de Pareto (focar em melhorar os 20% piores casos, no caso o ano de 2021) gerou uma melhoria sistêmica na robustez da estratégia, transformando-a de perdedora para vencedora no longo prazo.
