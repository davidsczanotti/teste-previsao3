# Análise da Discrepância de 2021

A diferença entre o resultado do teste isolado (-11% ou positivo) e o teste completo (-38%) ocorre devido ao **"Cold Start" (Início Frio)** dos indicadores.

1.  **Teste Isolado (2021 apenas):**
    - Ao carregar dados começando em `2021-01-01`, indicadores longos como a **EMA 200** (usada no filtro de tendência) precisam de 200 períodos para serem calculados.
    - Consequência: A estratégia ficou "cega" (sem sinais) durante os primeiros meses de 2021 (Jan-Jun), que foi justamente o período mais volátil e difícil. Ao não operar, ela "economizou" prejuízos.

2.  **Teste Completo (2017-2025):**
    - Em 2021, a estratégia já possui histórico desde 2017. Os indicadores estão formados e ativos.
    - Consequência: A estratégia operou em Jan/Fev/Mai de 2021, tomando os stops normais da volatilidade daquele ano.

**Conclusão:**
O resultado de **-38%** é o desempenho **real** e honesto para 2021.
Apesar disso, as melhorias implementadas (Entrada Antecipada + Alvo Maior) salvaram a estratégia no longo prazo, garantindo o lucro total de **+97%** (contra o prejuízo original de -47%). O ano de 2021 continua sendo o "calcanhar de Aquiles", mas não quebra mais a conta no longo prazo.
