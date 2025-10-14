# Plano de Melhorias para a Estratégia (Estágio 2)

Este documento descreve um plano de ação para evoluir a estratégia `candle_pattern7_rl`, com foco em torná-la mais robusta e adaptável às constantes mudanças nas condições do mercado (regimes).

---

## 1. Aprendizado Contínuo (Online Learning)

**Objetivo:** Fazer o agente se adaptar continuamente aos dados mais recentes do mercado, evitando a degradação do modelo ao longo do tempo.

**Passo a Passo Sugerido:**

1.  **Definir o Ciclo de Treinamento:** Estabeleça uma frequência para o re-treinamento (ex: semanal, mensal, trimestral). Um ciclo mensal é um bom ponto de partida.

2.  **Coleta de Novos Dados:** Garanta que seu processo de obtenção de dados (`get_historical_klines`) consiga baixar apenas os dados mais recentes desde o último treinamento.

3.  **Automatizar o Re-treinamento:** Crie um script (ex: `run_retraining.sh`) que execute os seguintes comandos:
    *   Baixar os dados mais recentes e juntá-los ao dataset de treino existente.
    *   Executar o script `train.py` usando o modelo treinado anteriormente como ponto de partida. O flag `--model` é essencial aqui.
    *   Exemplo de comando no script:
        ```bash
        # Caminho para o modelo treinado na última execução
        MODELO_ANTERIOR="reports/agents/candle_pattern7_rl/BTCUSDT_15m.npz"
        
        # Comando para continuar o treinamento
        poetry run python -m src.strategies.candle_pattern7_rl.train \
          --ticker BTCUSDT --interval 15m --days 3650 \
          --episodes 100 \ # Menos episódios, pois é um ajuste fino
          --model "$MODELO_ANTERIOR" \
          # ... (outros parâmetros importantes)
        
        echo "Re-treinamento concluído!"
        ```

4.  **Validação e Implantação:** Após o re-treinamento, execute um backtest rápido em um período de validação recente (que não foi usado no treino) para garantir que o desempenho do agente não piorou. Se os resultados forem satisfatórios, o novo arquivo `.npz` do modelo se torna o modelo de produção.

---

## 2. Evolução da Arquitetura do Agente

**Objetivo:** Substituir a rede neural atual por uma arquitetura mais poderosa, capaz de interpretar melhor as sequências de dados temporais.

**Passo a Passo Sugerido:**

1.  **Escolha da Arquitetura:**
    *   **Transformers:** Ideal para capturar a importância relativa entre os candles na janela de observação. É o estado da arte para muitas tarefas de sequência.
    *   **LSTMs/GRUs:** Uma opção mais clássica e robusta para modelagem de séries temporais, focada em manter uma "memória" de eventos passados.

2.  **Modificação do Código:**
    *   O principal arquivo a ser modificado será o `policy.py`. Você precisará definir uma nova classe de política (ex: `TransformerPolicy` ou `LSTMPolicy`) que implemente a nova arquitetura.
    *   Bibliotecas como `PyTorch` ou `TensorFlow/Keras` já possuem implementações de alto nível para essas camadas, o que facilita a integração.

3.  **Ajuste do Ambiente (`env.py`):** A dimensionalidade da observação pode precisar de ajustes para se adequar à entrada da nova arquitetura (especialmente para LSTMs, que esperam um formato de `[batch, timesteps, features]`).

4.  **Treinamento e Comparação:** Treine um novo agente do zero com a nova arquitetura e compare seu desempenho (lucro, drawdown, número de trades) com o agente original em um mesmo período de validação.

---

## 3. Enriquecimento do Contexto (Features)

**Objetivo:** Fornecer ao agente uma visão mais completa do mercado, além do tempo gráfico atual.

**Passo a Passo Sugerido:**

1.  **Análise Multi-Timeframe:**
    *   No `env.py`, dentro da função que prepara os dados (como `_sliding_window_features`), adicione a lógica para carregar dados de outros tempos gráficos (ex: 1h e 4h).
    *   Calcule indicadores nesses tempos gráficos maiores (ex: `SMA(20)` no 1h, `ADX(14)` no 4h).
    *   Adicione esses valores como novas features na observação que o agente recebe a cada passo. Cuidado para alinhar os timestamps corretamente.

2.  **Dados Alternativos:**
    *   **Funding Rates:** Se estiver operando futuros perpétuos, crie uma função para buscar o histórico de funding rates e adicione-o como uma feature. Um funding rate muito positivo pode indicar euforia e uma possível reversão.
    *   **Outros:** Investigue APIs para dados de sentimento ou volatilidade.

3.  **Re-treinamento:** Com as novas features, o espaço de observação do agente mudou. Você precisará re-treinar o modelo do zero para que ele aprenda a utilizar essas novas informações.

---

## 4. Detecção de Regime de Mercado

**Objetivo:** Informar explicitamente ao agente qual é o comportamento atual do mercado.

**Passo a Passo Sugerido:**

1.  **Definir os Regimes:** Crie uma função que classifique cada candle no histórico em um regime. Exemplos:
    *   **Volatilidade:** Use o `ATR` (Average True Range). Se `ATR` > `SMA(ATR, 50)`, regime de "Alta Volatilidade".
    *   **Tendência:** Use o `ADX`. Se `ADX` > 25, regime de "Tendência Forte".
    *   **Combinação:** `ADX > 25` e `Close > SMA(200)` -> "Tendência de Alta Forte".

2.  **Integrar como Feature:**
    *   No `env.py`, adicione a classificação de regime como uma nova feature na observação do agente. Use "one-hot encoding" se tiver múltiplas classes de regime (ex: `[1, 0, 0]` para Tendência de Alta, `[0, 1, 0]` para Tendência de Baixa, etc.).

3.  **Treinamento:** Assim como na adição de features, o agente precisará ser treinado do zero para aprender a reagir de forma diferente a cada regime de mercado. Ele pode aprender, por exemplo, que em "Mercado Lateral" a melhor estratégia é não fazer nada ou fazer trades curtos, enquanto em "Tendência de Alta Forte" ele pode segurar a posição por mais tempo.
