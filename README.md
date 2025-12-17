# 📈 TrendSurfer & SuperTrend AI - Sistema de Análise Quant

Este é um sistema modular em Python desenvolvido para **análise técnica**, **backtesting** e **otimização** de estratégias de trading automatizado, com foco no mercado brasileiro (B3) e suporte a Criptomoedas.

O projeto evoluiu de cruzamentos simples de médias móveis para algoritmos adaptativos baseados em IA, como o **SuperTrend AI** (baseado em clusterização não-supervisionada).

---

## 🚀 Funcionalidades Principais

*   **Scanner de Mercado:** Varre automaticamente listas de ativos (ex: carteira IDIV) e gera um ranking de performance comparando a Estratégia vs. Buy & Hold.
*   **Análise Individual (CLI):** Ferramenta de terminal que fornece um "Veredito" instantâneo (Comprar/Vender/Manter) para qualquer ativo, incluindo histórico recente de trades simulados.
*   **Inteligência Artificial:**
    *   **SuperTrend AI:** Estratégia que usa *K-Means Clustering* para adaptar dinamicamente os parâmetros de tendência à volatilidade recente do mercado.
    *   **Otimização de Hiperparâmetros:** Integração com **Optuna** para descobrir matematicamente os melhores parâmetros (Médias, Stops, Multiplicadores) para cada ativo específico.
*   **Backtesting Profissional:** Engine de simulação que considera taxas, slippage, custos de transação e gestão de portfólio (100% Equity ou Risco Fixo).

---

## 🛠️ Instalação

Este projeto utiliza **Poetry** para gerenciamento de dependências.

1.  **Pré-requisitos:** Python 3.10+ e Poetry instalados.
2.  **Instalação:**

```bash
# Instalar dependências do projeto
poetry install
```

---

## 💻 Como Usar

### 1. Analisar um Ativo (O "Cérebro")
Use o `main.py` para receber um relatório técnico completo sobre uma ação específica. Ele mostra a tendência atual, o preço de Stop Loss dinâmico e o resultado dos últimos 5 trades teóricos.

```bash
# Analisar um ticker específico
poetry run python main.py WEGE3.SA

# Ou entrar no modo interativo (digite os tickers sequencialmente)
poetry run python main.py
```

**Exemplo de Saída:**
> 🤖 RELATÓRIO SUPERTREND AI: WEGE3.SA
> 🌊 SURFANDO A ALTA (HOLD) | Stop Loss: R$ 45.33

### 2. Scanner de Oportunidades (O "Radar")
Roda a estratégia selecionada (padrão: SuperTrend AI) em toda a carteira teórica do **IDIV (Índice de Dividendos)**.

```bash
poetry run python run_scanner.py
```
*   **Saída:** Gera um relatório CSV na pasta `reports/` (ex: `reports/scanner_idiv_20251217.csv`) e exibe o ranking no terminal.

### 3. Otimização de Estratégia (A "Inteligência")
Se você quer descobrir qual a melhor média móvel ou stop loss para um ativo específico (ex: PETR4), use o otimizador.

```bash
poetry run python run_optimizer.py
```
*(Edite o arquivo `run_optimizer.py` para alterar o ativo alvo).*

---

## 📂 Estrutura do Projeto

```text
.
├── main.py                 # CLI para análise individual interativa
├── run_scanner.py          # Script de varredura em lote (IDIV)
├── run_optimizer.py        # Script de otimização com Optuna
├── reports/                # Pasta onde os relatórios CSV são salvos
├── src/
│   ├── core/               # Núcleo do sistema
│   │   ├── backtest.py     # Engine de execução de trades e cálculo de PnL
│   │   ├── indicators.py   # Biblioteca de indicadores (RSI, ADX, SuperTrend, etc.)
│   │   └── signals.py      # Lógica de decisão (Compra/Venda) das estratégias
│   └── strategies/
│       └── supertrend_ai.py # Implementação da lógica de Clusterização (LuxAlgo)
└── scripts/
    └── legacy/             # Scripts antigos e exemplos arquivados
```

## 📊 Estratégias Disponíveis

O sistema suporta múltiplas estratégias, configuráveis via código:

1.  **SuperTrend AI (Recomendada):** Adapta o fator do SuperTrend usando Machine Learning (Clustering) para identificar o melhor comportamento recente. Ótima para ativos cíclicos e voláteis.
2.  **V6 Robust (Dynamic Volatility):** Usa ATR para definir Stops dinâmicos e ADX para filtrar tendências fortes.
3.  **Trend Surfer v4:** Cruzamento clássico de médias (9x21) com filtro de tendência macro (EMA 200) e Momentum (CCI).
4.  **EMA Cross:** Cruzamento simples de médias (Baseline).

---

## ⚠️ Disclaimer

Este software é uma ferramenta de **análise e educação financeira**. Resultados passados (backtests) não garantem lucros futuros. O mercado de renda variável envolve riscos. Use com responsabilidade.