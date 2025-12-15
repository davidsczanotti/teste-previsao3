# Scanner de Mercado - EMA Only Strategy

Este módulo é uma ferramenta de **Asset Selection (Seleção de Ativos)**. Ele automatiza o processo de rodar a estratégia "EMA Only / Custom CCI" (configuração otimizada) em dezenas de ativos simultaneamente para identificar onde ela tem melhor performance histórica.

## 🚀 Como Usar

### Comandos Básicos

O ponto de entrada é o script `src/scanner/run.py`. Execute via Poetry:

1.  **Escanear Ações Brasileiras (B3)**
    Analisa uma lista curada (~80 ativos) das ações mais líquidas (IBOV/SMLL).
    ```bash
    poetry run python src/scanner/run.py --type b3
    ```

2.  **Escanear Criptomoedas (Binance)**
    Busca automaticamente pares USDT ativos na Binance.
    ```bash
    poetry run python src/scanner/run.py --type crypto
    ```

3.  **Escanear Tudo (B3 + Cripto)**
    Gera um ranking unificado comparando Ações e Criptos.
    ```bash
    poetry run python src/scanner/run.py --type all
    ```

4.  **Teste Rápido (Limitado)**
    Útil para verificar se tudo funciona sem esperar o download de todos os ativos. Exemplo: 5 ativos de cada.
    ```bash
    poetry run python src/scanner/run.py --type all --limit 5
    ```

---

## ⚙️ Personalização (Adicionar/Remover Tickers)

As listas de ativos são gerenciadas no arquivo:
📂 `src/scanner/fetchers.py`

### Para Ações B3:
Edite a função `get_b3_tickers()`:
```python
def get_b3_tickers() -> List[str]:
    tickers = [
        "PETR4.SA", "VALE3.SA", ... 
        "SUA_ACAO_AQUI.SA", # Adicione seu ticker com .SA
    ]
    # Remova linhas de ativos que não deseja analisar
    return tickers
```

### Para Criptomoedas:
A função `get_crypto_tickers()` busca dinamicamente na API da Binance.
Se quiser forçar uma lista manual, você pode editar o retorno dela ou adicionar filtros (ex: filtrar por volume mínimo).

---

## 📊 Analisando o Relatório (scanner_results.csv)

O script gera um arquivo CSV (`scanner_results.csv`) na raiz do projeto e exibe o TOP 20 no terminal.

### Colunas do Relatório:

1.  **Ticker:** Código do ativo (ex: `MGLU3.SA`, `BTCUSDT`).
2.  **Retorno Total (%):** Lucro acumulado no período (2017-2025). Quanto maior, melhor.
3.  **Win Rate (%):** Taxa de acerto. Para essa estratégia (Trend Following), valores acima de **35-40%** são excelentes. Valores abaixo de 30% indicam dificuldade.
4.  **Profit Factor:** Relação Lucro Bruto / Prejuízo Bruto.
    *   `> 1.0`: Lucrativa.
    *   `> 1.5`: Excelente.
    *   `< 1.0`: Prejuízo.
5.  **Max Drawdown (%):** A pior queda acumulada do topo ao fundo. Indica o risco.
    *   Exemplo: `-40%` significa que em algum momento seu capital caiu 40% antes de recuperar.
6.  **Type:** B3 (Ação) ou CRYPTO.

### Exemplo de Interpretação (Fictício):

| Ticker | Retorno Total (%) | Win Rate (%) | Profit Factor | Max Drawdown (%) | Análise |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MGLU3.SA** | 877.6% | 43.3% | 2.10 | -40.1% | **Excelente.** Ativo de crescimento forte, ideal para a estratégia. |
| **WEGE3.SA** | 85.1% | 38.1% | 1.55 | -38.5% | **Muito Bom.** Crescimento consistente e risco controlado. |
| **PETR4.SA** | 14.0% | 37.0% | 1.15 | -41.9% | **Ok.** Lucrativa, mas retorno baixo para o risco (drawdown alto). |
| **LREN3.SA** | -72.9% | 27.8% | 0.60 | -78.8% | **Péssimo.** Ativo cíclico/lateral ruim para Trend Following. Evitar. |

### Dica de Ouro (Seleção de Ativos):
Não olhe apenas para o **Retorno Total**. Procure ativos com **Profit Factor > 1.2** e **Win Rate > 35%**. Isso indica consistência, não apenas sorte em um único trade.