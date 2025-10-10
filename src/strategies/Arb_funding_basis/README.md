# Estratégia de Arbitragem de Funding Rate (Delta Neutro)

Este documento descreve a implementação, o uso e os resultados de uma estratégia de arbitragem de taxa de financiamento (*funding rate*), também conhecida como *basis trading*.

## 1. Princípios da Estratégia

O objetivo é lucrar com a diferença de juros entre o mercado à vista (spot) e o de futuros perpétuos, mantendo uma exposição neutra à variação de preço do ativo.

- **Conceito**: A estratégia é "delta neutro", o que significa que o lucro não depende se o preço do ativo sobe ou desce.

- **Mecânica da Operação**:
  1.  **Compra no Spot**: Comprar uma quantidade X do ativo (ex: 1 BTC).
  2.  **Venda no Perpétuo**: Vender a mesma quantidade X do ativo no mercado de futuros perpétuos (short 1 BTC).
  3.  **Coleta de Funding**: Enquanto a taxa de *funding* for positiva, o trader que está vendido (short) recebe pagamentos periódicos (geralmente a cada 8 horas).

- **Fonte de Lucro**:
  - **Lucro Bruto**: A soma de todos os pagamentos de *funding* recebidos.
  - **Custo**: As taxas de negociação para abrir e fechar as posições spot e de futuros.
  - **Lucro Líquido**: `Funding Coletado - Taxas Pagas`.

- **Gatilhos de Entrada e Saída**:
  - **Entrada**: A posição é aberta quando o *funding rate* ultrapassa um limiar positivo definido (ex: `> 0.01%`), tornando a operação potencialmente lucrativa.
  - **Saída**: A posição é fechada quando o *funding rate* cai para zero ou se torna negativo, eliminando a fonte de lucro.

## 2. Implementação

O script `arb_funding_basis.py` implementa um backtest completo para essa estratégia.

- **Coleta de Dados**: Busca o histórico de preços do ativo no mercado spot, o preço do USDC (para monitorar riscos de de-peg) e o histórico de taxas de financiamento da Binance.
- **Simulação**: Itera sobre os dados hora a hora, simulando a lógica de entrada e saída com base nos limiares de *funding rate*.
- **Cálculos**: Calcula o P&L total, o total de *funding* acumulado e as taxas pagas, considerando inclusive os descontos por uso de BNB.
- **Visualização**: Gera um gráfico da curva de capital ao final do backtest.

## 3. Principais Resultados Alcançados

O backtest executado em um longo período demonstrou que a estratégia é robusta e lucrativa, quase dobrando o capital inicial.

- **Período**: 2017-01-01 a 2025-10-05
- **Capital Inicial**: $1,000.00
- **Capital Final**: **$1,957.87**
- **P&L Total**: **$957.87 (95.79%)**

---

- **Número de Trades**: 19 (estratégia de baixa frequência)
- **Total de Funding Coletado**: $1,100.31
- **Total de Taxas Pagas**: $142.44
- **Resultado Líquido (Funding - Taxas)**: **$957.87**

**Conclusão Principal**: A estratégia se provou historicamente viável. Os ganhos com a coleta da taxa de financiamento superaram consistentemente os custos operacionais (taxas), gerando um lucro líquido positivo e de baixo risco ao longo do tempo.

## 4. Passo a Passo de Como Usar

Para executar o backtest, utilize o comando abaixo no terminal, ajustando os parâmetros conforme necessário.

```bash
poetry run python -m src.strategies.Arb_funding_basis.arb_funding_basis --start 2017-01-01 --end 2025-10-05 --capital 1000 --entry 0.0001 --exit 0.0
```

### Parâmetros Principais:

- `--symbol`: O par de negociação (ex: `BTCUSDT`). Default: `BTCUSDT`.
- `--start`: A data de início do backtest (formato: `YYYY-MM-DD`).
- `--end`: A data de fim do backtest (opcional, default é a data atual).
- `--capital`: O capital inicial em dólares.
- `--leverage`: A alavancagem a ser usada na posição de futuros.
- `--entry`: O *funding rate* mínimo para abrir uma posição (ex: `0.0001` para 0.01%).
- `--exit`: O *funding rate* que dispara o fechamento da posição (ex: `0.0` para 0%).
- `--no-bnb`: Adicione esta flag para simular o backtest sem os descontos de taxa do BNB.