# AGENTS.md

Guia e prompt-base para agentes (LLMs) que auxiliam na **pesquisa, teste e comparação** de estratégias de indicadores neste repositório.

> **Objetivo do agente:** automatizar o ciclo de: (1) preparar dados, (2) gerar/ajustar configurações de estratégia, (3) rodar backtests controlados, (4) registrar métricas e (5) produzir relatórios curtos e reprodutíveis.

---

## 1) Visão geral do projeto

**Estrutura relevante**

```
src/
  strategies/
    <nome_da_estrategia>/
    ...
  ...

data/
  klines_cache.db        # OHLCV histórico por símbolo/timeframe
  configs.db             # versões de configurações testadas
  optuna_studies.db      # estudos de otimização/hiperparâmetros
```

**Premissas**

* Ambiente Python gerenciado por **Poetry**.
* Dados de mercado obtidos da **Binance** (modo *paper/backtest*, nunca produção).
* Cada estratégia tem um módulo em `src/strategies/<nome>` com uma função de **backtest** acessível por CLI ou Python.

---

## 2) Ferramentas que o agente pode usar

> O agente deve preferir **comandos reprodutíveis**, **artefatos versionados** e **comandos limpos (sem flags)**. Todas as variações de parâmetros devem estar em arquivos JSON.

### 2.1 Regras gerais (obrigatórias)

- Comandos limpos: não passar flags de estratégia nos comandos; usar arquivos `.json` versionados.
- Passo 0 obrigatório: atualizar o cache antes de qualquer backtest.
- Consumo offline: backtests devem ler exclusivamente do `data/klines_cache.db`. Se faltar dado, falhar solicitando atualização do cache.
- Artefatos por estratégia: relatórios e gráficos em `src/strategies/<nome>/reports/`. Configurações ficam em `src/strategies/<nome>/config.json`.

### 2.2 Comandos de terminal

- Atualizar cache de dados (sempre primeiro):

  ```bash
  poetry run python -m scripts.populate_cache <SYMBOL> <TIMEFRAME>
  # Ex.: poetry run python -m scripts.populate_cache BTCUSDT 1m
  ```

  - Script oficial: `scripts/populate_cache.py`
  - Fallback (se o modo `-m` não estiver disponível):

    ```bash
    poetry run python scripts/populate_cache.py <SYMBOL> <TIMEFRAME>
    # Ex.: poetry run python scripts/populate_cache.py BTCUSDT 1m
    ```

- Backtest — al_brooks (comando limpo, sem flags):

  ```bash
  poetry run python -m src.strategies.al_brooks.backtest
  ```

  > O backtest lê a configuração de `src/strategies/al_brooks/config.json`. Parâmetros são versionados nesse arquivo.

- Otimização — al_brooks (sem flags):

  ```bash
  poetry run python -m src.strategies.al_brooks.optimize
  ```

  > Lê parâmetros do bloco `optimize` em `src/strategies/al_brooks/config.json` e salva a configuração.

- Sincronização de configuração: não é necessária. Edite `src/strategies/al_brooks/config.json` diretamente.

### 2.2 Acesso ao Python

* O agente pode propor *snippets* Python para:

  * gerar grades de parâmetros;
  * calcular métricas adicionais (Sharpe, MDD, Calmar);
  * exportar resultados para CSV/Markdown.

---

## 3) Protocolos e limites (segurança)

* **Nunca** executar ordens reais. Este repositório é **somente backtesting** e análise offline.
* **Determinismo**: sempre registrar seed, versões e janela temporal.
* **Offline por padrão**: backtests devem consumir dados do cache local. Se faltar dado, instruir a atualizar o cache.

---

## 4) Workflow padrão que o agente deve seguir

### Passo 1: Atualizar a base de dados (Passo 0 obrigatório)

Garanta que `data/klines_cache.db` esteja atualizado com os dados mais recentes da Binance.

```bash
poetry run python -m scripts.populate_cache BTCUSDT 1m
```

> Se múltiplos símbolos/timeframes forem usados, repetir para cada par/TF.

### Passo 2: Definir hipótese e configuração

* Redigir **hipótese** em 1–2 frases (ex.: “Cruzamento de médias com filtro RSI > 50 reduz *whipsaw* em tendência de alta”).
* Construir **config JSON** compatível com a estratégia escolhida (exemplo abaixo).

### Passo 3: Executar backtest controlado (sem flags)



### Passo 4: Coletar e validar métricas



### Passo 5: Persistir resultados

* Se houver suporte: salvar em `configs.db` e/ou `optuna_studies.db`.
* Exportar um **relatório Markdown** enxuto para `reports/<estrategia>_<symbol>_<tf>.md`.

### Passo 6: Concluir com recomendações

* Sumarizar em 3–5 bullets o que funcionou, falhou e próximos passos.

---

## 5) Esquemas de dados (para o agente)

### 5.1 Configuração — al_brooks (JSON)

Arquivo de configuração consumido pelo backtest: `src/strategies/al_brooks/config.json`

```json
{
  "ticker": "BTCUSDT",
  "interval": "1m",
  "days": 365,
  "lot_size": 0.1,
  "ema_fast_period": 10,
  "ema_medium_period": 20,
  "ema_slow_period": 50,
  "risk_reward_ratio": 2.0,
  "max_avg_deviation_pct": 0.5,
  "adx_period": 14,
  "adx_threshold": 22.0,
  "atr_period": 14,
  "atr_stop_multiplier": 1.5,
  "atr_trail_multiplier": 0.5,
  "htf_lookback": 20,
  "use_htf_bias": true,
  "min_trades_per_window": 15,
  "min_atr": 0.0,
  "taker_fee_pct": 0.0004,
  "slippage_pct": 0.0005
}
```

### 5.2 Estrutura mínima de resultados (JSON)

```json
{
  "strategy": "al_brooks",
  "symbol": "BTCUSDT",
  "interval": "1m",
  "period": {"start": "2023-01-01", "end": "2023-12-31"},
  "trades": 120,
  "total_pnl": 153.2,
  "win_rate": 0.46,
  "profit_factor": 1.32,
  "avg_win": 5.8,
  "avg_loss": 4.2,
  "chart_path": "src/strategies/al_brooks/reports/charts/al_brooks_backtest_BTCUSDT.png",
  "config_path": "src/strategies/al_brooks/config.json",
  "seed": 42,
  "run_env": {"python": "3.11", "poetry": "1.8.x", "lib_versions": {"pandas": "..."}}
}
```

---

## 6) Prompt-base para LLM (System Message)

> Cole isto como **System Prompt** quando for ativar um agente.

```
Você é um agente de backtesting de estratégias de trading **offline**. Siga estritamente as políticas:
1) Nunca execute ordens reais; apenas leia/gerencie dados locais e rode backtests.
2) Antes de qualquer backtest, GARANTA que o cache de dados esteja atualizado.
3) Seja reprodutível: logue comandos, seeds e intervalos de data.
4) Prefira comandos `poetry run ...` sem flags explícitas; as variações devem estar em arquivos .json versionados.
5) Produza no final: (a) tabela de métricas, (b) caminhos de artefatos, (c) próximos passos sucintos.

Ferramentas disponíveis (chamar em ordem quando necessário):
- Atualizar cache: `poetry run python -m scripts.populate_cache <SYMBOL> <TF>`
- Backtest al_brooks (limpo): `poetry run python -m src.strategies.al_brooks.backtest`
- Configuração: editar `src/strategies/al_brooks/config.json`
- Otimização sem flags: `poetry run python -m src.strategies.al_brooks.optimize`

Workflow obrigatório por tarefa:
A) Confirmar símbolos/TFs, janela de datas e estratégia.
B) Executar atualização do cache para cada (símbolo, TF).
C) Validar config JSON contra a estratégia escolhida.
D) Rodar backtest, salvar relatórios e métricas.
E) Gerar sumário final e recomendações.

Restrições:
- Limitar paralelismo a 5 execuções.
- Se trades < 30 ou dados < 2000 candles, sinalizar baixa significância.
```

---

## 7) Exemplos de uso (Few-shot)

### Exemplo 1 — Al Brooks Inside Bar (BTCUSDT 1m)

**Passo 0 — Atualizar cache:**

```bash
poetry run python -m scripts.populate_cache BTCUSDT 1m
```

**Passo 1 — Ajustar config (se necessário):** edite `src/strategies/al_brooks/config.json`.

**Passo 2 — Backtest (limpo):**

```bash
poetry run python -m src.strategies.al_brooks.backtest
```

**Saída esperada (resumo):** gráfico em `src/strategies/al_brooks/reports/charts/al_brooks_backtest_BTCUSDT.png` e métricas no console. (Opcional: consolidar em Markdown.)

### Exemplo 2 — Otimização simples (grade)

1. Gerar grade de `fast_ma` ∈ {5, 9, 12} × `slow_ma` ∈ {20, 26, 50}.
2. Rodar backtests sequenciais (≤5 em paralelo).
3. Salvar top‑3 por `sharpe` e `max_drawdown` < 25% em `configs.db`.

---

## 8) Checklist de PRONTO‑PARA‑RODAR

* [ ] `poetry install` concluído
* [ ] `.env`/chaves **não** exigidas para backtest offline
* [ ] `data/klines_cache.db` atualizado para símbolos/TFs desejados
* [ ] Config ativa presente em `src/strategies/<nome>/reports/active/ALBROOKS_<SYMBOL>_<TF>.json` (ou aceitar defaults)
* [ ] Estratégia escolhida possui função de backtest acessível
* [ ] Janela de datas definida (campo `days` na config ativa) e seed fixado (se aplicável)
* [ ] Pasta `src/strategies/<nome>/reports/` existe e é versionada (ou ignorada em `.gitignore` conforme política)

---

## 9) Próximos passos sugeridos

* Padronizar export de métricas em JSON + Markdown por backtest.
* Adicionar script `scripts/report.py` para consolidar múltiplos backtests em um único HTML/MD.
* Integrar com `optuna` (já há `optuna_studies.db`) para *Bayesian search* de hiperparâmetros.
