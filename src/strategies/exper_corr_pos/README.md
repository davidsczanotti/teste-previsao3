# exper_corr_pos — Mixture of Experts (RL) para BTCUSDT 1d

Este experimento implementa um agente de Aprendizagem por Reforço com Mixture of Experts (MoE) para BTCUSDT (1d), seguindo o fluxo e as boas práticas descritas no `AGENTS.md`. A proposta aqui é explorar **correlação positiva** entre sinais: o agente só entra forte quando múltiplos especialistas concordam (técnico + modelo de direção, multiframe, cointegração de pares e padrões de candle), privilegiando precisão em detrimento de volume de trades.

## Decisões-chave
- Ativo/timeframe: BTCUSDT 1d, desde ~2017 (cache local `data/klines_cache.db`).
- Ambiente e ações: `{short, flat, long}` com capital lógico inicial de 1000, alavancagem configurável, stops móveis por ATR, trailing adicional por pico/vale de lucro e teto de notional por trade (1000 USD), penalidade de turnover e encerramento antecipado por piso de equity ou drawdown máximo. As janelas de treino podem iniciar em pontos aleatórios (`random_start`) para evitar viés de começo de série.
- MoE (PyTorch): especialistas habilitados via `model.especialistas` + gating com softmax (temperatura ajustável) e top‑k esparso. Um regularizador de balanceamento mantém o uso distribuído.
  - **TrendML** — EMAs, ratio/accel, Donchian width, volatilidades realizadas (14/30), matriz de lags, momentum, slopes e preditor direcional online (pseudo-LightGBM) com features horárias.
  - **MultiFrame** — concordância entre tendência semanal (1W) e gatilhos diários (1d) com pullback/RSI, ATR relativo e volatilidade do timeframe superior.
  - (opcionais) **Spread** e **Pattern** — spread beta/z-score, correlações roll com ETH, z-score do spread, padrões de candle enriquecidos (gaps, wick imbalance, métricas normalizadas por ATR).
- Treino RL: PPO adaptado ao MoE com **curriculum learning** (`train.curriculum`) — primeiras fases sem `random_start`, janelas maiores e rollouts curtos; depois libera regime completo. O gating escolhe/pondera especialistas por passo; schedule de entropia reduz exploração ao longo dos episódios.
- Logging: CSVs e gráficos locais + integração opcional com Weights & Biases (configurar `logging.wandb`) para acompanhar métricas, checkpoints e artefatos em tempo real.
- Avaliação: Backtest, monitoramento contínuo (`metrics.csv/png`), visualização de ações (`visualize.py`) e análise do gating (`visualize_gating.py`). O Walk‑Forward inclui Monte Carlo (ruído em preço/features), stress de custos (±50%), lags (1–5) e análise de regimes (vol baixa/média/alta).
- Otimização: bloco `optimize` no `config.json` usa Optuna para varrer `ppo.learning_rate`, `ppo.gamma`, `model.top_k` e entropia inicial, salvando resultados em `reports/optuna/`.

Observação de avaliação: a avaliação greedy durante o treino é determinística (sem `random_start` e usando a janela fixa do fim dos dados). Isso deixa o `moe_policy_best_eval.pt` reprodutível e coerente com a auditoria.

## Estrutura do projeto
```
src/strategies/exper_corr_pos/
  __init__.py
  config.json                 # parâmetros do experimento (sem flags)
  features.py                 # cálculo de features para experts e contexto
  env.py                      # ambiente de trading (Gym‑like, com stub se gym não existir)
  models.py                   # experts, gating network e política MoE (PyTorch)
  trainer.py                  # PPO trainer adaptado ao MoE
  data.py                     # carregamento do BTCUSDT 1d e alinhamento de features
  train.py                    # treino contínuo (config‑driven)
  walk_forward.py             # walk‑forward (treino por janela + validação)
  reports/                    # artefatos (checkpoints, resumos, etc.)
```

## Relatórios e auditoria
- Os relatórios adicionais são controlados por `config.json` (bloco `reports`) e gerados pelo auditor:
  - `reports.trade_ledger.enabled/path`
  - `reports.gating_attribution.enabled/path/plot_path`
  - `reports.regime_summary.enabled/path`

### Auditoria (sem flags específicos de estratégia)
```bash
BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.scripts.audit_policy
```
Artefatos (por padrão):
- `src/strategies/exper_corr_pos/reports/train/trade_ledger.csv`
  - colunas principais: `entry_ts`, `exit_ts`, `side`, `size`, `entry_price`, `exit_price`, `duration_bars`, `duration_hours`, `pnl_net`, `pnl_gross`, `cost`, `bonus`, `reason` (ex.: `trail_atr`, `trail_profit`, `flip`), além de pesos do gate por expert no momento de entrada (`entry_weight_*`) e média no trade (`avg_weight_*`).
- `src/strategies/exper_corr_pos/reports/train/gating_attribution.csv` e `gating_attribution.png`
  - visão de contribuição média dos experts em `win/loss/flat`.
- `src/strategies/exper_corr_pos/reports/train/regime_summary.json`
  - métricas agregadas por regime: `htf_trend_state` (−1/0/+1), `vol_bucket` (low/medium/high) e combinações.

O auditor lê `visualize.days` para a janela (padrão 90) e força avaliação determinística (sem `random_start`).

## Dados e modo offline
- Leitura: sempre do cache local (`data/klines_cache.db`) via `utils.data_loader`.
- Atualização (opcional, com rede): parâmetros vem do `config.json` (`data.*`). Exemplos:
  ```bash
  # apenas o par base (usa data.base_symbol/timeframe/start/lookback_days)
  poetry run python -m scripts.populate_cache BTCUSDT

  # popular todos os símbolos declarados (base, confirm e extras)
  poetry run python -m scripts.populate_cache

  # sobrescrever intervalo ou datas
  poetry run python -m scripts.populate_cache BTCUSDT 4h --start "2020-01-01 00:00:00"
  ```
- Execução offline (recomendado p/ reprodutibilidade): use a variável `BINANCE_OFFLINE=1` e um cache Numba local para acelerar funções do pandas_ta:
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache ...
  ```

## Configuração (sem flags)
Todos os parâmetros ficam em `src/strategies/exper_corr_pos/config.json`:
- `data`: símbolo principal (`base_symbol`), par de confirmação (`confirm_symbol`), timeframe base e superior, horizonte do preditor direcional e janela usada para o spread BTC×ETH.
- `env`: custos, tamanho/alavancagem da posição (fixo ou dinâmico), multiplicadores de ATR (stop/trailing), penalidade de turnover, pisos de equity/drawdown, `accounting_mode` (`"mtm"` evita dupla contagem de PnL) e janela/aleatoriedade de início
  - Novos: `max_trade_notional` (teto em USD por trade; default 1000), `profit_trail_pct` (percentual para trailing por lucro sobre pico/vale). O trailing por lucro complementa o stop/trailing por ATR, mantendo stop loss ativo.
  - `idle_penalty_factor`: fator (0 a 1) para aplicar penalidade automática quando o agente fica em cash. A cada candle flat, o ambiente debita `init_equity × fator / window_bars` do reward; com `factor = 1.0` e `window_bars = 365`, isso equivale a ~US$ 2.74 por dia parado.
  - `hold_bonus_alpha`: fator da “gorjeta” ao fechar uma posição. O bônus (ou malus) adicionado ao reward é `alpha × duração × PnL`, reforçando lucros longos e penalizando prejuízos prolongados.
- `model`: camadas dos experts e do gating, número de experts, nomes didáticos, temperatura e top‑k
- `ppo`: hiperparâmetros do PPO (gamma, lambda, clip, lr, coeficientes etc.)
- `train`: episódios, passos por rollout, device, diretório de saída, schedule de entropia, espaçamento de logs/gráficos/avaliações, seed global (`seed`)
- `logging`: integrações opcionais de monitoramento (ex.: `wandb` com `project`, `entity`, `tags`, `watch`, `artifact_prefix`)
- `pbt`: parâmetros da população (tamanho, rounds, episódios por round, paralelismo, threads, checkpoint inicial e se o campeão deve ser promovido automaticamente para `moe_policy_final.pt`)
- `walk_forward`: agenda (dias de treino/val/step), episódios por janela, device, diretório
- `docs`: descrições em português para cada parâmetro (apenas leitura humana; o código ignora este bloco)

Edite o JSON e rode os comandos “limpos” abaixo — não há necessidade de flags.

## Comandos
- Treino contínuo
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_pos.train
  ```
  Artefatos: `src/strategies/exper_corr_pos/reports/train/`
  - Para enviar métricas/artefatos ao Weights & Biases, edite `logging.wandb.enabled` para `true` no `config.json` (configure também `project/entity/name` conforme sua conta) antes de rodar o treino.

## Fluxo (Botões do Fliperama)

- Start (começar do zero)
  1) Limpar artefatos anteriores (treino/optuna/WF)
     ```bash
     poetry run python -m src.strategies.exper_corr_pos.scripts.reset_reports --force
     ```
  2) Atualizar cache local (base e confirm do config)
     ```bash
     poetry run python -m scripts.populate_cache BTCUSDT 1d --start "2017-01-01 00:00:00"
     poetry run python -m scripts.populate_cache ETHUSDT 1d --start "2017-01-01 00:00:00"
     ```
  3) Otimizar hiperparâmetros (aplica vencedor ao `config.json` automaticamente; gera backup)
     ```bash
     BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
       poetry run python -m src.strategies.exper_corr_pos.optimize
     ```
     Saídas: `src/strategies/exper_corr_pos/reports/optuna/<timestamp>/{summary.json,summary.md,trials.csv}` + `config_backup_<timestamp>.json`.
  4) Treinar com os melhores parâmetros
     ```bash
     BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
       poetry run python -m src.strategies.exper_corr_pos.train
     ```

- Continue (retomar do último agente)
  - Verifique no `config.json`:
    - `train.resume: true`
    - `train.resume_path: "src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt"`
  - Rodar
    ```bash
    BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
      poetry run python -m src.strategies.exper_corr_pos.train
    ```

- Relatórios e Status (sem re‑treinar)
  - Consolidar métricas/gráficos do treino atual
    ```bash
    BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.make_reports
    ```
  - Visualizar backtest (preço + ações + equity)
    ```bash
    BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.visualize
    ```
  - Análise do gating (pesos/top‑k, drawdown/ruína)
    ```bash
    BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.visualize_gating
    ```
  - Onde olhar: `src/strategies/exper_corr_pos/reports/train/{metrics.csv,metrics.png,expert_usage.png,gating_heatmap.png}`

- Walk‑Forward (robusto)
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_pos.walk_forward
  ```
  Saídas: `wf_summary.json` + `wf_summary.md` em `src/strategies/exper_corr_pos/reports/walk_forward/` (inclui Monte Carlo, stress de custos/lag e regimes).

- Stress Tests (rápidos no modelo final)
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_pos.scripts.stress_tests
  ```
  Saídas: `src/strategies/exper_corr_pos/reports/train/stress_tests.json|md`.

- Tests (unitários)
  ```bash
  poetry run pytest -q
  ```
  Cobertura inicial: `tests/test_models.py` (gating/política) e `tests/test_env.py` (dinâmica e ruína do ambiente).

- Atualização automática de relatórios durante o treino
  - A cada `train.plot_every` episódios o treino atualiza, em sincronia:
    - `metrics.csv` e `metrics.png` (losses, entropy/balance, rewards, greedy equity)
    - `expert_usage.png` (média dos últimos `train.usage_window` episódios)
  - Se `train.usage_window` não estiver definido, ele usa o mesmo valor de `train.plot_every` por padrão.
  - Sugestão simples: defina `plot_every` = `usage_window` (ex.: 25) no `config.json` para manter tudo sincronizado.

- Visualização do backtest (preço + ações + equity)
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.visualize
  ```
  Gera `src/strategies/exper_corr_pos/reports/train/visual_backtest.png` com legenda dos experts.

- Análise do gating (pesos/top‑k, drawdown/ruína)
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.visualize_gating
  ```
  Artefatos: `gating_trace.csv`, `gating_heatmap.png` (com marcas de drawdown/ruína) e `gating_usage.png`.

- (Opcional) Regerar gráficos sem treinar
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.make_reports
  ```
  Lê o `metrics.csv` atual e atualiza `metrics.png` e `expert_usage.png` respeitando `train.plot_every` e `train.usage_window`.

- Otimização de hiperparâmetros (Optuna)
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_pos.optimize
  ```
  Usa o bloco `optimize` do `config.json` para varrer `ppo.learning_rate`, `ppo.gamma`, `model.top_k` e `train.entropy_coef_start`. Resultados de cada trial ficam em `src/strategies/exper_corr_pos/reports/optuna/<timestamp>/trial_XXXX/`, e o resumo global (`summary.json`, `best_config.json`, `trials.csv`, `summary.md`) é salvo na mesma pasta. Ao final, o script aplica automaticamente os hiperparâmetros vencedores no `config.json` (backup `config_backup_<timestamp>.json`) e roda, por padrão, um baseline configurado em `optimize.baseline` para comparação direta.

## Treino em População (PBT‑lite)
- Ideia: rodar 2–3 treinamentos em paralelo (seeds/hipers levemente mutados), medir `greedy_equity` e promover um campeão por rodada. Os demais retomam do campeão no próximo round.
- Comando base (2 runs × 3 rounds × 400 episódios; WSL/CPU):
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_pos.scripts.pop_runner \
      --base src/strategies/exper_corr_pos/config.json \
      --pop 2 --rounds 3 --episodes 400 --concurrency 2
  ```
- Para começar a população do seu “cérebro” atual (resume do checkpoint):
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_pos.scripts.pop_runner \
      --base src/strategies/exper_corr_pos/config.json \
      --pop 2 --rounds 3 --episodes 400 --concurrency 2 \
      --seed_checkpoint src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt \
      --promote_to_root
  ```
- Com os parâmetros definidos em `pbt` dentro do `config.json`, dá para rodar sem flags adicionais:
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_pos.scripts.pop_runner \
      --base src/strategies/exper_corr_pos/config.json
  ```
- Se preferir, você ainda pode sobrescrever valores pela linha de comando (ex.: `--rounds 5`).
- Com `--promote_to_root` (ou `"promote_to_root": true` em `pbt`), o checkpoint do campeão de cada round é copiado automaticamente para `src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt`, permitindo retomar o treino padrão diretamente do melhor modelo da população.
- Para orquestrar tudo automaticamente (rodar PBT, promover campeão e continuar o treino principal), use o script abaixo:
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_pos.scripts.auto_cycle \
      --base src/strategies/exper_corr_pos/config.json
  ```
- O `auto_cycle` roda o `pop_runner` com os parâmetros definidos em `pbt`, verifica se o `scoreboard` ganhou um novo campeão, e — caso sim — promove o checkpoint (respeitando `promote_to_root`) e dispara `train.py` para continuar o “cérebro” oficial. Use `--skip-train` se quiser apenas promover o campeão e rodar o treino principal depois.
- Toda execução de `train.py` registra um manifesto em `src/strategies/exper_corr_pos/reports/train/run_manifest.json` com hash do config, seed e parâmetros de avaliação/risco (quando disponível, também o commit Git). Use esse manifesto como diário das séries.
- Artefatos da população:
  - `src/strategies/exper_corr_pos/reports/train/pop/run_{i}/round_{r}/` — outdirs por run e round
  - `src/strategies/exper_corr_pos/reports/train/pop/configs/` — configs geradas por round
  - `src/strategies/exper_corr_pos/reports/train/pop/scoreboard.json` — histórico de campeões

Observações
- Limite `concurrency` e as variáveis de threads para evitar competição de CPU (2 é seguro nesta máquina).
- O runner promove um campeão apenas quando a avaliação “sem ruína” bate o melhor `greedy_equity` do round.

## Limpeza de `reports/train`
- Script utilitário para enxugar a pasta de treino e manter somente o essencial (modelos finais/recordistas, últimas `epXX`, métricas e gráficos):
  ```bash
  poetry run python -m src.strategies.exper_corr_pos.scripts.clean_train_reports \
    --dir src/strategies/exper_corr_pos/reports/train \
    --keep-ep 3
  ```
- Mantém por padrão:
  - `moe_policy_best_eval.pt` (se houver)
  - `moe_policy_final.pt` (alias atualizado por `final_every`)
  - as últimas `N` (`--keep-ep`) `moe_policy_ep*.pt`
  - `metrics.csv`, `metrics.png`, `expert_usage.png`
  - `gating_heatmap.png`, `gating_usage.png` (se existirem)
- Use `--dry-run` para listar o que seria removido sem deletar.

- Walk‑Forward (treina por janela e valida OOS)
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_pos.walk_forward
  ```
  Artefatos: `src/strategies/exper_corr_pos/reports/walk_forward/` (`wf_summary.json` + `wf_summary.md`, checkpoints por janela e stats de Monte Carlo/custos/lag/regimes).

- Limpeza de checkpoints intermediários (`moe_policy_ep*.pt`)
  ```bash
  # visualiza o que seria removido (sem apagar nada)
  poetry run python -m src.strategies.exper_corr_pos.scripts.prune_checkpoints --dry-run

  # remove checkpoints antigos mantendo, por padrão, os 3 mais recentes em cada pasta
  poetry run python -m src.strategies.exper_corr_pos.scripts.prune_checkpoints

  # mantém somente 1 checkpoint em toda a árvore de train/ e train/pop/
  poetry run python -m src.strategies.exper_corr_pos.scripts.prune_checkpoints --keep 1
  ```
  - Caso queira limpar apenas um diretório específico:
    ```bash
    poetry run python -m src.strategies.exper_corr_pos.scripts.prune_checkpoints \
      --dir src/strategies/exper_corr_pos/reports/train/pop/run_0 \
      --keep 2
    ```

- Arquivar configurações e campeões (rastreabilidade)
  - Automático (recomendado):
    ```bash
    poetry run python -m src.strategies.exper_corr_pos.scripts.archive_run --include-checkpoint
    ```
    - Gera uma pasta em `src/strategies/exper_corr_pos/reports/runs/<prefix>_<timestamp>` contendo:
      * `config.json` usado no run
      * `reports/train/pop/scoreboard.json`
      * (opcional) `reports/train/moe_policy_final.pt` quando `--include-checkpoint` for usado
    - Customize o prefixo/destino:
      ```bash
      poetry run python -m src.strategies.exper_corr_pos.scripts.archive_run \
        --prefix campeao --dest /tmp/meu_run --include-checkpoint
      ```
  - Fluxo manual (caso precise arquivar em outro formato):
  1. Crie uma pasta para registrar o run:
     ```bash
     mkdir -p src/strategies/exper_corr_pos/reports/runs/$(date +%Y%m%d_%H%M)
     ```
  2. Copie o `config.json` e o `scoreboard.json` do PBT:
     ```bash
     run_dir=src/strategies/exper_corr_pos/reports/runs/$(date +%Y%m%d_%H%M)
     mkdir -p "$run_dir"
     cp src/strategies/exper_corr_pos/config.json "$run_dir/config.json"
     cp src/strategies/exper_corr_pos/reports/train/pop/scoreboard.json "$run_dir/scoreboard.json"
     ```
  3. Opcional: copie o checkpoint do campeão promovido (já está em `reports/train/moe_policy_final.pt`) e outros artefatos relevantes:
     ```bash
     cp src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt "$run_dir/moe_policy_final.pt"
     ```
  Assim cada run fica registrado com os parâmetros usados e o histórico de campeões.

## Reinício completo (do zero)
Use este fluxo quando quiser reiniciar totalmente o experimento.

1. Limpe os artefatos anteriores:
   ```bash
   poetry run python -m src.strategies.exper_corr_pos.scripts.reset_reports --force
   ```
   (remova `--force` se preferir confirmar manualmente).
2. Garanta que o cache tenha os dados necessários (apenas quando precisar reatualizar):
   ```bash
   poetry run python -m scripts.populate_cache BTCUSDT
   ```
3. Exporte o modo offline e rode o treino principal:
   ```bash
   BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
     poetry run python -m src.strategies.exper_corr_pos.train
   ```
4. Gere rapidamente métricas e gráficos:
   ```bash
   poetry run python -m src.strategies.exper_corr_pos.make_reports
   BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.visualize
   BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.visualize_gating
   ```
5. (Opcional) Execute a população (`scripts.auto_cycle`) e o walk-forward para validar OOS.

## Avaliação e Relatórios (AGENTS.md)
- Backtest com custos e stop móvel; export de trades (quando aplicável).
- Monte Carlo: perturbação por blocos na sequência de trades para distribuição de P&L/PF/MDD.
- Walk‑Forward: adaptativo (reotimização a cada janela) e/ou fixo (parâmetros ancorados) — comparar consistência.
- Relatório Markdown consolidando: MDD/Calmar/Ulcer, Sharpe/Sortino rolantes, tail risk, turnover, stress de custos/lag/perturbação de features, splits por regime (baixa/alta vol) e gráficos (equity+MDD/UI, Sharpe rolante, histogramas, heatmaps por hora, etc.).

### Checklist de robustez (resumo)
- Risco: MDD absoluto/relativo, Ulcer Index, Calmar/Mar, Time Under Water
- Qualidade: Sharpe/Sortino (rolantes), skew/kurtosis, PF, hit rate e payoff ratio
- Execução: turnover, cost/alpha, slippage efetivo, duração média e trades/dia
- Estabilidade: correlações (benchmark e entre sinais), Information Ratio por regime, estabilidade de Sharpe em janelas
- Calibração (se probabilístico): Brier/ACE, reliability plot, Q‑Q
- Testes: custos ±2–3×, lag (1 bar), partial fills, price perturbation, feature noise/dropout, grid pequena de hypers, bootstrap/permutation, Reality Check simples
- Gate prático: Sharpe OOS ≥ 1, MDD ≤ 15–20%, UI ≤ 5–8, PF ≥ 1.3, ≥ 60% das janelas OOS positivas, sensibilidade a custos < 30%

## Observações
- O ambiente funciona sem a dependência `gym` (há um stub leve), mas se quiser os wrappers e compatibilidade plena, instale `gym`.
- Para execuções longas, considere GPU (`device: "cuda"`) e ajuste `episodes`/`rollout_steps` paulatinamente.
- Monitorar risco: `metrics.csv/png` mostra entropia, drawdown (via ruína), equity greedy e uso dos experts. Acompanhe também `expert_usage.png` e `gating_heatmap.png`.
- Para expandir o MoE, basta acrescentar novas features e atualizar `model.num_experts`/`model.expert_names`; o gating decidirá quando usar cada especialista.

Qualquer alteração no comportamento deve ser feita editando o `config.json`. Os scripts já consomem esse arquivo e salvam os resultados dentro da pasta da estratégia para manter tudo organizado e reprodutível.


BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.scripts.audit_policy



////////////
Fluxo Com Otimização (recomendado agora)

Passo 0 — Atualizar cache (mesmo de cima)
poetry run python -m scripts.populate_cache 

Otimização (Optuna, curriculum curto, baseline e logs periódicos)
BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache poetry run python -m src.strategies.exper_corr_pos.optimize

O script:
roda baseline (controlado por optimize.baseline no src/strategies/exper_corr_pos/config.json);
executa trials com logs por episódio (intervalo vem de optimize.log_every ou episodes/20);
salva summary.md/json, trials.csv e pastas trial_XXXX/ em src/strategies/exper_corr_pos/reports/optuna/<timestamp>;
aplica automaticamente o vencedor no src/strategies/exper_corr_pos/config.json e cria backup config_backup_<timestamp>.json.
Treino final (já com os melhores hypers aplicados)

BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache poetry run python -m src.strategies.exper_corr_pos.train
Onde ver e como “conversar” sobre o resultado

Otimização:
Scoreboard e comparação com baseline no final do terminal.
Relatórios: src/strategies/exper_corr_pos/reports/optuna/<timestamp>/summary.md e summary.json.
Treino:
Métricas/uso: src/strategies/exper_corr_pos/reports/train/metrics.csv|png, expert_usage.png.
Gating/pesos: src/strategies/exper_corr_pos/reports/train/gating_heatmap.png, gating_usage.png.
- Stress tests rápidos (custo/lag/Monte Carlo sobre o modelo final)
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_pos.scripts.stress_tests
  ```
  Salva `stress_tests.json|md` em `src/strategies/exper_corr_pos/reports/train/` com baseline vs. sensibilidades.

- Testes unitários (pytest)
  ```bash
  poetry run pytest -q
  ```
  Cobertura inicial em `tests/test_models.py` (gating e política) e `tests/test_env.py` (dinâmica básica do ambiente).


*********************************************
Se a sua intenção de “Start” for treino do zero mantendo histórico de estudos (recomendado), já está pronto para o próximo passo (populate cache → optimize → train).

Se você quer um “zero absoluto” (sem histórico nem reuso de estudo):

Opcional: remover resultados antigos do Optuna
rm -rf src/strategies/exper_corr_pos/reports/optuna
Opcional: zerar o banco de estudos
rm -f data/optuna_studies.db (ou altere optimize.study_name no config.json para criar um estudo novo)
**********************************************