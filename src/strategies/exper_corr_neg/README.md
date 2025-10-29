# exper_corr_neg — Mixture of Experts (RL) para BTCUSDT 1h

Este experimento implementa um agente de Aprendizagem por Reforço com Mixture of Experts (MoE) para BTCUSDT (1h), seguindo o fluxo e as boas práticas descritas no `AGENTS.md`. A ideia de “correlação negativa” aqui não é entre tickers, e sim entre algoritmos/sinais: combinamos especialistas que tendem a performar bem em regimes opostos (tendência vs. reversão, direção vs. volatilidade, momentum de preço vs. reversão de volume) e deixamos um decodificador (gating) ponderar cada um dinamicamente.

## Decisões-chave
- Ativo/timeframe: BTCUSDT 1h, desde ~2017 (cache local `data/klines_cache.db`).
- Ambiente e ações: `{short, flat, long}` com capital lógico inicial de 1000, alavancagem configurável, stops móveis por ATR, trailing adicional por pico/vale de lucro e teto de notional por trade (1000 USD), penalidade de turnover e encerramento antecipado por piso de equity ou drawdown máximo. As janelas de treino podem iniciar em pontos aleatórios (`random_start`) para evitar viés de começo de série.
- MoE (PyTorch): 6 especialistas pequenos + gating com softmax (temperatura ajustável) e top‑k esparso. Regularização de balanceamento mantém o uso distribuído.
  - Trend — EMAs, Donchian, momentum
  - Mean‑Reversion — RSI, Bollinger, z‑score
  - Volatility — ATR, realized vol (range/rv)
  - Volume/Flow — OBV, Chaikin, spikes de volume
  - Squeeze/Breakout — largura de Bollinger, percentil de ATR
  - Pattern — forma de candles (corpo/pavio), padrões simples (doji, hammer, engulfing) e contagens rolling
- Treino RL: PPO adaptado ao MoE. O gating escolhe/pondera especialistas por passo; schedule de entropia reduz exploração ao longo dos episódios.
- Avaliação: Backtest, monitoramento contínuo (`metrics.csv/png`), visualização de ações (`visualize.py`) e análise do gating (`visualize_gating.py`). Walk‑Forward e Monte Carlo podem ser habilitados conforme necessidade.

## Estrutura do projeto
```
src/strategies/exper_corr_neg/
  __init__.py
  config.json                 # parâmetros do experimento (sem flags)
  features.py                 # cálculo de features para experts e contexto
  env.py                      # ambiente de trading (Gym‑like, com stub se gym não existir)
  models.py                   # experts, gating network e política MoE (PyTorch)
  trainer.py                  # PPO trainer adaptado ao MoE
  data.py                     # carregamento do BTCUSDT 1h e alinhamento de features
  train.py                    # treino contínuo (config‑driven)
  walk_forward.py             # walk‑forward (treino por janela + validação)
  reports/                    # artefatos (checkpoints, resumos, etc.)
```

## Dados e modo offline
- Leitura: sempre do cache local (`data/klines_cache.db`) via `utils.data_loader`.
- Atualização (opcional, com rede):
  ```bash
  poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"
  ```
- Execução offline (recomendado p/ reprodutibilidade): use a variável `BINANCE_OFFLINE=1` e um cache Numba local para acelerar funções do pandas_ta:
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache ...
  ```

## Configuração (sem flags)
Todos os parâmetros ficam em `src/strategies/exper_corr_neg/config.json`:
- `env`: custos, tamanho/alavancagem da posição (fixo ou dinâmico), multiplicadores de ATR (stop/trailing), penalidade de turnover, pisos de equity/drawdown, `accounting_mode` (`"mtm"` evita dupla contagem de PnL) e janela/aleatoriedade de início
  - Novos: `max_trade_notional` (teto em USD por trade; default 1000), `profit_trail_pct` (percentual para trailing por lucro sobre pico/vale; ex. 0.02 = 2%). O trailing por lucro complementa o stop/trailing por ATR, mantendo stop loss ativo.
  - `idle_penalty_factor`: fator (0 a 1) para aplicar penalidade automática quando o agente fica em cash. A cada candle flat, o ambiente debita `init_equity × fator / window_bars` do reward; com `factor = 1.0` e `window_bars = 8760`, isso equivale a ~US$ 0.11 por hora parado.
- `model`: camadas dos experts e do gating, número de experts, nomes didáticos, temperatura e top‑k
- `ppo`: hiperparâmetros do PPO (gamma, lambda, clip, lr, coeficientes etc.)
- `train`: episódios, passos por rollout, device, diretório de saída, schedule de entropia, espaçamento de logs/gráficos/avaliações, seed global (`seed`)
- `pbt`: parâmetros da população (tamanho, rounds, episódios por round, paralelismo, threads, checkpoint inicial e se o campeão deve ser promovido automaticamente para `moe_policy_final.pt`)
- `walk_forward`: agenda (dias de treino/val/step), episódios por janela, device, diretório
- `docs`: descrições em português para cada parâmetro (apenas leitura humana; o código ignora este bloco)

Edite o JSON e rode os comandos “limpos” abaixo — não há necessidade de flags.

## Comandos
- Treino contínuo
  ```bash
  BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
    poetry run python -m src.strategies.exper_corr_neg.train
  ```
  Artefatos: `src/strategies/exper_corr_neg/reports/train/`

- Atualização automática de relatórios durante o treino
  - A cada `train.plot_every` episódios o treino atualiza, em sincronia:
    - `metrics.csv` e `metrics.png` (losses, entropy/balance, rewards, greedy equity)
    - `expert_usage.png` (média dos últimos `train.usage_window` episódios)
  - Se `train.usage_window` não estiver definido, ele usa o mesmo valor de `train.plot_every` por padrão.
  - Sugestão simples: defina `plot_every` = `usage_window` (ex.: 25) no `config.json` para manter tudo sincronizado.

- Visualização do backtest (preço + ações + equity)
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_neg.visualize
  ```
  Gera `src/strategies/exper_corr_neg/reports/train/visual_backtest.png` com legenda dos experts.

- Análise do gating (pesos/top‑k, drawdown/ruína)
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_neg.visualize_gating
  ```
  Artefatos: `gating_trace.csv`, `gating_heatmap.png` (com marcas de drawdown/ruína) e `gating_usage.png`.

- (Opcional) Regerar gráficos sem treinar
  ```bash
  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_neg.make_reports
  ```
  Lê o `metrics.csv` atual e atualiza `metrics.png` e `expert_usage.png` respeitando `train.plot_every` e `train.usage_window`.

## Treino em População (PBT‑lite)
- Ideia: rodar 2–3 treinamentos em paralelo (seeds/hipers levemente mutados), medir `greedy_equity` e promover um campeão por rodada. Os demais retomam do campeão no próximo round.
- Comando base (2 runs × 3 rounds × 400 episódios; WSL/CPU):
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_neg.scripts.pop_runner \
      --base src/strategies/exper_corr_neg/config.json \
      --pop 2 --rounds 3 --episodes 400 --concurrency 2
  ```
- Para começar a população do seu “cérebro” atual (resume do checkpoint):
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_neg.scripts.pop_runner \
      --base src/strategies/exper_corr_neg/config.json \
      --pop 2 --rounds 3 --episodes 400 --concurrency 2 \
      --seed_checkpoint src/strategies/exper_corr_neg/reports/train/moe_policy_final.pt \
      --promote_to_root
  ```
- Com os parâmetros definidos em `pbt` dentro do `config.json`, dá para rodar sem flags adicionais:
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_neg.scripts.pop_runner \
      --base src/strategies/exper_corr_neg/config.json
  ```
- Se preferir, você ainda pode sobrescrever valores pela linha de comando (ex.: `--rounds 5`).
- Com `--promote_to_root` (ou `"promote_to_root": true` em `pbt`), o checkpoint do campeão de cada round é copiado automaticamente para `src/strategies/exper_corr_neg/reports/train/moe_policy_final.pt`, permitindo retomar o treino padrão diretamente do melhor modelo da população.
- Para orquestrar tudo automaticamente (rodar PBT, promover campeão e continuar o treino principal), use o script abaixo:
  ```bash
  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_neg.scripts.auto_cycle \
      --base src/strategies/exper_corr_neg/config.json
  ```
- O `auto_cycle` roda o `pop_runner` com os parâmetros definidos em `pbt`, verifica se o `scoreboard` ganhou um novo campeão, e — caso sim — promove o checkpoint (respeitando `promote_to_root`) e dispara `train.py` para continuar o “cérebro” oficial. Use `--skip-train` se quiser apenas promover o campeão e rodar o treino principal depois.
- Toda execução de `train.py` registra um manifesto em `src/strategies/exper_corr_neg/reports/train/run_manifest.json` com hash do config, seed e parâmetros de avaliação/risco (quando disponível, também o commit Git). Use esse manifesto como diário das séries.
- Artefatos da população:
  - `src/strategies/exper_corr_neg/reports/train/pop/run_{i}/round_{r}/` — outdirs por run e round
  - `src/strategies/exper_corr_neg/reports/train/pop/configs/` — configs geradas por round
  - `src/strategies/exper_corr_neg/reports/train/pop/scoreboard.json` — histórico de campeões

Observações
- Limite `concurrency` e as variáveis de threads para evitar competição de CPU (2 é seguro nesta máquina).
- O runner promove um campeão apenas quando a avaliação “sem ruína” bate o melhor `greedy_equity` do round.

## Limpeza de `reports/train`
- Script utilitário para enxugar a pasta de treino e manter somente o essencial (modelos finais/recordistas, últimas `epXX`, métricas e gráficos):
  ```bash
  poetry run python -m src.strategies.exper_corr_neg.scripts.clean_train_reports \
    --dir src/strategies/exper_corr_neg/reports/train \
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
    poetry run python -m src.strategies.exper_corr_neg.walk_forward
  ```
  Artefatos: `src/strategies/exper_corr_neg/reports/walk_forward/`

- Limpeza de checkpoints intermediários (`moe_policy_ep*.pt`)
  ```bash
  # visualiza o que seria removido (sem apagar nada)
  poetry run python -m src.strategies.exper_corr_neg.scripts.prune_checkpoints --dry-run

  # remove checkpoints antigos mantendo, por padrão, os 3 mais recentes em cada pasta
  poetry run python -m src.strategies.exper_corr_neg.scripts.prune_checkpoints

  # mantém somente 1 checkpoint em toda a árvore de train/ e train/pop/
  poetry run python -m src.strategies.exper_corr_neg.scripts.prune_checkpoints --keep 1
  ```
  - Caso queira limpar apenas um diretório específico:
    ```bash
    poetry run python -m src.strategies.exper_corr_neg.scripts.prune_checkpoints \
      --dir src/strategies/exper_corr_neg/reports/train/pop/run_0 \
      --keep 2
    ```

## Reinício completo (do zero)
Use este fluxo quando quiser reiniciar totalmente o experimento.

1. Limpe os artefatos anteriores:
   ```bash
   poetry run python -m src.strategies.exper_corr_neg.scripts.reset_reports --force
   ```
   (remova `--force` se preferir confirmar manualmente).
2. Garanta que o cache tenha os dados necessários (apenas quando precisar reatualizar):
   ```bash
   poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"
   ```
3. Exporte o modo offline e rode o treino principal:
   ```bash
   BINANCE_OFFLINE=1 NUMBA_CACHE_DIR=$PWD/.numba_cache \
     poetry run python -m src.strategies.exper_corr_neg.train
   ```
4. Gere rapidamente métricas e gráficos:
   ```bash
   poetry run python -m src.strategies.exper_corr_neg.make_reports
   BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_neg.visualize
   BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_neg.visualize_gating
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
