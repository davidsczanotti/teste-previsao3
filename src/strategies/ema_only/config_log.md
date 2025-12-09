ema_only — Config Log
=====================

Objetivo: registrar, de forma resumida, **todas as alterações relevantes** feitas em
`src/strategies/ema_only/config.json`, junto com o motivo e os artefatos de treino
relacionados. Assim conseguimos comparar facilmente se uma mudança ajudou ou piorou.

Formato sugerido por entrada:

- Data / ID lógico
- Hash do config (`sha256`) para rastreabilidade
- Motivo da alteração (1–3 frases)
- Parâmetros alterados (chave e novo valor; quando possível, antes→depois)
- Artefatos associados (modelo, métricas, gráficos)

---

## 2025-12-02 — ema_only_rl_v1_4h_intraday

- **config_sha256**: `442691dac3885b81318a7435d4e867356f4925600880420ba8d819ea2d5a77fe`
- **Motivo**: migrar o baseline EMA-only para um agente RL em candles de 4h, adicionando
  um especialista intraday de 1h e endurecendo o consenso / controle de churn para
  um perfil mais “swing” e menos scalper.
- **Parâmetros alterados (estado atual do config)**:
  - `data.timeframe`: `"4h"` (antes: baseline rule-based era `"1h"`, conforme README).
  - `data.intraday_timeframe`: `"1h"` — série intraday usada pelo novo expert.
  - `data.intraday_window_hours`: `12` — janela de 12 candles 1h para medir alinhamento.
  - `data.intraday_min_alignment`: `0.75` — exige ≥75% dos candles 1h alinhados com a
    tendência 4h para o expert intraday aprovar.
  - Bloco `rl.train` definido para PPO (SB3) em `4h`:
    - `total_timesteps`: `800000`, `n_steps`: `256`, `batch_size`: `256`,
      `learning_rate`: `0.0003`, `gamma`: `0.99`, `n_epochs`: `5`.
  - Bloco `rl.reward` ajustado para reduzir churn e filtrar regimes ruins:
    - `trade_penalty`: `0.01`
    - `dd_threshold_pct`: `0.02`, `dd_penalty`: `0.02`
    - `min_hold_bars`: `5`
    - `churn_penalty`: `0.01`
    - `align_bonus`: `0.001`
    - `atr_risk_scale`: `0.5`
    - `enforce_ref_bias`: `true`
    - `reward_scale`: `1.0`
    - `consensus_bonus`: `0.003`
    - `consensus_threshold`: `0.8`
    - `vol_max_atr_rel`: `0.015`
    - `vol_penalty`: `0.02`
    - `gating_penalty`: `0.02`
    - `turnover_penalty`: `0.01`
- **Lógica nova de especialista intraday (1h)**:
  - Em `rl_train.py`, para cada candle 4h calculamos um `intraday_align_ratio`:
    fração, numa janela de `intraday_window_hours` candles 1h, em que a tendência
    1h (`ema_fast_1h > ema_slow_1h`) está alinhada com a tendência 4h
    (`ema_fast_4h > ema_slow_4h`).
  - `exp_intraday_trend = 1` se `intraday_align_ratio >= intraday_min_alignment`,
    caso contrário `0`.
  - `experts_mean` passa a ser a média de 5 experts:
    `exp_trend` (4h), `exp_ref` (ref 1d), `exp_macd`, `exp_slope`, `exp_intraday_trend`.
- **Artefatos de treino associados** (estado atual):
  - Modelo PPO: `src/strategies/ema_only/reports/rl/ppo_ema_only.zip`
  - Métricas: `src/strategies/ema_only/reports/rl/metrics.csv`
  - Gráficos:
    - `src/strategies/ema_only/reports/rl/metrics.png` (reward, PnL, trades ao longo do treino)
    - `src/strategies/ema_only/reports/rl/actions.png` (preço/EMAs + ações do agente em 2025)
- **Observação qualitativa**:
  - PnL do treino mostra tendência ascendente ao longo dos timesteps, indicando que o
    PPO está capturando um edge razoável no ambiente configurado.
  - Número de trades por janela caiu em relação ao baseline inicial, mas ainda
    mostra picos altos — o agente continua relativamente ativo, porém menos
    scalper e mais próximo de um estilo swing em 4h.

> Próximas entradas devem seguir este formato, sempre anotando **por que**
> o config foi ajustado, **quais campos** mudaram e **quais artefatos** foram
> gerados para comparação.

---

## 2025-12-02 — ema_only_rl_v2_monthly_reward

- **config_sha256**: `4e642fb9b2394be22730093181933b56909cbf6fc34f9aa301803293aa617fd5`
- **Motivo**: aproximar melhor o objetivo de um trader humano que pensa em
  resultado mensal (e tem “contas a pagar”), agregando o reward por mês em vez
  de olhar PnL por candle e introduzindo um custo fixo por episódio.
- **Parâmetros alterados**:
  - Em `rl.reward`:
    - `living_cost_per_episode`: `20.0` (novo) — custo fixo distribuído ao longo
      de todo o episódio, representando despesas de operação/vida.
    - `use_monthly_reward`: `true` (novo) — ativa o modo de recompensa mensal.
  - Parâmetros previamente ajustados que permanecem vigentes:
    - `consensus_threshold`: `0.7`
    - `trade_penalty`: `0.01`
    - `turnover_penalty`: `0.01`
    - `min_hold_bars`: `5`
    - `intraday_min_alignment`: `0.6`
- **Alterações de lógica no ambiente (`rl_env.py`)**:
  - `RLConfig` passou a incluir:
    - `living_cost_per_episode: float`
    - `use_monthly_reward: bool`
  - No `reset`:
    - calcula `self._living_penalty_per_step = living_cost_per_episode / len(df)`
    - inicializa estado mensal:
      - `self._use_monthly` (flag),
      - `self._month_reward_accum = 0.0`,
      - `self._current_month` a partir da primeira `Date`.
  - No `step`:
    - reward por candle continua sendo calculado como antes (PnL mark-to-market,
      custos, penalidades, bônus, living cost).
    - se `use_monthly_reward = True`:
      - acumula em `self._month_reward_accum`.
      - só libera esse valor como reward quando:
        - muda o mês (detecção via `Date` do próximo candle) **ou**
        - o episódio termina.
      - nos outros candles, o reward retornado é `0.0`.
    - se `use_monthly_reward = False`, o comportamento é o antigo (reward por candle).
- **Artefatos de treino associados**:
  - Modelo PPO (v2): `src/strategies/ema_only/reports/rl/ppo_ema_only.zip`
  - Métricas de treino: `src/strategies/ema_only/reports/rl/metrics.csv`
  - Gráficos de treino: `src/strategies/ema_only/reports/rl/metrics.png`
  - Backtest mensal (jan–nov/2025): `src/strategies/ema_only/reports/rl/monthly_pnl_ema_only_rl.json`
- **Observação qualitativa (jan–nov/2025)**:
  - Mesmo com reward mensal e custo fixo, o backtest greedy ainda mostra equity
    praticamente flat (1000) mês a mês em 2025, indicando que a política
    aprendida em v2 segue evitando exposição relevante na janela de validação.
  - Próximos ajustes devem focar em mexer mais na estrutura (ex.: reduzir ainda
    mais penalidades de churn, permitir shorts, ou simplificar o gate) se a
    meta for aumentar o risco/retorno mensal em 2025.

---

## 2025-12-03 — ema_only_rl_v3_long_short_monthly

- **config_sha256**: `32692566b6530899a5a30c6565983b6f717713666a8ba3664342b77f7e94074f`
- **Motivo**: permitir que o agente opere tanto comprado quanto vendido (long/short),
  remover penalidades artificiais por trade e manter o foco em retorno mensal com
  custo fixo, aproximando o comportamento de um trader humano que aceita o risco
  do capital em jogo.
- **Principais alterações**:
  - `rl.reward.trade_penalty`: `0.0` (antes `0.01`).
  - `rl.reward.turnover_penalty`: `0.0` (antes `0.01`).
  - `rl.reward.living_cost_per_episode`: mantido em `20.0`.
  - `rl.reward.use_monthly_reward`: `true` (reward segue agregado por mês).
- **Alterações de lógica no ambiente (`rl_env.py`)**:
  - `RLConfig`:
    - novo campo `allow_short: bool = True`.
  - `EmaEnv`:
    - `action_space` passa a ser `Discrete(4)` com semântica:
      - `0`: hold (mantém posição atual)
      - `1`: alvo long (`position = +1`)
      - `2`: alvo short (`position = -1`, se `allow_short=True`)
      - `3`: alvo flat (`position = 0`)
    - `position` agora pode ser `-1` (short), `0` (flat), `1` (long).
    - Filtro de viés/ref:
      - long é favorecido quando preço ≥ `ref_ema`,
      - short quando preço ≤ `ref_ema`,
      - ambos bloqueados por `vol_max_atr_rel` quando atr_rel acima do limite.
    - Filtro de consenso:
      - long: `experts_mean >= consensus_threshold`
      - short: `experts_mean <= 1 - consensus_threshold`
    - Gate só atua ao **entrar a partir de flat**:
      - se o alvo for long/short e o filtro reprovar, a posição permanece flat
        e é aplicada `gating_penalty`.
    - Fechamento de posição:
      - Se o alvo difere da posição atual, a posição existente é fechada:
        - long: `pnl = (price - entry_price) * lot_size`
        - short: `pnl = (entry_price - price) * lot_size`
      - PnL é aplicado em `equity`, cobra-se o custo (`fee_pct + slippage_pct`)
        via `_apply_cost`, e (se configurado) `trade_penalty`/`turnover_penalty`
        são debitados da equity (no momento estão 0).
      - Se `pnl > 0` e `realized_bonus_coef > 0`, adiciona-se um bônus ao reward.
    - Abertura de nova posição:
      - Se o alvo não é flat, após eventual fechamento abre-se nova posição
        (long ou short), cobrando custo de entrada via `_apply_cost` e
        incrementando `trades`.
    - Mark-to-market:
      - long: `unreal = (price - entry_price) * lot_size`
      - short: `unreal = (entry_price - price) * lot_size`
      - reward de PnL continua sendo `mtm_equity - last_equity`, depois de
        atualizar `equity` com PnL realizado/custos.
    - Drawdown e bônus/alinhamento:
      - drawdown intraday continua baseado na trajetória de equity, independente
        de long/short.
      - bônus de alinhamento (`align_bonus`) permanece aplicado apenas em long
        quando `ema_fast > ema_slow > ref_ema`.
    - Modo mensal:
      - permanece igual à versão v2: reward por candle é acumulado e liberado
        apenas no fim de cada mês/episódio.
- **Visualização (`rl_visualize.py`)**:
  - `actions.png` agora interpreta:
    - pontos verdes (`^`) para ações com `action == 1` (alvo long),
    - pontos vermelhos (`v`) para `action == 2` (alvo short).
- **Resultado do backtest mensal (jan–nov/2025, greedy)**:
  - Equity inicial: `1000.0`.
  - PnL mensal aproximado:
    - jan: `-0.91%`
    - fev: `-0.88%`
    - mar: `+2.44%`
    - abr: `-0.44%`
    - mai: `0.0%`
    - jun: `-0.28%`
    - jul: `0.0%`
    - ago: `+0.36%`
    - set: `-0.06%`
    - out: `-1.46%`
    - nov: `-2.24%`
  - Equity final (~nov/2025): `965.19` (drawdown moderado, com alguns meses
    positivos, mas resultado acumulado ainda negativo).
- **Conclusão qualitativa v3**:
  - O agente agora efetivamente **opera** (long e short) em 2025 e o PnL mensal
    reflete esse comportamento, com ganhos em alguns meses e perdas em outros.
  - A curva ainda é levemente negativa no período analisado; próximos passos
    podem focar em:
      - ajustar melhor os filtros de consenso/ref para shorts,
      - experimentar outros níveis de `living_cost_per_episode`,
      - ou calibrar o período de treino/validação para evitar overfitting no
        histórico pré‑2025.

---

## 2025-12-03 — ema_only_rl_v4_trend_entry_shaping

- **config_sha256**: `7c161cf12c1e1c38fa875ca5a509f3fe627aa48ac634f18965e27cf13a6d227d`
- **Motivo**: introduzir um shaping mais “emocional” de entrada — um pequeno
  bônus ao abrir trades a favor da tendência EMA (fast vs slow) e uma pequena
  penalidade ao abrir contra a tendência, mantendo o custo fixo/living cost e
  o reward mensal.
- **Parâmetros adicionados/alterados**:
  - Em `rl.reward`:
    - `trend_entry_bonus`: `0.5` — bônus aplicado no momento de abertura de
      uma nova posição quando o sinal da posição (long=+1, short=-1) coincide
      com o sinal de `ema_fast - ema_slow`.
    - `trend_entry_penalty`: `0.5` — penalidade aplicada na abertura quando o
      sinal da posição é oposto ao sinal de `ema_fast - ema_slow`.
  - Demais campos relevantes mantidos:
    - `trade_penalty`: `0.0`
    - `turnover_penalty`: `0.0`
    - `living_cost_per_episode`: `20.0`
    - `use_monthly_reward`: `true`
    - `consensus_threshold`: `0.7`
    - `intraday_min_alignment`: `0.6`
    - `allow_short`: `true` (no código).
- **Alterações de lógica no ambiente (`rl_env.py`)**:
  - `RLConfig` passou a ter:
    - `trend_entry_bonus: float`
    - `trend_entry_penalty: float`
  - Em `EmaEnv.step`, no bloco de abertura de nova posição:
    - Após definir `self.position` e `self.entry_price`, calcula:
      - `trend = sign(ema_fast - ema_slow)` (se valores finitos; caso
        contrário, 0).
      - `pos_sign = sign(self.position)` (+1 long, -1 short).
    - Se `trend != 0` e `pos_sign != 0`:
      - `align = trend * pos_sign`
      - Se `align > 0` → `reward += trend_entry_bonus`
      - Se `align < 0` → `reward -= trend_entry_penalty`
    - Isso é puramente shaping (não altera `equity`), reforçando entradas
      alinhadas à tendência das EMAs e desincentivando “teimosia” contra a
      tendência.
- **Resultados (jan–nov/2025, backtest v4)**:
  - Após treinar novamente (`total_timesteps = 800k`) e rodar
    `rl_backtest`, o resumo mensal ficou:
    - jan–mai: equity ~1000 (sem PnL relevante);
    - jun: leve perda (~−0,05%);
    - jul–nov: equity praticamente estável (~999,5).
  - Ou seja, o shaping de entrada tornou o agente ainda mais seletivo/flat em
    2025 (comportamento quase neutro), o que pode ser desejável como baseline
    “super conservador”, mas ainda não produz um perfil agressivo de lucro
    mensal.
- **Conclusão qualitativa v4**:
  - O bônus/penalidade de entrada baseado em EMA alinha decisões com a
    tendência técnica, mas, combinado com os demais filtros (consenso,
    intraday, ref, living cost), acabou empurrando o agente para operar muito
    pouco na janela de validação.
  - Se o objetivo é aumentar retorno (e aceitamos mais volatilidade), os
    próximos ajustes devem provavelmente:
      - reduzir o living cost ou torná‑lo condicional ao tempo em flat,
      - ou relaxar parte dos filtros (ex.: permitir operações com consenso um
        pouco menor) enquanto mantemos esse shaping de tendência.

---

## 2025-12-03 — ema_only_rl_v5_relaxed_filters

- **config_sha256**: `dd8c758b193652e6f1fc6114502e75529fcb3dfe26bc274aacad9f8b870dd0c0`
- **Motivo**: deixar o agente um pouco mais “solto” para usar as EMAs a favor
  do lucro, reduzindo o custo fixo e afrouxando levemente os filtros de
  consenso/intraday e de hold mínimo, mantendo o shaping de tendência nas
  entradas.
- **Alterações de configuração**:
  - Em `data`:
    - `intraday_min_alignment`: `0.5` (antes `0.6`) — exige 50% de alinhamento
      intraday 1h vs 4h, em vez de 60%, para o expert intraday aprovar.
  - Em `rl.reward`:
    - `min_hold_bars`: `3` (antes `5`) — o agente pode realizar antes, reduz
      a penalização por trades mais curtos.
    - `churn_penalty`: `0.005` (antes `0.01`) — penalidade menor por fechar
      antes de `min_hold_bars`.
    - `atr_risk_scale`: `0.3` (antes `0.5`) — menos punição em ambientes de
      alta volatilidade (ATR relativa).
    - `consensus_threshold`: `0.6` (antes `0.7`) — afrouxa um pouco o gate de
      consenso dos experts (mais trades em zonas “ok”).
    - `living_cost_per_episode`: `10.0` (antes `20.0`) — reduz pela metade o
      custo fixo lógico por episódio (mais tolerância a períodos em que o
      agente trabalha pouco).
    - `trend_entry_bonus` / `trend_entry_penalty`: mantidos em `0.5`.
- **Resultados (jan–nov/2025, backtest v5)**:
  - Equity inicial: `1000`.
  - PnL mensal:
    - jan: −0.06%
    - fev: −0.32%
    - mar: −0.25%
    - abr: +0.03%
    - mai: +0.27%
    - jun: −0.42%
    - jul: +0.21%
    - ago: −0.18%
    - set: −0.23%
    - out: +0.66%
    - nov: 0.00%
  - Equity final: ~`997` (perda de ~0,3% no período, com vários meses
    levemente positivos e negativos).
- **Conclusão qualitativa v5**:
  - Com os filtros relaxados, o agente volta a operar mais em 2025, mas ainda
    com perfil bem moderado (resultado agregado quase flat).
  - O shaping de tendência (bônus/penalidade nas entradas) ajuda a manter
    coerência com as EMAs; o PnL ainda é pequeno porque os custos e o
    living cost continuam relevantes e a estratégia é relativamente
    conservadora.
  - Essa versão já captura um comportamento mais humano: respeita a tendência
    das EMAs, entra/saí com frequência moderada, e não explode o risco; ganhos
    mais agressivos exigiriam ou maior alavancagem (lot_size) ou menos
    proteção (custos e filtros mais leves).

---

## 2025-12-03 — ema_only_rl_v6_soft_trend_penalty

- **config_sha256**: `46e25be8da754579c5640d74e972ae1fc7853ce0651889942d5ec2398cec12a4`
- **Motivo**: aplicar uma penalidade pequena ao abrir trades contra a tendência
  das EMAs e um bônus igualmente pequeno ao abrir a favor da tendência, em vez
  do shaping forte anterior, para que o agente ainda aprenda principalmente
  pelo PnL mensal, mas receba um “empurrão” leve em direção às regras visuais
  que um trader humano usaria.
- **Parâmetros alterados (antes → depois)**:
  - Em `rl.reward`:
    - `trend_entry_bonus`: `0.5` → `0.01`
    - `trend_entry_penalty`: `0.5` → `0.02` (penalidade ~2x maior que o bônus
      para desincentivar entradas contra a tendência das EMAs).
- **Lógica de ambiente**:
  - A implementação em `rl_env.py` permanece a mesma da versão v4/v5:
    - ao abrir uma nova posição, calcula-se `trend = sign(ema_fast - ema_slow)`
      e `pos_sign = sign(position)`; se `trend * pos_sign > 0`, aplica-se
      `trend_entry_bonus`, se `< 0`, aplica-se `trend_entry_penalty`.
  - A mudança aqui é puramente de **intensidade** do shaping, reduzindo o peso
    relativo desse termo em relação ao PnL mensal, custos e demais bônus/
    penalidades (churn, consenso, etc.).
- **Próximos passos sugeridos**:
  - Re-treinar o agente com este config (`train.py`) e comparar:
    - PnL mensal 2025 (`rl_backtest.py`),
    - padrão de ações em `reports/rl/actions.png` (espera-se ainda respeito à
      tendência, mas com mais liberdade para explorar).
  - Se o agente ainda insistir em muitas entradas contra a tendência, considerar
    subir levemente `trend_entry_penalty` (ex.: 0.03) ou combinar com filtros
    adicionais baseados em regime.

---

## 2025-12-05 — ema_only_rl_v7_flip_gate_short_bonus

- **config_sha256**: `cc487fd747737416afc0c90c8e49b25e9d0e49e798281d3f1586bc004786baff` (JSON sem mudanças; ajuste foi no ambiente).
- **Motivo**: corrigir assimetria long/short aplicando o gate também em flips de posição e alinhar o bônus de short aos degraus de EMAs (fast < slow = bônus 1x; fast < slow e slow < ref = bônus 2x).
- **Alterações de lógica no ambiente (`rl_env.py`)**:
  - O gate de entrada passa a rodar sempre que há tentativa de abrir/flipar posição; se o filtro reprova, a posição atual é mantida (ou permanece flat) e aplica-se `gating_penalty`.
  - O bônus de short em dois degraus agora checa `ema_slow < ref_ema` para conceder o bônus cheio (em vez de `ema_fast < ref_ema`).
- **Artefatos do backtest (jan–nov/2025, greedy)**:
  - Resumo mensal: `src/strategies/ema_only/reports/rl/monthly_pnl_ema_only_rl.json`.
  - Ações e preço/EMAs: `src/strategies/ema_only/reports/rl/actions.png`.
  - Log detalhado por candle/ação: `src/strategies/ema_only/reports/rl/actions_debug.csv`.
- **Resultado do backtest mensal (equity inicial 1000)**:
  - PnL por mês (%): jan −10.13, fev −10.43, mar +7.09, abr +25.36, mai +21.28, jun −8.68, jul +8.88, ago +7.12, set −3.53, out +1.90, nov +16.46.
  - Equity final: `1598.03` (+59.8% no período).
- **Observação qualitativa**: o agente voltou a operar ambos os lados com frequência; os bônus/punições de entrada agora refletem a hierarquia fast/slow/ref, e o gate evita flips “cegos”, resultando em uma curva mais inclinada no ciclo de validação 2025.

---

## 2025-12-03 — ema_only_rl_v7_avoid_tops_bottoms

- **config_sha256**: `4dcfae9a73f731aaf68e7ee117922dd68945fe032d1ff226f1c5d59f4579b7fb`
- **Motivo**: aproximar o comportamento “nunca comprar topo / nunca vender
  fundo” que um trader humano adota, forçando o agente a abrir posições apenas
  quando o preço estiver relativamente próximo da EMA rápida, em vez de
  comprar rompimentos muito esticados ou vender em capitulações.
- **Parâmetros adicionados em `rl.reward`**:
  - `max_long_entry_dist_fast_pct`: `0.005` — distância máxima permitida entre
    o preço e a `ema_fast` (em % relativa à ema_fast) para abrir **long**.
    Se `(close - ema_fast) / ema_fast > 0.005` e o agente tentar abrir long a
    partir de flat, o gate bloqueia a entrada e aplica `gating_penalty`.
  - `max_short_entry_dist_fast_pct`: `0.005` — análogo para **short**: evita
    abrir venda quando o preço está mais de 0,5% abaixo da `ema_fast`.
- **Alterações de lógica em `rl_env.py`**:
  - `RLConfig` passou a ter:
    - `max_long_entry_dist_fast_pct: float`
    - `max_short_entry_dist_fast_pct: float`
  - No bloco de gate (entradas a partir de `position == 0`), além dos filtros
    de viés de referência (`ref_long_ok/ref_short_ok`) e consenso dos experts,
    agora é calculada a distância
    `dist_fast = (price - ema_fast) / ema_fast`:
    - Para long:
      - se `max_long_entry_dist_fast_pct > 0` e `dist_fast` for maior que esse
        limite, a entrada é bloqueada e aplicado `gating_penalty`.
    - Para short:
      - se `max_short_entry_dist_fast_pct > 0` e `dist_fast < -max_short_entry_dist_fast_pct`,
        a entrada é bloqueada (evita vender fundo).
- **Intuição**:
  - Em tendência de alta, o agente passa a comprar preferencialmente em
    **pullbacks até a ema_fast**, não em candles extremamente esticados para
    cima (topos locais).
  - Em tendência de baixa, análogo: prefere abrir short em repiques até perto
    da ema, não nas mínimas extremas.
- **Próximos passos sugeridos**:
  - Re-treinar (`train.py`) e rodar:
    - `rl_backtest.py` para ver o PnL mensal 2025,
    - `visualize.py` para inspecionar se as entradas long/short de fato
      acontecem mais “coladas” na ema_fast.
  - Se o agente ficar conservador demais (poucos trades), considerar subir o
    limite para `0.0075` ou `0.01`; se ainda comprar topo, reduzir para
    `0.003` ou mesmo `0.0` (obriga comprar sempre abaixo/na ema_fast).

---

## 2025-12-03 — ema_only_rl_v8_pullback_and_exit_shaping

- **config_sha256**: `07e0ff53dd9360b4dfc939d6f4c5790507ae777dd3ae7da72502e6f0e4265763`
- **Motivo**: transformar parte da lógica “não comprar topo / não vender
  fundo” em **conselhos suaves**, em vez de regras duras, para o agente
  aprender um comportamento mais humano: comprar pullbacks até a ema rápida,
  vender em repiques e segurar posição enquanto o preço estiver do lado bom
  da tendência.
- **Ajustes em `rl.reward`**:
  - `max_long_entry_dist_fast_pct`: `0.005` → `0.012`
  - `max_short_entry_dist_fast_pct`: `0.005` → `0.012`
    - O gate duro passa a bloquear apenas entradas em extremos muito
      esticados em relação à `ema_fast`; casos intermediários ficam a cargo
      do PPO decidir, usando os especialistas e o reward.
  - `pullback_entry_bonus`: `0.01` (novo)
  - `trend_exit_penalty`: `0.01` (novo)
- **Alterações em `rl_env.py`**:
  - `RLConfig` recebeu:
    - `pullback_entry_bonus: float`
    - `trend_exit_penalty: float`
  - Na abertura de nova posição:
    - Após calcular `ema_fast`/`ema_slow` e aplicar o shaping de tendência
      (`trend_entry_bonus/penalty`), é calculado
      `dist_fast_entry = (price - ema_fast) / ema_fast`.
    - Se `pullback_entry_bonus > 0`:
      - long: se `dist_fast_entry <= 0` (preço em/abaixo da `ema_fast`),
        aplica-se um pequeno bônus de entrada (compra em pullback).
      - short: se `dist_fast_entry >= 0` (preço em/acima da `ema_fast`),
        aplica-se o mesmo bônus (venda em repique).
  - No fechamento de posição:
    - Ao encerrar um long ou short, calcula-se
      `dist_fast_close = (price - ema_fast) / ema_fast`.
    - Se `trend_exit_penalty > 0`:
      - long: se `dist_fast_close > 0` (preço ainda acima da `ema_fast`),
        aplica-se uma pequena penalidade (saída “cedo demais”).
      - short: se `dist_fast_close < 0` (preço ainda abaixo da `ema_fast`),
        idem (cobertura de short ainda em tendência de baixa).
- **Intuição**:
  - Em vez de proibir tudo, o agente recebe **recompensas/penalidades
    adicionais** que o incentivam a:
    - abrir posições mais próximas da EMA rápida (pullbacks/repiques),
    - evitar realizar lucro/perda enquanto o preço ainda está bem posicionado
      em relação à EMA.
  - O filtro duro por distância continua existindo, mas só em esticadas
    maiores (~1,2%), funcionando como “cinto de segurança”, não como piloto
    automático.
- **Próximos passos sugeridos**:
  - Re-treinar (`train.py`) e comparar:
    - PnL mensal 2025 (`rl_backtest.py`),
    - padrão de entradas/saídas em `actions.png` (espera-se menos vendas em
      fundos óbvios e mais compras em volta da ema_fast).
  - Se o shaping estiver fraco, aumentar levemente
    `pullback_entry_bonus` e/ou `trend_exit_penalty` (ex.: 0.015–0.02); se
    estiver forte demais (agente fica preso em posições ruins), reduzir.

---

## 2025-12-03 — ema_only_rl_v9_restore_v5_baseline

- **config_sha256**: `7fef0645577c39513cb5ba794a1f8d3d0bbcde883bef3c9b716358cfd5e370f5`
- **Motivo**: voltar ao comportamento de referência da versão v5 (que já era
  quase flat com mais trades), agora com as melhorias de logging (`actions_debug.csv`)
  e o `actions.png` corrigido (mostrando apenas entradas reais), para que os
  próximos ajustes sejam feitos em cima de um baseline conhecido e com
  visualização fiel.
- **Ajustes em `config.json`**:
  - Em `rl.reward`, removidos os campos introduzidos em v7/v8:
    - `max_long_entry_dist_fast_pct`
    - `max_short_entry_dist_fast_pct`
    - `pullback_entry_bonus`
    - `trend_exit_penalty`
  - Restaurados os pesos de shaping de tendência da v5:
    - `trend_entry_bonus`: `0.5`
    - `trend_entry_penalty`: `0.5`
  - Demais campos relevantes permanecem como em v5:
    - `trade_penalty`: `0.0`
    - `dd_threshold_pct`: `0.02`, `dd_penalty`: `0.02`
    - `min_hold_bars`: `3`, `churn_penalty`: `0.005`
    - `align_bonus`: `0.001`
    - `atr_risk_scale`: `0.3`
    - `enforce_ref_bias`: `true`
    - `reward_scale`: `1.0`
    - `consensus_bonus`: `0.003`, `consensus_threshold`: `0.6`
    - `vol_max_atr_rel`: `0.015`, `vol_penalty`: `0.02`
    - `gating_penalty`: `0.015`
    - `turnover_penalty`: `0.0`
    - `living_cost_per_episode`: `10.0`
    - `use_monthly_reward`: `true`
    - `data.intraday_min_alignment`: `0.5`
- **Observação**:
  - A lógica extra de logging e visualização (arquivo
    `actions_debug.csv` no `rl_backtest` e `actions.png` marcando apenas
    0→1/0→−1) permanece ativa, mas ela não altera o ambiente/reward — apenas
    nossa capacidade de observar o comportamento do agente.
  - A partir desta versão, novos experimentos de ajuste de tendência/pullback
    devem ser comparados com os resultados de v5/v9 usando os mesmos gráficos
    e logs.

---

## 2025-12-03 — ema_only_rl_v10_trend_flip_penalty

- **config_sha256**: `58ae78d6b4747553b58690e659ff57ef1d9c4bd7ab0eefae3c76689c42d68ffa`
- **Motivo**: reduzir o “vai e volta” de maio em plena tendência de alta,
  penalizando inversões de posição (long → short, short → long) quando os
  especialistas de tendência continuam alinhados, ou seja, quando um trader
  humano provavelmente preferiria apenas segurar a posição atual.
- **Ajustes em `rl.reward`**:
  - Adicionado `trend_flip_penalty`: `0.1`.
    - Valor moderado: mais fraco que o `trend_entry_penalty` (0.5), para
      desencorajar mas não proibir totalmente as viradas.
- **Lógica adicionada em `rl_env.py`**:
  - `RLConfig` ganhou o campo:
    - `trend_flip_penalty: float`.
  - No início de `step`, antes de fechar a posição:
    - Se `desired_pos != 0`, `position != 0` e `desired_pos != position`
      (tentativa de inverter posição) e `trend_flip_penalty > 0`:
      - Lê `exp_trend` e `exp_ref` nas features:
        - Se `exp_trend >= 0.5` e `exp_ref >= 0.5` (regime de **alta forte**):
          - Se a posição atual é `1` (long) e o alvo é `-1` (short),
            aplica `trend_flip_penalty` ao reward.
        - Se `exp_trend < 0.5` e `exp_ref < 0.5` (regime de **baixa forte**):
          - Se a posição atual é `-1` (short) e o alvo é `1` (long),
            aplica `trend_flip_penalty`.
      - Em regimes mistos (ex.: `exp_trend` e `exp_ref` divergentes), nenhuma
        penalidade é aplicada — o agente é livre para virar a mão.
- **Intuição**:
  - Em abril, quando o mercado sobe forte e o agente já está comprado, tentar
    inverter para short em plena tendência passa a ficar mais caro em termos
    de reward, incentivando a manter o swing.
  - Em maio, esperamos menos alternâncias inúteis entre long/short em zonas
    onde os especialistas ainda apontam para a mesma direção, reduzindo churn
    e o desgaste de PnL observados no mês.
- **Próximos passos sugeridos**:
  - Re-treinar (`train.py`) e comparar o PnL mensal, em especial maio e julho.
  - Ver no novo `actions.png` se:
    - há menos viradas de mão em tendências fortes,
    - o agente tende a “respeitar” mais o trade atual quando `exp_trend` e
      `exp_ref` concordam.
  - Ajustar `trend_flip_penalty` para cima (ex.: 0.15–0.2) se ainda houver
    muitas viradas desnecessárias, ou para baixo se o agente ficar rígido
    demais em mudanças de regime.

---

## 2025-12-03 — ema_only_rl_v11_atr_trailing_stop

- **config_sha256**: `7be25ed70bccbd081ca6ee3205f95a94149efe83ba73f6e36555bd4a4c9db482`
- **Motivo**: evitar que trades longos devolvam grande parte do lucro ao
  ficar “presos” contra a nova tendência, adicionando um stop mecânico
  baseado em ATR que se movimenta a favor da operação (trailing stop).
- **Novos parâmetros em `rl.reward`**:
  - `atr_stop_mult`: `2.0` — múltiplo de ATR para definir o stop inicial:
    - long: `stop = entry_price - 2 * ATR`
    - short: `stop = entry_price + 2 * ATR`
  - `atr_trail_mult`: `1.0` — múltiplo de ATR para o trailing:
    - long: em cada barra, candidato a novo stop:
      `trail = price - 1 * ATR`; o stop sobe com o preço (`max`), nunca desce.
    - short: `trail = price + 1 * ATR`; o stop desce com o preço (`min`),
      nunca sobe.
- **Alterações em `rl_env.py`**:
  - `RLConfig` passou a ter:
    - `atr_stop_mult: float`
    - `atr_trail_mult: float`
  - No `reset`:
    - inicializa `self.stop_price = 0.0`.
  - No início de `step`:
    - calcula `atr_value = atr_rel * close` (quando `atr_rel` está disponível).
    - se houver posição (`position != 0`) e `atr_trail_mult > 0`:
      - atualiza `self.stop_price` conforme as fórmulas de trailing acima.
  - Ao abrir nova posição:
    - define `self.stop_price` usando `atr_stop_mult` (ou deixa 0.0 se ATR ou
      o multiplicador não estiverem disponíveis).
  - Ao fechar posição (qualquer motivo):
    - zera `self.stop_price`.
  - Antes do bloco de fechamento normal:
    - aplica o stop ATR como força de saída:
      - long: se `price <= stop_price`, força `desired_pos = 0`.
      - short: se `price >= stop_price`, força `desired_pos = 0`.
- **Intuição**:
  - Em tendências fortes, o agente ainda pode segurar posições longas/curtas,
    mas à medida que o preço anda a favor, o stop “sobe” (ou “desce”) junto,
    travando parte do lucro.
  - Quando a reversão passa de um ruído de 1–2 ATRs, o stop é acionado e a
    posição é fechada, em vez de deixar o PPO decidir sozinho ficar long em
    plena tendência contrária (como vimos em março).
- **Próximos passos sugeridos**:
  - Re-treinar (`train.py`) e observar:
    - se os períodos de perda prolongada contra a tendência (como março)
      diminuem,
    - se o agente passa a realizar mais lucro parcial em swings longos.
  - Ajustar:
    - `atr_stop_mult` maior (ex.: 2.5–3.0) para stops mais largos,
    - ou `atr_trail_mult` menor/maior para trailing mais solto ou mais apertado,
      conforme o comportamento desejado.

---

## 2025-12-03 — ema_only_rl_v12_wider_atr_and_free_flip

- **config_sha256**: `99d2acb5647b4ad94910b4e12f1d0eae2e212afe66b014d13d4a3f3b74437cdd`
- **Motivo**: reduzir a frequência de stops em ruído (especialmente em meses
  como junho e outubro) e liberar o agente para explorar melhor operações
  short, removendo a penalidade explícita de virar posição.
- **Alterações em `rl.reward`**:
  - `atr_stop_mult`: `2.0` → `3.0`
    - O stop inicial fica mais largo (3 ATRs) em relação ao preço de entrada,
      evitando que correções pequenas interrompam trades potencialmente bons.
  - `atr_trail_mult`: `1.0` → `1.5`
    - O trailing stop acompanha o preço com uma folga maior (1,5 ATR),
      reduzindo stops muito apertados em movimentos ainda saudáveis.
  - `trend_flip_penalty`: `0.1` → `0.0`
    - Remove a penalização direta por virar de long→short ou short→long em
      plena tendência, deixando o PPO decidir com base no PnL/experts se vale
      a pena inverter ou não.
- **Expectativa de comportamento**:
  - Menos sequências de pequenos stops negativos em fases laterais ou de transição,
    já que o stop só é acionado em movimentos mais significativos.
  - O agente fica mais livre para experimentar shorts quando os experts
    indicarem tendência de baixa, já que não há mais custo extra específico
    para a inversão; o gate de consenso/ref continua controlando entradas ruins.

---

## 2025-12-03 — ema_only_rl_v13_strong_trend_gate

- **config_sha256**: `99d2acb5647b4ad94910b4e12f1d0eae2e212afe66b014d13d4a3f3b74437cdd` (mesmo config, lógica alterada apenas em código).
- **Motivo**: tornar o agente ainda mais seletivo, exigindo confirmação
  explícita dos especialistas de tendência para abrir posições, de forma a
  evitar operar em regimes “meio termo” como fevereiro/março/agosto onde o
  consenso é fraco.
- **Alterações em `rl_env.py` (gate de entrada)**:
  - Após calcular `cons` (`experts_mean`) e o limiar de consenso
    (`consensus_threshold`), passamos a derivar dois flags:
    - `trend_long_ok`
    - `trend_short_ok`
  - Lógica:
    - Se `exp_trend` e `exp_ref` existirem:
      - Quando `cons >= 0.5`:
        - `trend_long_ok = True` somente se `exp_trend >= 0.5` **e**
          `exp_ref >= 0.5` (ambos apontando para alta).
        - `trend_short_ok = True` somente se `exp_trend < 0.5` **e**
          `exp_ref < 0.5` (ambos apontando para baixa).
        - Caso haja discordância ou valores não finitos, ambos ficam `False`.
      - Quando `cons < 0.5`:
        - `trend_long_ok = False` e `trend_short_ok = False` — nenhum lado
          é considerado “tendência forte”.
  - No gate (entradas a partir de `position == 0`):
    - Para long:
      - condição de entrada passa a ser
        `ref_long_ok and cons_long_ok and trend_long_ok`
        (antes não checávamos `trend_long_ok`).
    - Para short:
      - condição de entrada passa a ser
        `ref_short_ok and cons_short_ok and trend_short_ok`.
    - Se a condição falhar, a ação de entrada é bloqueada (`desired_pos = 0`)
      e aplica-se `gating_penalty` como antes.
- **Intuição**:
  - O agente só entra:
    - **comprado** quando:
      - preço está acima da ref_ema (se `enforce_ref_bias`),
      - consenso dos experts é suficientemente alto (`experts_mean >= threshold`),
      - e os dois especialistas de tendência (`exp_trend`, `exp_ref`) concordam
        com a direção de alta.
    - **vendido** quando o oposto vale para a tendência de baixa.
  - Em meses como fevereiro/março/agosto, onde `experts_mean` médio é ~0.3–0.4
    e os especialistas divergem com frequência, o agente tende a operar muito
    menos, reduzindo a quantidade de pequenos trades negativos nesses regimes.

---

## 2025-12-04 — ema_only_rl_v14_gate_relaxed

- **config_sha256**: `87ec1f537d2a9d0155c21a920b419022b8472f581180f24b74cce46b84f97ff2`
- **Motivo**: destravar shorts e reduzir o sufocamento do gate, suavizando o
  shaping/penalidades e encurtando stops para melhorar o balanço win/loss.
- **Parâmetros alterados (antes → depois)**:
  - `rl.reward.consensus_threshold`: `0.6` → `0.55`
  - `rl.reward.gating_penalty`: `0.015` → `0.01`
  - `rl.reward.trend_entry_bonus` / `trend_entry_penalty`: `0.5`/`0.5` → `0.1`/`0.1`
  - `rl.reward.trend_flip_penalty`: `0.0` → `0.05`
  - `rl.reward.atr_stop_mult`: `3.0` → `2.5`
  - `rl.reward.atr_trail_mult`: `1.5` → `1.0`
  - Reintroduzidos em `rl.reward`:
    - `max_long_entry_dist_fast_pct`: `0.0075`
    - `max_short_entry_dist_fast_pct`: `0.0075`
    - `pullback_entry_bonus`: `0.01`
    - `trend_exit_penalty`: `0.01`
- **Alteração de lógica no gate (código `rl_env.py`)**:
  - `trend_long_ok = exp_trend >= 0.5 and exp_ref >= 0.5`
  - `trend_short_ok = exp_trend < 0.5 and exp_ref < 0.5`
  - Removida a exigência de `cons >= 0.5` para checar tendência, liberando
    ~32% dos sinais de short que eram bloqueados.
- **Artefatos**: ainda não re‑treinado/backtestado com esta versão (treinar +
  `rl_backtest.py` para gerar novos `metrics.csv`/`monthly_pnl_...json`).
- **Hipótese esperada**:
  - Mais passes no gate (especialmente shorts), menor custo por bloqueio e
    stops/trailing mais apertados devem reduzir a razão perda/ganho (antes ~2.3x)
    e aumentar o número de operações aproveitáveis.

---

## 2025-12-04 — ema_only_rl_v15_entry_bonus_tiers

- **config_sha256**: `89f2c9ecce20e2e64a4d8ac550d3ab8d405e1ccaffa694fa52a17ea492d7956b`
- **Motivo**: alinhar o comportamento do agente ao “trader humano de EMAs”:
  privilegiar entradas quando a EMA rápida está acima da média e, melhor ainda,
  quando a média está acima da lenta, sem usar a ref_ema como bloqueio duro.
- **Parâmetros alterados**:
  - `enforce_ref_bias`: `false` (ref_ema vira só indicador de regime, não gate).
  - `trend_entry_bonus` / `trend_entry_penalty`: `0.0` / `0.0` (shaping antigo desativado).
  - Novos degraus de bônus na entrada:
    - `entry_bonus_fast_over_slow`: `0.05` (EMA rápida > EMA média).
    - `entry_bonus_full_trend`: `0.10` (EMA rápida > EMA média > EMA lenta/ref).
- **Lógica no ambiente (`rl_env.py`)**:
  - Ao abrir uma posição:
    - **Long**: se `ema_fast > ema_slow`, aplica `entry_bonus_fast_over_slow`; se também `ema_slow > ref_ema`, aplica `entry_bonus_full_trend` (patamar cheio).
    - **Short**: espelho, usando `<` (fast < slow < ref_ema).
    - Se apenas o primeiro degrau for atendido, recebe só o bônus do degrau 1; se ambos, recebe o patamar cheio.
  - Mantidos: bônus de pullback (`pullback_entry_bonus`) e filtros de topo/fundo por distância da EMA rápida.
- **Outros campos mantidos**:
  - `consensus_threshold`: `0.55`, `gating_penalty`: `0.01`
  - Stops/trailing: `atr_stop_mult`: `2.5`, `atr_trail_mult`: `1.0`
  - Proteções de topo/fundo: `max_long_entry_dist_fast_pct` / `max_short_entry_dist_fast_pct`: `0.0075`
- **Próximos passos**:
  - Re-treinar (`train.py`) e rodar `rl_backtest.py` para ver se o agente passa a operar mais nas fases de tendência, usando a ref_ema apenas como “plus” (não como bloqueio).

---

## 2025-12-04 — ema_only_rl_v16_looser_short_gate

- **config_sha256**: `6e5be48c63d9793390eda081f64c531d48b33ba82fb5f5fa5e37264225faeee3`
- **Motivo**: liberar mais operações de venda (short) e sinais em reversões,
  reduzindo o consenso mínimo e exigindo apenas um dos especialistas apontando
  para baixa.
- **Parâmetros alterados**:
  - `rl.reward.consensus_threshold`: `0.55` → `0.5`.
- **Lógica no ambiente (`rl_env.py`)**:
  - `trend_short_ok` passa a ser verdadeiro se **exp_trend < 0.5 OU exp_ref < 0.5**
    (antes exigia ambos < 0.5). Para long, permanece exp_trend >= 0.5 **e**
    exp_ref >= 0.5.
- **Demais campos mantidos**: bônus em degraus de entrada, stops/trailing,
  filtros de topo/fundo e ref_ema apenas como indicador (sem gate).
- **Próximo passo**: re-treinar e rodar `rl_backtest.py` para verificar se o
  volume de shorts aumenta e se o PnL melhora em meses de reversão.

---

## 2025-12-04 — ema_only_rl_v17_cons045_trend_or_tighter_stops

- **config_sha256**: `9edd81c21975704892d3c7f3e644922eb75ddcc3acc5f831080af84e7dd12326`
- **Motivo**: abrir mais sinais (especialmente em reversões) e reduzir perda média
  apertando stops/trailing e afrouxando um pouco a distância das EMAs rápidas.
- **Parâmetros alterados**:
  - `consensus_threshold`: `0.5` → `0.45`
  - `atr_stop_mult`: `2.5` → `2.0`
  - `atr_trail_mult`: `1.0` → `0.8`
  - `max_long_entry_dist_fast_pct`: `0.0075` → `0.01`
  - `max_short_entry_dist_fast_pct`: `0.0075` → `0.01`
- **Lógica no gate (`rl_env.py`)**:
  - `trend_long_ok` agora aceita **exp_trend >= 0.5 OU exp_ref >= 0.5**
    (antes precisava dos dois).
  - `trend_short_ok` permanece como OU para baixa.
- **Campos mantidos**:
  - Bônus em degraus de entrada (`entry_bonus_fast_over_slow`, `entry_bonus_full_trend`),
    pullback/exit, ref_ema apenas como indicador (sem gate).
- **Próximo passo**: re-treinar e rodar `rl_backtest.py` para ver se:
  - há mais trades no 2º semestre,
  - a perda média cai (stops/trailing mais apertados),
  - o PnL em meses mistos melhora com consenso mais baixo.

---

## 2025-12-05 — ema_only_rl_v18_open_long_gate_cons035

- **config_sha256**: `ebc8688beae91542847fcc6cc9bf2432a2fd4b13affd25fdc93420919cc6a236`
- **Motivo**: testar um “modo aberto” para entradas long, baixando ainda mais o
  consenso e removendo o gate de consenso/tendência/distância para compras,
  enquanto mantém controle para shorts.
- **Parâmetros alterados**:
  - `consensus_threshold`: `0.45` → `0.35`
  - `override_long_gate`: `true` (novo) — não bloqueia entradas long por
    consenso/tendência/distância; continua aplicando o gating_penalty se bloqueado
    apenas por distância? (na lógica, override ignora o block inteiro).
  - `override_short_gate`: `false` (short segue com gate normal).
- **Lógica no ambiente (`rl_env.py`)**:
  - Gate:
    - long: se `override_long_gate=True`, não há bloqueio por consenso/trend/distância.
    - short: permanece exigindo consenso/tendência/distância (com OR na tendência).
- **Demais campos mantidos**: stops 2.0/0.8, distâncias 1%, bônus em degraus,
  ref_ema só como indicador.
- **Próximo passo**: re-treinar e rodar `rl_backtest.py` para ver se:
  - o volume de longs aumenta e melhora PnL,
  - e se manter gate para shorts evita exposição ruim em reversões.

---

## 2025-12-05 — ema_only_rl_v19_bear_regime_stop

- **config_sha256**: `ebc8688beae91542847fcc6cc9bf2432a2fd4b13affd25fdc93420919cc6a236`
- **Motivo**: evitar devolver lucros em reversões fortes, forçando zerar longs
  quando os especialistas sinalizam regime de baixa.
- **Lógica adicionada em `rl_env.py`**:
  - Calcula `in_bear_regime` se `exp_trend < 0.4` OU `exp_ref < 0.4`.
  - Se o agente estiver **long** e `in_bear_regime=True`, força `desired_pos=0`
    (stop de regime), independente do stop ATR.
- **Demais parâmetros**: mantidos em relação à v18 (consensus 0.35, gate aberto
  para long, stops 2.0/0.8, distâncias 1%, etc.).
- **Próximo passo**: rodar `rl_backtest.py` (não precisa retreinar) para medir
  impacto: redução de perdas em agosto/novembro e saldo final.

---

## 2025-12-05 — ema_only_rl_v20_bear_gate_and_short_cons

- **config_sha256**: `6bf2d958085585ca16ecbc3cf77eb0ef42f536bae50005cbc6f72ce1da5581e9`
- **Motivo**: reduzir compras em regimes de baixa e dar mais liberdade para shorts nesses regimes.
- **Parâmetros alterados**:
  - `bear_regime_threshold`: `0.45` (novo) — define o “bear estrito” quando exp_trend < 0.45 e exp_ref < 0.45.
  - `block_long_in_bear`: `true` (novo) — bloqueia novas entradas long em bear estrito (aplica gating_penalty).
  - `bear_consensus_short_threshold`: `0.3` (novo) — em bear estrito, shorts usam consenso especial mais baixo.
  - `override_long_gate`: permanece `true` (gate liberado), mas o bloqueio de bear tem prioridade.
- **Lógica no ambiente (`rl_env.py`)**:
  - Calcula `in_bear_strict` com `bear_regime_threshold`.
  - Long: se `block_long_in_bear=True` e `in_bear_strict=True`, bloqueia a entrada mesmo com override.
  - Short: se `in_bear_strict=True`, usa `bear_consensus_short_threshold` para `cons_short_ok`.
- **Demais campos mantidos**: consenso global 0.35, stops 2.0/0.8, distâncias 1%, bônus em degraus, stop de regime para longs (zera se exp_trend/ref < 0.4).
- **Próximo passo**: rodar `rl_backtest.py` (modelo atual) para ver se:
  - reduz compras no meio da queda,
  - aumenta captura de shorts em regime de baixa,
  - e melhora o PnL em agosto–novembro.

---

## 2025-12-05 — ema_only_rl_v21_short_gate_open

- **config_sha256**: `fffe7784d94ea410f6860f1c8a1160658ae94041a5981bfb5d13391fd8a93050`
- **Motivo**: liberar completamente o gate para shorts, a exemplo do long, para medir se capturamos melhor as quedas recentes.
- **Parâmetro alterado**:
  - `override_short_gate`: `false` → `true`
- **Demais campos mantidos**: consenso 0.35, bear regime (block_long_in_bear=true, bear_consensus_short_threshold=0.3), stops 2.0/0.8, distâncias 1%, bônus em degraus.
- **Próximo passo**: rodar `rl_backtest.py` (modelo atual) para ver impacto no PnL de agosto–novembro.

---

## 2025-12-05 — ema_only_rl_v22_tighter_stops

- **config_sha256**: `e24e3eb2f908a56c82a0cdf672e7b49ef775db3ac60e028ad2405e1cf3886682`
- **Motivo**: reduzir perdas por trade e proteger melhor lucros em quedas, antes de retreinar.
- **Parâmetros alterados**:
  - `atr_stop_mult`: `2.0` → `1.5`
  - `atr_trail_mult`: `0.8` → `0.5`
- **Demais campos mantidos**: gate aberto para long/short, bear regime (block_long_in_bear=true, bear_consensus_short_threshold=0.3), consenso 0.35, distâncias 1%, bônus em degraus, stop de regime para longs.
- **Próximo passo**: rodar `rl_backtest.py` (modelo atual) para efeito imediato e, idealmente, retreinar com o ambiente simplificado.

---

## 2025-12-05 — ema_only_rl_v24_monthly_targets

- **config_sha256**: `50ef1e61dab6238be82a85d81ec01b0d449151768a66c5d8a5b95d1e41e3831d`
- **Motivo**: dar um norte explícito de retorno mensal (tiers 2%/4%/5%) e punir meses negativos.
- **Parâmetros adicionados/alterados**:
  - `monthly_target_tiers`: `[[0.02, 0.5], [0.04, 1.0], [0.05, 1.5]]` — bônus aplicado ao fechar o mês se o retorno >= tier.
  - `monthly_shortfall_penalty`: `0.5` — penalidade se o retorno mensal for < 0.
  - `max_long_entry_dist_fast_pct` / `max_short_entry_dist_fast_pct`: `0.0` (sem bloqueio por distância).
  - `exit_on_fast_slow_cross`: `true` (sai quando fast cruza slow contra a posição).
- **Lógica no ambiente (`rl_env.py`)**:
  - Ao fechar o mês (reward mensal agregado), calcula `month_ret` e adiciona o bônus do maior tier atingido; se `month_ret < 0`, aplica `monthly_shortfall_penalty`.
- **Próximo passo**: retreinar (`poetry run python -m src.strategies.ema_only.train`) e backtestar com o modelo novo para avaliar se o agente passa a perseguir as metas mensais.

---

## 2025-12-05 — ema_only_rl_v25_trend_surfer_gate

- **config_sha256**: `24333a9d50da8b1939a670412dcc8d37639a38c7fa674f8ba6bd0f49d084d19a`
- **Motivo**: aproximar a lógica do “Trend Surfer” (EMA34/89/200 + ATR 2.5) e sair do modo scalper perdedor.
- **Parâmetros alterados**:
  - `consensus_threshold`: `0.45`
  - `min_hold_bars`: `5`, `churn_penalty`: `0.01`
  - `atr_stop_mult`: `2.5`, `atr_trail_mult`: `2.5`
  - `max_long_entry_dist_fast_pct` / `max_short_entry_dist_fast_pct`: `0.0`
  - `trend_flip_penalty`: `0.05`
  - `living_cost_per_episode`: `10.0`
  - `override_long_gate`: `false`, `override_short_gate`: `false` (gates reativados)
- **Lógica no ambiente (`rl_env.py`)**:
  - Gate exige alinhamento das EMAs: long só entra com `ema_fast > ema_slow` e `slope_ref > 0`; short só com `ema_fast < ema_slow` e `slope_ref < 0`. O slope da `ref_ema` é calculado para evitar operar em 200 “flat”.
- **Demais campos mantidos**: metas mensais (tiers 2/4/5), stop de regime (zera long se exp_trend/ref < 0.4), saída fast<slow ativada, bear regime (block_long_in_bear=true, bear_consensus_short_threshold=0.3), bônus em degraus e pullback.
- **Próximo passo**: retreinar (`train.py`) e rodar `rl_backtest.py` para avaliar se o agente aprende a surfar tendências em vez de scalpar.

---

## 2025-12-05 — ema_only_rl_v26_zero_penalties

- **config_sha256**: `576df136f7201d1ac85c9afb66b7440d2a9de82b203c583b69ecded74a64dbb4`
- **Motivo**: remover todas as punições do reward para observar o comportamento puro das regras de tendência/stop.
- **Parâmetros zerados**:
  - `dd_penalty`, `churn_penalty`, `vol_penalty`, `gating_penalty`, `trend_flip_penalty`, `trend_exit_penalty`, `monthly_shortfall_penalty`
  - `living_cost_per_episode`: `0.0`
- **Demais campos mantidos**: filtros de tendência (fast/slow + slope_ref), stops ATR 2.5/2.5, metas mensais (só bônus), min_hold_bars=5, consenso 0.45.
- **Observação**: com zero penalidades, o reward passa a depender quase só do PnL e dos bônus (consenso, entry tiers, metas mensais). Requer retreino para avaliar efeito real.

---

## 2025-12-05 — ema_only_rl_v27_dynamic_sizing

- **config_sha256**: `576df136f7201d1ac85c9afb66b7440d2a9de82b203c583b69ecded74a64dbb4` (mesmo hash, lógica adicionada no código).
- **Motivo**: usar position sizing dinâmico baseado em % de risco por trade e stop ATR, em vez de lote fixo.
- **Lógica adicionada em `rl_env.py`**:
  - Novos campos em `RLConfig`:
    - `risk_per_trade_pct` (default 0, opcional; ex. 0.01 = 1% do capital).
    - `max_position_pct` (default 0.95) para limitar o tamanho da posição em relação ao capital.
  - Ao abrir posição:
    - Se `risk_per_trade_pct > 0` e há ATR e `atr_stop_mult > 0`, calcula stop_dist = atr_stop_mult * atr_value.
    - `risk_amount = equity * risk_per_trade_pct`.
    - `lot = (risk_amount / stop_dist) / price`, limitado por `max_position_pct * equity / price`.
    - Se não houver ATR ou risco configurado, usa `cfg.lot_size` como fallback.
  - Custos, PnL e MTM passam a usar `self.position_size` (por trade) em vez de `cfg.lot_size` fixo.
- **Parâmetros atuais relacionados**:
  - `risk_per_trade_pct`: ainda 0.0 no config (ativar definindo, por ex., 0.01).
  - `max_position_pct`: 0.95 (limite padrão).
- **Próximo passo**: definir `risk_per_trade_pct` no `config.json` (ex.: 0.01–0.02), retreinar e backtestar para ver o impacto do sizing dinâmico.

---

## 2025-12-05 — ema_only_rl_v28_sizing_on

- **config_sha256**: `e3572e886a17c34c9c6df71beb8665b597d5f079d09ca44683a5417daaacdfbd`
- **Motivo**: ativar o position sizing dinâmico para arriscar uma fração do capital por trade.
- **Parâmetros alterados**:
  - `risk_per_trade_pct`: `0.01` (1% do capital por trade)
  - `max_position_pct`: `0.95` (limite superior da posição em relação ao capital, para pagar taxas)
- **Demais campos**: mantidos (penalidades zeradas, stops 2.5/2.5, gates ativos com slope_ref, metas mensais só com bônus).
- **Próximo passo**: retreinar (`train.py`) e rodar `rl_backtest.py` para avaliar o impacto do sizing dinâmico.

---

## 2025-12-05 — ema_only_rl_v29_symmetric_trend_gate

- **config_sha256**: `e3572e886a17c34c9c6df71beb8665b597d5f079d09ca44683a5417daaacdfbd`
- **Motivo**: garantir simetria total nas checagens de tendência dos experts para long e short.
- **Alteração no ambiente (`rl_env.py`)**:
  - `trend_long_ok` agora exige `exp_trend >= 0.5` **e** `exp_ref >= 0.5`.
  - `trend_short_ok` agora exige `exp_trend < 0.5` **e** `exp_ref < 0.5`.
  - Ou seja, ambos os especialistas precisam concordar com a direção, tanto para compras quanto para vendas.
- **Demais parâmetros**: mantidos (sizing dinâmico, penalidades zeradas, stops 2.5/2.5, gates com slope_ref, metas mensais só bônus).
- **Próximo passo**: rodar backtest/treino e verificar se a simetria reduz bloqueios só de um lado e melhora a captura de shorts.

---

## 2025-12-05 — ema_only_rl_v30_experts_toggles_off

- **config_sha256**: `4368d021431782a36bfef518d956e2ca6a8a461dd3d80efd8fc19251740fb11a`
- **Motivo**: permitir ligar/desligar especialistas e consenso via config e, neste experimento, desligar todos.
- **Parâmetros alterados**:
  - Adicionado `experts_enabled` em `reward`:
    ```json
    {
      "exp_trend": false,
      "exp_ref": false,
      "exp_macd": false,
      "exp_slope": false,
      "exp_intraday_trend": false,
      "consensus": false
    }
    ```
- **Lógica no ambiente (`rl_env.py`)**:
  - Ignora consenso se `consensus=false` (usa 0.5 neutro).
  - Só checa `exp_trend`/`exp_ref` se estiverem habilitados; se off, não bloqueia por eles.
- **Próximo passo**: rodar backtest/treino com todos experts desligados para ver o comportamento “puro” das EMAs/ATR; depois, reativar seletivamente se necessário.

---

## 2025-12-05 — ema_only_rl_v31_cross_only_no_slope

- **config_sha256**: `1e35e6fecc1f41f6f5d178bbe15a8b14f67bbee28cbce8615628853e29270918`
- **Motivo**: focar só nas EMAs e nos cruzamentos, desligando consenso/experts e o filtro de inclinação da ref_ema.
- **Parâmetros/config**:
  - `experts_enabled`: todos `false` (exp_trend, exp_ref, exp_macd, exp_slope, exp_intraday_trend, consensus).
  - `ref_slope_enabled`: `false` — não usa slope da ref_ema para gate.
- **Lógica no ambiente (`rl_env.py`)**:
  - Gate passa a considerar apenas fast>slow (long) ou fast<slow (short) e ref_price (se ref bias ligado), sem checar experts/consenso/slope_ref.
- **Demais campos**: mantidos (sizing dinâmico 1%, penalidades zeradas, stops ATR 2.5/2.5, metas mensais só bônus, saída fast<slow).
- **Próximo passo**: treinar/backtestar para ver se o agente volta a operar com base só em cruzamentos de EMAs.

---

## 2025-12-05 — ema_only_rl_v32_cross_lookback_cfg

- **config_sha256**: `83644076672fffbcd4de949d9dbf45d0386c3c7b52496785e159169b1eb46285`
- **Motivo**: flexibilizar o gatilho de cruzamento fast/slow, permitindo desligar ou configurar a janela via config.
- **Parâmetros adicionados/alterados**:
  - `cross_lookback_bars`: `0` (0 = sem exigência de cruzamento recente; se >0, exige fast<=slow (long) ou >= (short) em até N barras passadas).
  - `ref_slope_enabled`: permanece `false`.
  - `experts_enabled`: segue com todos `false`.
- **Lógica no ambiente (`rl_env.py`)**:
  - Se `cross_lookback_bars` > 0, requer cruzamento fast/slow dentro da janela; se 0, o cruzamento recente é ignorado (basta fast>slow ou fast<slow).
- **Próximo passo**: rodar backtest com lookback 0 (cruzamento livre) e avaliar; depois, opcionalmente testar lookback curto (ex. 2) se voltar a overtradar.

---

## 2025-12-05 — ema_only_rl_v33_cross_bonus

- **config_sha256**: `3add1ba8b71c3828e6ea32357f1cbd7d6789f6928b989792f956866f110e5f7f`
- **Motivo**: recompensar cruzamentos recentes sem obrigar o gate a vetar entradas “antigas”.
- **Parâmetros adicionados**:
  - `cross_bonus_tiers`: `[[3, 2.0], [4, 1.0], [5, 0.5]]` — bônus no momento da entrada se o cruzamento ocorreu há até 3/4/5 barras.
  - `cross_lookback_bars`: `0` (continua sem obrigatoriedade de cruzamento recente).
  - `ref_slope_enabled`: `false`; `experts_enabled`: todos `false`.
- **Lógica no ambiente (`rl_env.py`)**:
  - Calcula se houve cruzamento fast/slow nas últimas `cross_lookback_bars`; se `0`, aceita qualquer fast>slow ou fast<slow.
  - Se encontrar cruzamento recente, aplica o maior bônus configurado (tiers) no reward de entrada.
- **Demais campos**: mantidos (sizing dinâmico 1%, penalidades zeradas, stops ATR 2.5/2.5, metas mensais só bônus, saída fast<slow).
- **Próximo passo**: treinar/backtestar para ver se o agente volta a operar com base só em cruzamentos de EMAs.

---

## 2025-12-05 — ema_only_rl_v34_experts_back_on

- **config_sha256**: `current_hash_placeholder`
- **Motivo**: reativar o gate de especialistas (Trend + Ref + Consensus) e o bônus de tendência, pois os resultados sem gate mostraram volatilidade excessiva (-9% jan, +27% jul). O objetivo é filtrar o ruído e estabilizar o retorno mensal em 2-5%.
- **Parâmetros alterados**:
  - `experts_master_enable`: `true` (reativado).
  - `experts_enabled`: `exp_trend=true`, `exp_ref=true`, `consensus=true`.
  - `trend_entry_bonus`: `0.2` (aumentado de 0.1 para reforçar o sinal).
  - `risk_per_trade_pct`: `0.01` (confirmado 1% para evitar drawdown excessivo).
- **Ajuste de Código (`rl_env.py`)**:
  - Corrigida a lógica do bônus "Short Level 2" para `fast < ref` (simetria correta com o texto do usuário), facilitando a detecção de reversões de baixa.
- **Próximo passo**: **Retreinar** obrigatoriamente para que o agente aprenda a nova política de risco e bônus.

---

## 2025-12-05 — ema_only_rl_v35_trend_surfer_tuning

- **config_sha256**: `4cc703bbb04b11a15068ca84d0d95f9f75b4a31cb7d8427742ca9e59d12c8797`
- **Motivo**: reduzir a volatilidade extrema (-17% / +40%) e estabilizar o retorno na faixa de 2-5% ao mês, ajustando o risco e exigindo maior consenso para filtrar o ruído em mercados laterais.
- **Parâmetros alterados**:
  - `consensus_threshold`: `0.5` (aumentado de 0.35 para 0.5) — exige metade dos experts a favor, filtrando sinais fracos.
  - `atr_stop_mult`: `2.5` (aumentado de 2.0) — stop inicial mais largo para evitar violinos em volatilidade normal.
  - `atr_trail_mult`: `1.5` (aumentado de 1.0) — trailing mais suave para capturar tendências longas sem sair prematuramente.
- **Demais campos mantidos**:
  - `risk_per_trade_pct`: `0.01` (mantido em 1% por enquanto, mas monitorando).
  - `vol_max_atr_rel`: `0.03`.
- **Próximo passo**: re-treinar e backtestar. Se a volatilidade persistir alta, o próximo ajuste será reduzir o risco para 0.5%.