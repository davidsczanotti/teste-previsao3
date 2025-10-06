**Triple RSI – Guia de Uso Completo (Binance, cinco minutos)**

**1. Instalação**
- Requisitos do projeto:
  - Python 3.12 instalado no sistema.
  - Poetry instalado para gerenciar o ambiente e as dependências.
- Instalar as dependências do projeto dentro da pasta do repositório:
  - `poetry install`

**2. Visão Geral do Fluxo de Trabalho**
- Primeiro, otimizar os parâmetros com o Optuna para o ativo e para o intervalo desejados.
- Em seguida, o otimizador salva dois tipos de arquivos: os relatórios da otimização e uma “configuração ativa”.
- Por fim, executar o backtest e o modo ao vivo. Ambos buscam automaticamente a “configuração ativa” correspondente ao par de ativo e ao intervalo configurados no código.

**3. Otimizar os Parâmetros com Optuna**
- Exemplo de execução para o par BTCUSDT, intervalo de cinco minutos, período de noventa dias e trezentas tentativas de busca:
  - `poetry run python -m src.strategies.triple_rsi.optimize --ticker BTCUSDT --interval 5m --days 90 --trials 300 --train_frac 0.8 --lot_size 0.1`
- O que cada argumento faz:
  - `--ticker`: símbolo do ativo na Binance (por exemplo, BTCUSDT).
  - `--interval`: intervalo das velas (por exemplo, `5m` significa cinco minutos, `15m` significa quinze minutos).
  - `--days`: quantidade de dias de histórico a carregar para a otimização.
  - `--trials`: quantidade de tentativas de busca que o Optuna executará para encontrar a melhor combinação.
  - `--train_frac`: fração do histórico usada para treinar (o restante é usado para validação temporal).
  - `--lot_size`: tamanho do lote para o backtest durante a otimização (apenas para simulação; não é aplicado automaticamente ao modo ao vivo).
- Resultado da otimização:
  - Ao final, o comando imprime algo como: “Relatórios salvos em: reports/…json e …md” e “Config ativa atualizada: reports/active/…json”.
  - Arquivos gerados no diretório `reports/`:
    - Arquivo JSON com os dados estruturados: `triple_rsi_optuna_<TICKER>_<INTERVALO>_<DATA_HORA>.json`.
    - Arquivo Markdown com um resumo legível por humanos: `triple_rsi_optuna_<TICKER>_<INTERVALO>_<DATA_HORA>.md`.
  - Além disso, é escrita uma “configuração ativa” em `reports/active/<TICKER>_<INTERVALO>.json` (por exemplo, `reports/active/BTCUSDT_5m.json`).

**4. Aplicação Automática dos Melhores Parâmetros**
- O projeto contém um módulo de configuração compartilhada que grava e carrega os parâmetros otimizados.
- Após a otimização, os modos de backtest e ao vivo procuram automaticamente a “configuração ativa” correspondente ao par e ao intervalo definidos no código principal (padrão: BTCUSDT e cinco minutos).
- Como está implementado:
  - O modo ao vivo tenta carregar `reports/active/BTCUSDT_5m.json` ao iniciar.
  - O backtest tenta carregar `reports/active/BTCUSDT_5m.json` ao iniciar.
- Se você otimizou outro ativo ou outro intervalo, altere os valores padrão diretamente nos arquivos:
  - Live: `src/strategies/triple_rsi/live.py`
  - Backtest: `src/strategies/triple_rsi/backtest.py`

**5. Executar o Backtest (usando a configuração ativa, quando existir)**
- Comando:
  - `poetry run python -m src.strategies.triple_rsi.backtest`
- O backtest tentará carregar automaticamente a “configuração ativa” para o par e o intervalo definidos nas variáveis padrão do arquivo. Caso não encontre, ele executará cenários padrão.
- Importante: execute sempre com a opção `-m` (`python -m src.strategies.triple_rsi.backtest`). Isso garante que os “imports relativos” funcionem corretamente dentro do pacote.

**6. Executar o Modo ao Vivo (usando a configuração ativa, quando existir)**
- Comando:
  - `poetry run python -m src.strategies.triple_rsi.live`
- Comportamento:
  - Se a “configuração ativa” para o par e intervalo padrão existir, o modo ao vivo a utilizará automaticamente.
  - Se não existir, o modo ao vivo usará parâmetros padrão internos.
- Dica importante sobre tamanho do lote no modo ao vivo:
  - O tamanho do lote definido no backtest não é aplicado automaticamente no modo ao vivo, por segurança. Ajuste `initial_capital` e `lot_size` na chamada do modo ao vivo conforme a sua necessidade, diretamente no arquivo `src/strategies/triple_rsi/live.py`.

**7. Personalizações Úteis do Otimizador**
- Forçar a estratégia invertida (ou não) durante a otimização:
  - Acrescente `--fix_invert true` para forçar a estratégia invertida, ou `--fix_invert false` para forçar a estratégia normal.
- Aumentar o horizonte de dados e a quantidade de tentativas de busca:
  - Aumente `--days` para carregar mais histórico e `--trials` para explorar mais combinações. Isso deixa a busca mais lenta, mas pode encontrar parâmetros melhores.

**8. Solução de Problemas**
- Mensagem “attempted relative import with no known parent package” ao executar arquivos diretamente:
  - Sempre execute os módulos com a opção `-m`, por exemplo: `poetry run python -m src.strategies.triple_rsi.backtest` e `poetry run python -m src.strategies.triple_rsi.live`.
- Mensagem “Nenhum dado retornado da Binance”:
  - Verifique a sua conexão de rede, o símbolo do ativo (`--ticker`) e o intervalo (`--interval`).
- Avisos de segurança “InsecureRequestWarning” durante requisições HTTPS:
  - Esses avisos ocorrem porque a verificação de certificados foi desativada no cliente da Binance. Caso deseje, podemos habilitar a verificação e suprimir os avisos com ajustes no código do cliente.

**9. Onde os Arquivos Gerados São Salvos**
- Relatórios de otimização:
  - `reports/triple_rsi_optuna_<TICKER>_<INTERVALO>_<DATA_HORA>.json`
  - `reports/triple_rsi_optuna_<TICKER>_<INTERVALO>_<DATA_HORA>.md`
- Configuração ativa (consumida automaticamente pelo backtest e pelo modo ao vivo):
  - `reports/active/<TICKER>_<INTERVALO>.json`
- Gráficos salvos pelo modo ao vivo:
  - Pasta `live_charts/`, com um arquivo de imagem por execução.

**Estratégia Estocástico (25/75) — BTCUSDT (novo)**
- Foram adicionados módulos para a estratégia baseada no Estocástico Lento com entradas/saídas por cruzamentos das linhas 25/75, conforme o vídeo referenciado.
- Backtest:
  - `poetry run python -m src.strategies.stochastic.backtest`
  - Padrões: `BTCUSDT`, intervalo `5m` (60 dias), `k=9`, `oversold=25`, `overbought=75`, `lot_size=0.001` no exemplo do arquivo.
- Live (gráfico + sinais):
  - `poetry run python -m src.strategies.stochastic.live`
  - Padrões: `BTCUSDT`, `5m`, `k=9`, níveis 25/75. Salva imagens em `live_charts/`.
- Observação:
  - A estratégia do Estocástico não depende dos módulos de otimização do Triple RSI e mantém o ticker BTCUSDT da Binance.

**Otimização do Estocástico (Optuna)**
- Comando de exemplo (BTCUSDT, 5m, 60 dias, 200 trials):
  - `poetry run python -m src.strategies.stochastic.optimize --ticker BTCUSDT --interval 5m --days 60 --trials 200 --train_frac 0.8 --lot_size 0.001 --fee_rate 0.001`
- Saída:
  - Relatórios em `reports/` (JSON/Markdown) com melhores parâmetros e métricas de treino/validação.
  - Configuração ativa salva em `reports/active/STOCH_BTCUSDT_5m.json` para consumo automático pelo backtest/live do Estocástico.
- Depois da otimização:
  - Backtest consumindo config ativa: `poetry run python -m src.strategies.stochastic.backtest`
  - Live consumindo config ativa: `poetry run python -m src.strategies.stochastic.live`
 - Observações:
   - `--fee_rate` modela taxa por lado (ex.: 0.001 = 0,1% por lado). O backtest também imprime métricas adicionais (profit factor, MDD, P&L médio por trade).

**Varredura por Intervalos (Stochastic)**
- Rodar otimização em múltiplos intervalos e comparar resultados:
  - `poetry run python -m src.strategies.stochastic.sweep --ticker BTCUSDT --intervals 5m,15m,1h --days 60 --trials 200 --train_frac 0.8 --lot_size 0.001 --fee_rate 0.0005`
- Saída:
  - Resumo em `reports/stoch_sweep_<TICKER>_<TIMESTAMP>.md` e `.json`, com melhores parâmetros por intervalo e indicação do melhor por PnL de validação.

**Estratégia Donchian Breakout (novo)**
- Otimização (com ATR trailing e filtro EMA opcional):
  - `poetry run python -m src.strategies.donchian.optimize --ticker BTCUSDT --interval 15m --days 120 --trials 250 --train_frac 0.8 --lot_size 0.001 --fee_rate 0.0005`
- Backtest (usando config ativa gerada):
  - `poetry run python -m src.strategies.donchian.backtest`
- Parâmetros principais otimizáveis:
  - `window_high/window_low`, `atr_period`, `atr_mult`, `use_ema/ema_period`, `allow_short`

**Estrutura de Pastas (organização por estratégia)**
- `src/strategies/triple_rsi/`: `optimize.py`, `backtest.py`, `live.py`, `config.py`
- `src/strategies/stochastic/`: `optimize.py`, `backtest.py`, `live.py`, `config.py`
- Compartilhados: `src/binance_client.py` (cliente de dados da Binance)
