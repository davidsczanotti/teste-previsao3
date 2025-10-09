Este diretório contém um playground didático de aprendizado por reforço focado em um único agente que aprende a comprar, esperar ou vender BTCUSDT usando apenas candles do banco local `data/klines_cache.db`. O objetivo é entender os fundamentos sem depender de indicadores ou redes neurais complexas.

**Antes de Começar**
- Popule ou atualize o cache: `poetry run python -m scripts.populate_cache BTCUSDT 5m`
- Execute os comandos a partir da raiz do projeto para que o caminho do banco funcione corretamente
- Verifique se há dados suficientes; o agente precisa de pelo menos algumas centenas de candles para treinar com ciclos completos de recompensa/punição

**Arquitetura**
- `data.py`: carrega candles do cache e calcula o retorno simples (`close.pct_change()`), servindo como única fonte de features
- `env.py`: implementa o ambiente `TradingEnvironment`, com ações `HOLD`, `BUY` e `SELL`, tamanho fixo de posição (0.1 BTC), saldo inicial de 1000 USDT, punição por zerar o saldo e recompensa baseada na variação de equity
- `agent.py`: define `StateDiscretizer` (quantiza retornos em três faixas) e `QLearningAgent`, um Q-Learning simples com epsilon-greedy
- `trainer.py`: orquestra episódios, gera logs passo a passo e calcula resumos por episódio
- `train.py`: script CLI (`python -m src.strategies.Reinforcement_Learning_in_Finance.train`) que conecta tudo, expondo hiperparâmetros e opções de logging

**Como Treinar**
- Comando padrão: `poetry run python -m src.strategies.Reinforcement_Learning_in_Finance.train --episodes 25 --render-episodes 2`
- Flags úteis: `--start`/`--end` (restrição de período), `--window-size` (tamanho da janela), `--learning-rate`/`--discount`/`--epsilon`/`--epsilon-decay`/`--epsilon-min` (hiperparâmetros do Q-Learning), `--render-every` (define a frequência de episódios exibidos após os iniciais)
- Use `--log-level DEBUG` para investigar o fluxo completo

**Leitura dos Logs**
- Cada passo mostra: número do passo, ação escolhida, preço de execução (close do candle atual), recompensa (variação de equity desde o passo anterior), saldo e equity atualizados, além de eventos como “Compra executada”
- Ao final de cada episódio o resumo reporta recompensa acumulada, saldo/equity final, quantidade de trades, taxa de acerto e valor atual de epsilon
- Estatísticas agregadas de todos os episódios são impressas ao final (recompensa média, melhor equity e saldo do último episódio)

**Próximos Passos**
- Experimente janelas maiores (`--window-size 12`) ou limites diferentes para discretização (`--threshold 0.0005`)
- Ajuste `--position-size` e `--fee` para simular cenários variados
- Use os métodos `q_values_for` do agente ou `trade_log` do ambiente em notebooks para inspecionar decisões passo a passo
- Evolua o agente substituindo a discretização por modelos contínuos (ex.: rede neural leve) quando se sentir confortável com a base
