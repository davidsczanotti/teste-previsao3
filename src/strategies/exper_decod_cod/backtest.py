import sys
import os
import numpy as np
import json

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.data_loader import load_data
from src.strategies.exper_decod_cod.env import TradingEnv
from src.strategies.exper_decod_cod.agent import Agent

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def main():
    """
    Função principal para executar um backtest simples com um agente aleatório.
    """
    # Carrega as configurações do arquivo config.json
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    env_config = config['env']
    agent_config = config['agent']
    training_config = config['training']

    SYMBOL = env_config['symbol']
    TIMEFRAME = env_config['timeframe']
    DAYS = env_config['days']

    print(f"Carregando dados para {SYMBOL}/{TIMEFRAME} dos últimos {DAYS} dias...")
    try:
        df = load_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=DAYS, use_cache_only=True)
        print("Dados carregados com sucesso.")
    except ValueError as e:
        print(f"Erro ao carregar dados: {e}")
        print(f"Por favor, execute o script 'populate_cache.py' primeiro:")
        print(f"poetry run python -m scripts.populate_cache {SYMBOL} {TIMEFRAME}")
        return

    env = TradingEnv(
        df,
        initial_balance=env_config['initial_balance'],
        lot_size=env_config['lot_size']
    )
    observation = env.reset()
    
    state_size = observation.shape[0]
    action_size = len(env.action_space)
    
    agent = Agent(
        state_size,
        action_size,
        buffer_size=agent_config['buffer_size'],
        learning_rate=agent_config['learning_rate'],
        gamma=agent_config['gamma'],
        batch_size=agent_config['batch_size'],
        epsilon=agent_config['epsilon'],
        epsilon_decay=agent_config['epsilon_decay'],
        epsilon_min=agent_config['epsilon_min']
    )

    print("\n--- Iniciando teste do ambiente com Agente aleatório ---")
    print(f"Saldo inicial: ${env.initial_balance:.2f}")
    print(f"Observação inicial (preço): {observation}")

    for step in range(10):
        action = agent.act(observation)
        
        obs, reward, done, info = env.step(action)

        agent.remember(observation, action, reward, obs, done)
        agent.learn() # O agente aprende com a experiência

        action_map = {0: "Aguardar", 1: "Comprar", 2: "Vender"}
        print(f"\nPasso: {step + 1}")
        print(f"Ação Tomada: {action_map[action]}")
        print(f"Nova Observação (preço): {obs}")
        print(f"Recompensa: {reward}")
        print(f"Patrimônio Líquido: ${env.net_worth:.2f}")

        observation = obs

        if done:
            print("\n--- Fim do Episódio ---")
            break
    
    print(f"\n--- Fim do teste ---")
    print(f"Tamanho da memória do agente: {len(agent.memory)}")

if __name__ == "__main__":
    main()
