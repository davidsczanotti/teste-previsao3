import argparse
import os
from collections import deque

import numpy as np
import pandas as pd
from .agent import DQNAgent
from .environment import ArbFundingEnv
from ..arb_funding_basis import load_data


def train(
    df_train: pd.DataFrame,
    n_episodes=2000,
    max_t=100000,  # Aumentado para permitir episódios mais longos
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    leverage=2.0,
    initial_capital=1000.0,
):
    """
    Treina o agente DQN.
    """
    env = ArbFundingEnv(df_train, leverage=leverage, initial_capital=initial_capital)
    agent = DQNAgent(state_size=env.observation_space_dim, action_size=env.action_space_dim, seed=0)

    scores = []  # Lista para armazenar os PnLs de cada episódio
    scores_window = deque(maxlen=100)  # Média dos últimos 100 PnLs
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(info.get("pnl", 0))
        scores.append(info.get("pnl", 0))
        eps = max(eps_end, eps_decay * eps)  # Decai o epsilon

        print(
            f"\rEpisódio {i_episode}\tP&L Médio (100 ep): ${np.mean(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisódio {i_episode}\tP&L Médio (100 ep): ${np.mean(scores_window):.2f}")

    # Salva o modelo treinado
    model_dir = "reports/agents/arb_funding_basis"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "arb_funding_dqn.keras")
    agent.qnetwork_local.save(model_path)
    print(f"\nModelo treinado salvo em '{model_path}'")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Treinar agente DQN para arbitragem de Funding Rate.")
    parser.add_argument("--start", default="2020-01-01", help="Data de início dos dados (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-01-01", help="Data de fim dos dados (YYYY-MM-DD)")
    parser.add_argument("--episodes", type=int, default=500, help="Número de episódios de treinamento")
    parser.add_argument("--leverage", type=float, default=2.0, help="Alavancagem da posição")
    args = parser.parse_args()

    print("Carregando dados de treinamento...")
    # Usamos a função do backtest original para carregar os dados
    df, _ = load_data(
        symbol="BTCUSDT",
        start_date_str=args.start,
        end_date_str=args.end,
    )
    df = df.reset_index()

    print(f"Iniciando treinamento com {args.episodes} episódios...")
    train(df, n_episodes=args.episodes, leverage=args.leverage)


if __name__ == "__main__":
    main()
