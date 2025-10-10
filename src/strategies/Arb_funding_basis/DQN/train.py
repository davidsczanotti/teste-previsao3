import argparse
from collections import deque

import pandas as pd
from .agent import DQNAgent
from .environment import ArbFundingEnv
from ..arb_funding_basis import load_data


def train(
    df_train: pd.DataFrame,
    n_episodes=2000,
    max_t=10000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
):
    """
    Treina o agente DQN.

    Args:
        df_train (pd.DataFrame): Dados de treino.
        n_episodes (int): Número de episódios de treinamento.
        max_t (int): Número máximo de passos por episódio.
        eps_start (float): Valor inicial de epsilon (exploração).
        eps_end (float): Valor mínimo de epsilon.
        eps_decay (float): Fator de decaimento de epsilon.
    """
    env = ArbFundingEnv(df_train)
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
            f"\rEpisódio {i_episode}\tP&L Médio (100 ep): ${sum(scores_window)/len(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisódio {i_episode}\tP&L Médio (100 ep): ${sum(scores_window)/len(scores_window):.2f}")

    # Salva o modelo treinado
    agent.qnetwork_local.save("arb_funding_dqn.keras")
    print("\nModelo treinado salvo em 'arb_funding_dqn.keras'")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Treinar agente DQN para arbitragem de Funding Rate.")
    parser.add_argument("--start", default="2020-01-01", help="Data de início dos dados (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-01-01", help="Data de fim dos dados (YYYY-MM-DD)")
    parser.add_argument("--episodes", type=int, default=500, help="Número de episódios de treinamento")
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
    train(df, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
