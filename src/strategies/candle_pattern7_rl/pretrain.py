from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np

from .config import Candle7RlConfig
from .env import Candle7Env
from .train import _apply_grads as _apply, _mlp_backward, _mlp_forward as _forward, _mlp_init


def build_expert_dataset(env: Candle7Env) -> dict[str, np.ndarray]:
    """Gera um dataset de observações e ações expert a partir da heurística do ambiente."""
    env._ensure_data()
    assert env._base_features is not None and env._heuristic_actions is not None

    obs_list = []
    act_list = []

    # Itera sobre todos os passos possíveis do ambiente
    for i in range(len(env._base_features)):
        # Simula o estado da observação (sem posição, para focar no sinal de entrada)
        base_obs = env._base_features[i]
        pos_flags = np.array([0.0, 0.0], dtype=np.float32)  # Assume flat
        obs = np.concatenate([base_obs, pos_flags])

        action = env._heuristic_actions[i]

        # Coleta apenas os momentos em que a heurística dá um sinal de entrada
        if action in [1, 3]:  # Open Long or Open Short
            obs_list.append(obs)
            act_list.append(action)
        # Adiciona alguns exemplos de "Hold" para balancear
        elif action == 0 and i % 20 == 0:
            obs_list.append(obs)
            act_list.append(action)

    return {"X": np.stack(obs_list), "y": np.array(act_list, dtype=np.int64)}


def pretrain(cfg: Candle7RlConfig, epochs: int = 10, lr: float = 1e-3):
    """Pré-treina o modelo usando Behavioral Cloning na heurística."""
    print("Iniciando pré-treinamento...")
    env = Candle7Env(symbol=cfg.ticker, interval=cfg.interval, days=cfg.days, long_only=cfg.long_only)
    dataset = build_expert_dataset(env)
    X, y = dataset["X"], dataset["y"]

    print(f"Dataset de expert criado com {len(X)} amostras.")
    if len(X) == 0:
        print("Nenhuma amostra de expert gerada. Abortando pré-treinamento.")
        return

    in_size = env.observation_size
    out_size = env.action_size
    hidden = cfg.hidden_size
    params = _mlp_init(in_size, hidden, out_size)

    batch_size = 128
    for epoch in range(epochs):
        total_loss = 0.0
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]

            # Inicializa acumuladores para gradientes
            gW1 = np.zeros_like(params[0])
            gb1 = np.zeros_like(params[1])
            gW2 = np.zeros_like(params[2])
            gb2 = np.zeros_like(params[3])
            # Cabeça de valor não é usada no pré-treino supervisionado, mas mantemos por compatibilidade
            gWv = np.zeros_like(params[4])
            gbv = np.zeros_like(params[5])

            for obs, target_action in zip(batch_X, batch_y):
                probs, _, cache = _forward(params, obs)
                loss = -np.log(probs[target_action] + 1e-8)
                total_loss += loss

                # Usamos o backward do MLP padrão com advantage = -1.0 (gradiente de CE) e sem erro de valor
                dW1_bc, db1_bc, dW2_bc, db2_bc, dWv_bc, dbv_bc = _mlp_backward(
                    params, cache, int(target_action), float(-1.0), 0.0
                )
                gW1 += dW1_bc
                gb1 += db1_bc
                gW2 += dW2_bc
                gb2 += db2_bc
                # Ignora updates de valor no pré-treino

            # Aplica gradientes (mantém cabeças de valor inalteradas)
            params = _apply(params, (gW1, gb1, gW2, gb2, gWv, gbv), lr, clip=1.0)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X):.4f}")

    # Salva o modelo pré-treinado
    out_dir = os.path.join("reports", "agents", "candle_pattern7_rl")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ppo_{cfg.ticker}_{cfg.interval}.npz")
    np.savez(
        out_path,
        W1=params[0],
        b1=params[1],
        W2=params[2],
        b2=params[3],
        Wv=params[4],
        bv=params[5],
        config=np.array([str(asdict(cfg))], dtype=object),
    )
    print(f"Modelo pré-treinado salvo em: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-treina o agente Candle7 RL")
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--long_only", action="store_true")
    args = parser.parse_args()

    config = Candle7RlConfig(
        ticker=args.ticker, interval=args.interval, days=args.days, long_only=args.long_only, hidden_size=128
    )
    pretrain(config)
