from __future__ import annotations

import argparse
import logging
from typing import Optional

from ..Reinforcement_Learning_in_Finance.data import load_price_history
from ..Reinforcement_Learning_in_Finance.env import TradingEnvironment
from .agent import AgentConfig, DQNAgent
from .trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line flags into a structured namespace."""

    parser = argparse.ArgumentParser(description="Treina um agente DQN usando candles armazenados em cache.")
    # Parâmetros do ambiente
    parser.add_argument("--symbol", default="BTCUSDT", help="Par de negociação")
    parser.add_argument("--interval", default="5m", help="Intervalo dos candles")
    parser.add_argument("--start", help="Data inicial (ex.: 2021-01-01)", default=None)
    parser.add_argument("--end", help="Data final opcional", default=None)
    parser.add_argument("--window-size", type=int, default=24, help="Número de retornos na observação")
    parser.add_argument("--position-size", type=float, default=0.002, help="Quantidade de BTC por trade")
    parser.add_argument("--balance", type=float, default=1_000.0, help="Saldo inicial em USDT")
    parser.add_argument("--fee", type=float, default=0.001, help="Taxa percentual por operação")

    # Parâmetros do treinamento
    parser.add_argument("--episodes", type=int, default=50, help="Quantidade de episódios de treinamento")
    parser.add_argument("--max-steps", type=int, default=None, help="Limite opcional de passos por episódio")
    parser.add_argument("--render-episodes", type=int, default=2, help="Quantos episódios logar passo a passo")
    parser.add_argument("--render-every", type=int, default=10, help="Loga um episódio a cada N iterações")
    parser.add_argument("--seed", type=int, default=None, help="Semente para reprodutibilidade")
    parser.add_argument("--log-level", default="INFO", help="Nível de log (DEBUG, INFO, ...)")

    # Hiperparâmetros do Agente DQN
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Taxa de aprendizado da rede")
    parser.add_argument("--discount", type=float, default=0.95, help="Fator de desconto (gamma)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Probabilidade inicial de explorar")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Fator de decaimento do epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Limite inferior de epsilon")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do lote para aprendizado")
    parser.add_argument("--buffer-size", type=int, default=10_000, help="Tamanho do replay buffer")
    parser.add_argument("--hidden-size", type=int, default=32, help="Tamanho da camada oculta da rede")
    parser.add_argument(
        "--target-update-freq", type=int, default=10, help="Frequência (em episódios) para atualizar a target network"
    )

    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Entry-point used by ``python -m`` execution."""
    args = args or parse_args()
    configure_logging(args.log_level)

    logging.info("Carregando candles de %s %s...", args.symbol, args.interval)
    data = load_price_history(args.symbol, args.interval, start=args.start, end=args.end)

    env = TradingEnvironment(
        data=data,
        window_size=args.window_size,
        position_size=args.position_size,
        starting_balance=args.balance,
        trading_fee=args.fee,
        seed=args.seed,
    )

    agent_config = AgentConfig(
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        hidden_size=args.hidden_size,
        target_update_freq=args.target_update_freq,
    )
    agent = DQNAgent(agent_config, input_size=args.window_size, seed=args.seed)

    trainer_config = TrainingConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_episodes=args.render_episodes,
        render_every=args.render_every,
    )
    trainer = Trainer(env, agent, trainer_config, logger=logging.getLogger("trainer"))

    logging.info("Iniciando treinamento DQN por %d episódios...", args.episodes)
    results = trainer.train()

    if not results:
        logging.warning("Nenhum episódio executado. Verifique os parâmetros.")
        return

    avg_reward = sum(result.total_reward for result in results) / len(results)
    best_equity_result = max(results, key=lambda result: result.final_equity)
    logging.info("")
    logging.info("===== Estatísticas finais =====")
    logging.info("Recompensa média: %.2f", avg_reward)
    logging.info(
        "Melhor equity: %.2f (episódio %d)",
        best_equity_result.final_equity,
        best_equity_result.index,
    )
    logging.info("Saldo final do último episódio: %.2f", results[-1].final_balance)
    logging.info("Epsilon final: %.3f", results[-1].epsilon)


if __name__ == "__main__":
    main()
