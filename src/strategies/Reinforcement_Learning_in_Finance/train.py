from __future__ import annotations

import argparse
import logging
from typing import Optional

from .agent import AgentConfig, QLearningAgent, StateDiscretizer
from .data import load_price_history
from .env import TradingEnvironment
from .trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line flags into a structured namespace."""

    parser = argparse.ArgumentParser(
        description="Treina um agente Q-learning simples usando candles armazenados em cache.",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Par de negociação (default: BTCUSDT)")
    parser.add_argument("--interval", default="5m", help="Intervalo dos candles (default: 5m)")
    parser.add_argument("--start", help="Data inicial (ex.: 2021-01-01 00:00:00)", default=None)
    parser.add_argument("--end", help="Data final opcional", default=None)
    parser.add_argument("--episodes", type=int, default=20, help="Quantidade de episódios de treinamento")
    parser.add_argument("--window-size", type=int, default=8, help="Número de retornos usados na observação")
    parser.add_argument("--position-size", type=float, default=0.002, help="Quantidade de BTC por trade")
    parser.add_argument("--balance", type=float, default=1_000.0, help="Saldo inicial em USDT")
    parser.add_argument("--fee", type=float, default=0.001, help="Taxa percentual cobrada em cada operação")
    parser.add_argument("--threshold", type=float, default=0.001, help="Limite de discretização dos retornos")
    parser.add_argument("--max-steps", type=int, default=None, help="Limite opcional de passos por episódio")
    parser.add_argument("--render-episodes", type=int, default=1, help="Quantos episódios logar passo a passo")
    parser.add_argument(
        "--render-every",
        type=int,
        default=None,
        help="Após os episódios iniciais, loga um episódio a cada N iterações",
    )
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Taxa de aprendizado do Q-learning")
    parser.add_argument("--discount", type=float, default=0.95, help="Fator de desconto (gamma)")
    parser.add_argument("--epsilon", type=float, default=0.4, help="Probabilidade inicial de explorar ações")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Fator de decaimento do epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Limite inferior de epsilon")
    parser.add_argument("--seed", type=int, default=None, help="Semente para reprodutibilidade opcional")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nível de log (DEBUG, INFO, WARNING...)",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Set up console logging with a friendly format."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Entry-point used by ``python -m`` execution."""

    args = args or parse_args()
    configure_logging(args.log_level)

    if args.window_size > 12:
        logging.error(
            "Erro: --window-size=%d é muito grande e causaria um estouro de memória. Use um valor <= 12.",
            args.window_size,
        )
        return

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

    discretizer = StateDiscretizer(window_size=args.window_size, threshold=args.threshold)
    agent_config = AgentConfig(
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )
    agent = QLearningAgent(agent_config, discretizer, seed=args.seed)

    trainer_config = TrainingConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_episodes=args.render_episodes,
        render_every=args.render_every,
    )
    trainer = Trainer(env, agent, trainer_config, logger=logging.getLogger("trainer"))

    logging.info("Iniciando treinamento por %d episódios...", args.episodes)
    results = trainer.train()

    if not results:
        logging.warning("Nenhum episódio executado. Verifique os parâmetros.")
        return

    avg_reward = sum(result.total_reward for result in results) / len(results)
    best_equity = max(results, key=lambda result: result.final_equity)
    logging.info("")
    logging.info("===== Estatísticas finais =====")
    logging.info("Recompensa média: %.2f", avg_reward)
    logging.info(
        "Melhor equity: %.2f (episódio %d)",
        best_equity.final_equity,
        best_equity.index,
    )
    logging.info("Saldo final do último episódio: %.2f", results[-1].final_balance)
    logging.info("Epsilon final: %.3f", results[-1].epsilon)


if __name__ == "__main__":
    main()
