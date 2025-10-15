from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .env import Candle7Env
from . import ablation as abl


def load_agent_policy(model_path: str):
    """Carrega a política salvo (.npz REINFORCE NumPy ou .pt PPO PyTorch)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
    policy_fn, policy_type, cfg = abl.load_policy(model_path)
    return policy_fn, policy_type, cfg


def run_visual_backtest(policy_fn, env: Candle7Env, max_steps: int | None = None):
    """
    Executa um episódio usando a política do agente e coleta dados para visualização.
    """
    obs = env.reset()
    done = False
    steps = 0
    history = []

    while not done:
        action_probs = policy_fn(obs)
        action = int(np.argmax(action_probs))  # Execução greedy

        step_result = env.step(action)

        # Armazena o estado ANTES da ação ser totalmente processada no próximo candle
        current_info = {
            "step": steps,
            "date": env._df["Date"].iloc[env._start_idx + env._i - 1],
            "close": env._closes[env._i - 1],
            "action": action,
            "position": env._pos,
            "reward": step_result.reward,
            "trade_action": None,
        }

        if "trade" in step_result.info:
            current_info["trade_action"] = step_result.info["trade"]

        history.append(current_info)

        # Se um trade foi fechado (posição voltou a zero), a recompensa é o PnL.
        if "trade" in step_result.info and env._pos == 0:
            history[-1]["pnl"] = step_result.reward

        obs = step_result.obs
        done = step_result.done
        steps += 1
        if max_steps and steps >= max_steps:
            break

    return pd.DataFrame(history)


def print_summary(df_history: pd.DataFrame):
    """Imprime um resumo das métricas do backtest."""
    print("\n--- Resumo do Backtest Visual ---")
    total_reward = df_history["reward"].sum()
    print(f"Recompensa Total do Episódio: {total_reward:.2f}")

    # Trades são onde 'trade_action' não é nulo
    trades_df = df_history.dropna(subset=["trade_action"])
    num_trade_actions = len(trades_df)
    print(f"Total de Ações de Trade (Entradas/Saídas): {num_trade_actions}")

    # Trades fechados são aqueles que têm um valor de PnL não nulo
    if "pnl" not in df_history.columns:
        print("Nenhum trade foi fechado durante o período.")
        return

    closed_trades_df = df_history.dropna(subset=["pnl"])
    num_closed_trades = len(closed_trades_df)

    if num_closed_trades > 0:
        wins = closed_trades_df[closed_trades_df["pnl"] > 0]
        num_wins = len(wins)
        win_rate = (num_wins / num_closed_trades) * 100
        total_pnl = closed_trades_df["pnl"].sum()
        total_loss = abs(closed_trades_df[closed_trades_df["pnl"] < 0]["pnl"].sum())
        profit_factor = wins["pnl"].sum() / total_loss if total_loss > 0 else float("inf")

        print(f"P&L Total (Trades Fechados): {total_pnl:.2f}")
        print(f"Total de Trades Fechados: {num_closed_trades}")
        print(f"Taxa de Acerto: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
    else:
        print("Nenhum trade foi fechado durante o período.")


def plot_backtest(df_history: pd.DataFrame, ticker: str, interval: str):
    """
    Plota o gráfico de preços com os sinais de compra e venda.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(18, 9))

    # Plota o preço de fechamento
    ax.plot(df_history["date"], df_history["close"], label="Close Price", color="gray", alpha=0.8, zorder=1)

    # Marca os trades
    buy_signals = df_history[df_history["trade_action"] == "BUY"]
    sell_signals = df_history[df_history["trade_action"] == "SELL"]
    buy_to_cover_signals = df_history[df_history["trade_action"] == "BUY_TO_COVER"]
    sell_short_signals = df_history[df_history["trade_action"] == "SELL_SHORT"]

    ax.scatter(buy_signals["date"], buy_signals["close"], label="Buy", marker="^", color="green", s=100, zorder=2)
    ax.scatter(sell_signals["date"], sell_signals["close"], label="Sell", marker="v", color="red", s=100, zorder=2)
    ax.scatter(
        sell_short_signals["date"],
        sell_short_signals["close"],
        label="Sell Short",
        marker="v",
        color="purple",
        s=100,
        zorder=2,
    )
    ax.scatter(
        buy_to_cover_signals["date"],
        buy_to_cover_signals["close"],
        label="Buy to Cover",
        marker="^",
        color="orange",
        s=100,
        zorder=2,
    )

    ax.set_title(f"Backtest Visual do Agente RL - {ticker} ({interval})")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço")
    ax.legend()

    # Salva o gráfico
    chart_dir = os.path.join("reports", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, f"candle_pattern7_rl_backtest_{ticker}_{interval}.png")
    plt.savefig(chart_path)
    print(f"Gráfico do backtest salvo em: {chart_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest visual para o agente Candle7 RL")
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=90, help="Período para o backtest visual")
    parser.add_argument("--model", help="Caminho para o arquivo do agente (.npz ou .pt)", required=True)
    # Overrides seguros (apenas se compatíveis com o modelo salvo); caso contrário, ignorados
    parser.add_argument("--include_mtf", choices=["auto", "true", "false"], default="auto",
                        help="Força MTF (se compatível com o modelo); padrão: auto (usa cfg salvo)")
    parser.add_argument("--mtf_timeframes", type=str, default=None,
                        help="Lista separada por vírgula para MTF (ex: '1h,4h'); só aplica se compatível")
    parser.add_argument("--include_regimes", choices=["auto", "true", "false"], default="auto",
                        help="Força regimes (se compatível com o modelo); padrão: auto (usa cfg salvo)")
    args = parser.parse_args()

    # 1. Carrega a política do agente
    policy_fn, policy_type, cfg = load_agent_policy(args.model)

    # 2. Cria o ambiente para o período de backtest
    obs_format = "structured" if policy_type in ("transformer", "lstm") else "flat"
    # Resolve overrides com segurança de formato
    cfg_include_mtf = bool(cfg.get("include_mtf", False))
    cfg_mtf_tfs = tuple(cfg.get("mtf_timeframes", ("1h", "4h")))
    cfg_include_reg = bool(cfg.get("include_regime_features", False))

    def resolve_bool(opt: str, cfg_val: bool, name: str) -> bool:
        if opt == "auto":
            return cfg_val
        want = (opt == "true")
        if want != cfg_val:
            print(f"WARN: Ignorando --{name}={opt} por incompatibilidade com o modelo salvo (usando {cfg_val}).")
            return cfg_val
        return want

    use_mtf = resolve_bool(args.include_mtf, cfg_include_mtf, "include_mtf")
    use_reg = resolve_bool(args.include_regimes, cfg_include_reg, "include_regimes")

    if args.mtf_timeframes and tuple([s.strip() for s in args.mtf_timeframes.split(',') if s.strip()]) != cfg_mtf_tfs:
        print("WARN: Ignorando --mtf_timeframes (incompatível com o modelo salvo).")

    env = Candle7Env(
        symbol=args.ticker,
        interval=args.interval,
        days=args.days,
        episode_len=None,  # Roda o período todo
        random_start=False,  # Começa do início
        obs_format=obs_format,
        include_mtf=use_mtf,
        mtf_timeframes=cfg_mtf_tfs,
        include_regime_features=use_reg,
        long_only=bool(cfg.get("long_only", False)),
    )

    # 3. Executa o backtest
    print("Executando backtest visual...")
    backtest_df = run_visual_backtest(policy_fn, env)

    # 4. Plota os resultados
    print("Gerando gráfico...")
    plot_backtest(backtest_df, args.ticker, args.interval)

    # 5. Imprime o resumo
    print_summary(backtest_df)

    print("Backtest visual concluído.")
