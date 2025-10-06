import argparse
from datetime import datetime, timedelta, UTC
from pathlib import Path
import json

import optuna
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt

# Evita abrir janelas de gráfico durante a otimização (headless)
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
# Garante que chamadas a plt.show() (dentro do backtest) não bloqueiem
plt.show = lambda *args, **kwargs: None

from ...binance_client import get_historical_klines
from .backtest import backtest_triple_rsi
from .config import save_active_config


def load_data(ticker: str, interval: str, days: int) -> pd.DataFrame:
    start_dt = datetime.now(UTC) - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(ticker, interval, start_str)
    if df.empty:
        raise RuntimeError("Nenhum dado retornado da Binance.")
    # Ordena e ajusta atributos
    df = df.sort_values("Date").reset_index(drop=True)
    df.attrs["ticker"] = ticker
    return df


def make_objective(df_train: pd.DataFrame, lot_size: float, day_trade: bool, fix_invert: str | None = None):
    def objective(trial: optuna.Trial) -> float:
        # Espaço de busca com restrições lógicas
        short = trial.suggest_int("short", 5, 35)
        med = trial.suggest_int("med", short + 2, 80)
        long = trial.suggest_int("long", med + 2, 140)

        buy_entry = trial.suggest_int("buy_entry", 20, 40)
        sell_entry = trial.suggest_int("sell_entry", 60, 80)
        buy_exit = trial.suggest_int("buy_exit", buy_entry, 90)
        sell_exit = trial.suggest_int("sell_exit", 10, sell_entry)

        if fix_invert is None:
            invert = trial.suggest_categorical("invert", [False, True])
        else:
            invert = True if str(fix_invert).lower() == "true" else False

        # Roda o backtest no conjunto de treino
        try:
            trades, pnl = backtest_triple_rsi(
                df_train.copy(),
                short_period=short,
                med_period=med,
                long_period=long,
                buy_entry=buy_entry,
                sell_entry=sell_entry,
                buy_exit=buy_exit,
                sell_exit=sell_exit,
                invert=invert,
                lot_size=lot_size,
                day_trade=day_trade,
            )
        except Exception as e:
            # Penaliza configurações inválidas
            trial.set_user_attr("error", str(e))
            return -1e12

        # Opcional: penalização por quantidade de trades para reduzir overfitting
        closed = len([t for t in trades if "pnl" in t])
        penalty = 0.0 * closed  # ajuste se quiser penalizar operações
        score = pnl - penalty
        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optimize Triple RSI with Optuna (Binance data)")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo na Binance (ex.: BTCUSDT)")
    parser.add_argument("--interval", default="5m", help="Intervalo de velas (ex.: 5m, 15m, 1h)")
    parser.add_argument("--days", type=int, default=90, help="Dias de dados históricos para buscar")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fração de treino (restante validação)")
    parser.add_argument("--lot_size", type=float, default=0.1, help="Tamanho do lote para o backtest")
    parser.add_argument("--day_trade", action="store_true", help="Ativa lógica de day trade (B3)")
    parser.add_argument("--trials", type=int, default=300, help="Número de trials do Optuna")
    parser.add_argument("--seed", type=int, default=42, help="Semente do sampler para reprodutibilidade")
    parser.add_argument("--fix_invert", choices=["true", "false"], help="Fixar invert como true/false; se ausente, otimiza invert")
    parser.add_argument("--outdir", default="reports", help="Diretório para salvar o relatório")
    args = parser.parse_args()

    print(f"Carregando dados: {args.ticker} @ {args.interval} por {args.days} dias...")
    df = load_data(args.ticker, args.interval, args.days)
    n = len(df)
    split = int(n * args.train_frac)
    df_train = df.iloc[:split].copy()
    df_valid = df.iloc[split:].copy()

    print(f"Total candles: {n} | Treino: {len(df_train)} | Validação: {len(df_valid)}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(df_train, args.lot_size, args.day_trade, args.fix_invert), n_trials=args.trials)

    print("\nMelhores parâmetros (treino):")
    print(study.best_params)
    print(f"Melhor P&L (treino): $ {study.best_value:.2f}")

    # Avaliação na validação temporal
    p = study.best_params
    # Avaliar em treino e validação, e construir um relatório de recomendações
    params = study.best_params

    trades_tr, pnl_tr = backtest_triple_rsi(
        df_train.copy(),
        short_period=params["short"],
        med_period=params["med"],
        long_period=params["long"],
        buy_entry=params["buy_entry"],
        sell_entry=params["sell_entry"],
        buy_exit=params["buy_exit"],
        sell_exit=params["sell_exit"],
        invert=params["invert"] if "invert" in params else (args.fix_invert == "true"),
        lot_size=args.lot_size,
        day_trade=args.day_trade,
    )

    trades_val, pnl_val = backtest_triple_rsi(
        df_valid.copy(),
        short_period=params["short"],
        med_period=params["med"],
        long_period=params["long"],
        buy_entry=params["buy_entry"],
        sell_entry=params["sell_entry"],
        buy_exit=params["buy_exit"],
        sell_exit=params["sell_exit"],
        invert=params["invert"] if "invert" in params else (args.fix_invert == "true"),
        lot_size=args.lot_size,
        day_trade=args.day_trade,
    )
    print("\nDesempenho na validação:")
    print(f"P&L (val): $ {pnl_val:.2f} | trades fechados: {len([t for t in trades_val if 'pnl' in t])}")

    # Construir relatório
    def summarize(trades, pnl, initial_cap=10000.0):
        closed = [t for t in trades if "pnl" in t]
        n = len(closed)
        wins = len([t for t in closed if t["pnl"] > 0])
        win_rate = (wins / n * 100.0) if n else 0.0
        return {
            "pnl": float(pnl),
            "num_trades": n,
            "win_rate": win_rate,
            "return_pct": (float(pnl) / initial_cap) * 100.0,
        }

    rec = {
        "ticker": args.ticker,
        "interval": args.interval,
        "days": args.days,
        "train_frac": args.train_frac,
        "lot_size": args.lot_size,
        "day_trade": args.day_trade,
        "best_params": {
            "short_period": params["short"],
            "med_period": params["med"],
            "long_period": params["long"],
            "buy_entry": params["buy_entry"],
            "sell_entry": params["sell_entry"],
            "buy_exit": params["buy_exit"],
            "sell_exit": params["sell_exit"],
            "invert": params["invert"] if "invert" in params else (args.fix_invert == "true"),
            "lot_size": args.lot_size,
        },
        "train_metrics": summarize(trades_tr, pnl_tr),
        "valid_metrics": summarize(trades_val, pnl_val),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = f"triple_rsi_optuna_{args.ticker}_{args.interval}_{ts}"

    # JSON
    json_path = outdir / f"{base}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    # Markdown amigável
    md_path = outdir / f"{base}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Triple RSI Optimization Report\n\n")
        f.write(f"- Ticker: {args.ticker}\n")
        f.write(f"- Interval: {args.interval}\n")
        f.write(f"- Days: {args.days}\n")
        f.write(f"- Train frac: {args.train_frac}\n")
        f.write(f"- Trials: {args.trials}\n")
        f.write(f"- Best PnL (train): $ {pnl_tr:.2f}\n")
        f.write(f"- PnL (valid): $ {pnl_val:.2f}\n")
        f.write("\n## Best Parameters\n")
        for k, v in rec["best_params"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Train Metrics\n")
        for k, v in rec["train_metrics"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Validation Metrics\n")
        for k, v in rec["valid_metrics"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## How to Apply\n")
        f.write("Update your calls with parameters below.\n\n")
        inv = rec["best_params"]["invert"]
        f.write("Live example (src/triple_rsi_live.py):\n\n")
        f.write(
            """
live_triple_rsi(
    ticker="{ticker}",
    short_period={short}, med_period={med}, long_period={long},
    buy_entry={be}, sell_entry={se}, buy_exit={bx}, sell_exit={sx},
    invert={inv},
    interval_minutes={interval_minutes}
)
""".format(
                ticker=args.ticker,
                short=params["short"],
                med=params["med"],
                long=params["long"],
                be=params["buy_entry"],
                se=params["sell_entry"],
                bx=params["buy_exit"],
                sx=params["sell_exit"],
                inv=str(inv),
                interval_minutes=5 if args.interval.endswith("m") else 15,
            )
        )
        f.write("\nBacktest example (src/triple_rsi_backtest.py):\n\n")
        f.write(
            """
backtest_triple_rsi(
    df.copy(),
    short_period={short}, med_period={med}, long_period={long},
    buy_entry={be}, sell_entry={se}, buy_exit={bx}, sell_exit={sx},
    invert={inv}, day_trade={day_trade}, lot_size={lot}
)
""".format(
                short=params["short"],
                med=params["med"],
                long=params["long"],
                be=params["buy_entry"],
                se=params["sell_entry"],
                bx=params["buy_exit"],
                sx=params["sell_exit"],
                inv=str(inv),
                day_trade=str(args.day_trade),
                lot=args.lot_size,
            )
        )

    # Atualiza configuração ativa para consumo pelo live/backtest
    active_path = save_active_config(rec, reports_dir=args.outdir)

    print(f"\nRelatórios salvos em: {json_path} e {md_path}")
    print(f"Config ativa atualizada: {active_path}")


if __name__ == "__main__":
    main()
