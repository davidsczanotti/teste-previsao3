#!/usr/bin/env python3
"""
Walk-Forward Validation para Estratégia AL Brooks
Implementa validação walk-forward para testar a robustez da estratégia
ao longo de diferentes períodos de mercado.
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from .backtest import backtest_al_brooks_inside_bar
from .optimize import make_objective, print_summary
from .config import AlBrooksConfig, load_active_config, save_active_config
from ...utils.data_loader import load_data
from ...utils.metrics import calculate_metrics, generate_summary_report

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Classe para implementar validação walk-forward
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1m",
        days: int = 365,
        lot_size: float = 0.1,
        min_trades_per_window: int = 15,
    ):
        """
        Inicializa o validador walk-forward

        Args:
            symbol: Símbolo do ativo
            timeframe: Timeframe das velas
            days: Dias de dados históricos
            lot_size: Tamanho do lote
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days
        self.lot_size = lot_size
        self.min_trades_per_window = min_trades_per_window
        self.results = []
        self.summary_stats = {}

    def create_periods(
        self, data: pd.DataFrame, optimization_window: int = 180, validation_window: int = 90, step_size: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Cria períodos de otimização e validação

        Args:
            data: DataFrame com dados históricos
            optimization_window: Tamanho da janela de otimização em dias
            validation_window: Tamanho da janela de validação em dias
            step_size: Passo entre períodos em dias

        Returns:
            Lista de períodos com índices de início/fim
        """
        periods = []
        total_candles = len(data)

        # Converte dias para número de candles (aproximado)
        # Considerando 24 candles por dia para timeframe de 15m (24*4=96)
        candles_per_day = 96 if self.timeframe == "1m" else 24
        opt_candles = optimization_window * candles_per_day
        val_candles = validation_window * candles_per_day
        step_candles = step_size * candles_per_day

        # Calcula total de candles necessário
        total_candles_needed = opt_candles + val_candles

        if total_candles_needed > total_candles:
            logger.warning(
                f"Dados insuficientes: necessário {total_candles_needed} candles, disponível {total_candles}"
            )
            return periods

        current_start = 0

        while current_start + total_candles_needed <= total_candles:
            # Período de otimização
            opt_end = current_start + opt_candles

            # Período de validação
            val_start = opt_end
            val_end = val_start + val_candles

            periods.append(
                {
                    "optimization_start": current_start,
                    "optimization_end": opt_end,
                    "validation_start": val_start,
                    "validation_end": val_end,
                    "period_number": len(periods) + 1,
                }
            )

            # Avança para o próximo período
            current_start += step_candles

        logger.info(f"Criados {len(periods)} períodos walk-forward")
        return periods

    def run_single_period(self, data: pd.DataFrame, period: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa otimização e validação para um único período

        Args:
            data: Dados históricos completos
            period: Dicionário com informações do período

        Returns:
            Resultados do período
        """
        logger.info(
            f"Processando período {period['period_number']}: "
            f"Otimização candle {period['optimization_start']} to {period['optimization_end']}"
        )

        # Filtra dados do período de otimização
        opt_data = data.loc[period["optimization_start"] : period["optimization_end"]]

        # Filtra dados do período de otimização
        opt_data = opt_data.reset_index(drop=True)

        # Executa otimização simplificada
        try:
            # Usando o otimizador completo do Optuna para consistência
            import optuna

            study = optuna.create_study(direction="maximize")
            study.optimize(
                make_objective(opt_data, self.lot_size),
                n_trials=50,  # Número de tentativas por período. Ajuste conforme necessário.
                show_progress_bar=False,
                gc_after_trial=True,
            )
            best_params = study.best_params
            best_score = study.best_value

            if study.best_trial:
                logger.info(f"Melhor score: {best_score:.4f}")
                logger.info(f"Melhores parâmetros: {best_params}")
        except Exception as e:
            logger.error(f"Erro na otimização do período {period['period_number']}: {e}")
            return {"period": period, "optimization_success": False, "error": str(e)}

        # Filtra dados do período de validação
        val_data = data.loc[period["validation_start"] : period["validation_end"]]
        val_data = val_data.reset_index(drop=True)

        # Executa backtest com parâmetros otimizados
        try:
            params_for_backtest = {
                "ema_fast_period": best_params["ema_fast_period"],
                "ema_medium_period": best_params["ema_medium_period"],
                "ema_slow_period": best_params["ema_slow_period"],
                "risk_reward_ratio": best_params["risk_reward_ratio"],
                "max_avg_deviation_pct": best_params["max_avg_deviation_pct"],
                "adx_threshold": best_params.get("adx_threshold", 22.0),
                "atr_stop_multiplier": best_params.get("atr_stop_multiplier", 1.5),
                "atr_trail_multiplier": best_params.get("atr_trail_multiplier", 0.5),
                "htf_lookback": best_params.get("htf_lookback", 20),
                "min_atr": best_params.get("min_atr", 0.0),
            }

            trades, total_pnl, _ = backtest_al_brooks_inside_bar(
                val_data.copy(),
                lot_size=self.lot_size,
                **params_for_backtest,
            )

            # Calcula métricas
            metrics = calculate_metrics(trades)
            closed_pnls = [t["pnl"] for t in trades if "pnl" in t]
            total_profit = sum(p for p in closed_pnls if p > 0)
            total_loss = abs(sum(p for p in closed_pnls if p < 0))

            result = {
                "period": period,
                "optimization_success": True,
                "validation_success": True,
                "best_params": best_params,
                "best_score": best_score,
                "trades": trades,
                "total_pnl": total_pnl,
                "metrics": metrics,
                "validation_pnl": total_pnl,
                "validation_trades": len(trades),
                "validation_win_rate": metrics.get("win_rate", 0),
                "validation_profit_factor": metrics.get("profit_factor", 0),
                "validation_profit": total_profit,
                "validation_loss": total_loss,
            }

            logger.info(
                f"Resultado validação: P&L ${result['validation_pnl']:.2f}, "
                f"Win Rate {result['validation_win_rate']:.2%}, "
                f"Profit Factor {result['validation_profit_factor']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Erro na validação do período {period['period_number']}: {e}")
            return {
                "period": period,
                "optimization_success": True,
                "best_params": best_params,
                "best_score": best_score,
                "validation_success": False,
                "error": str(e),
            }

    def run_walk_forward(
        self, optimization_window: int = 180, validation_window: int = 90, step_size: int = 30
    ) -> Dict[str, Any]:
        """
        Executa validação walk-forward completa

        Args:
            optimization_window: Tamanho da janela de otimização em dias
            validation_window: Tamanho da janela de validação em dias
            step_size: Passo entre períodos em dias

        Returns:
            Resultados agregados da validação
        """
        logger.info("Iniciando validação walk-forward...")

        # Carrega dados
        data = load_data(self.symbol, self.timeframe, self.days)
        logger.info(f"Dados carregados: {len(data)} candles de {data.index.min()} a {data.index.max()}")

        # Cria períodos
        periods = self.create_periods(data, optimization_window, validation_window, step_size)

        # Se não houver períodos, retorna resultado vazio
        if not periods:
            logger.warning("Nenhum período walk-forward pôde ser criado com os parâmetros fornecidos")
            return {
                "periods": [],
                "results": [],
                "summary_stats": {
                    "total_periods": 0,
                    "successful_periods": 0,
                    "success_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "median_pnl": 0.0,
                    "std_pnl": 0.0,
                    "max_pnl": 0.0,
                    "min_pnl": 0.0,
                    "avg_win_rate": 0.0,
                    "median_win_rate": 0.0,
                    "avg_profit_factor": 0.0,
                    "median_profit_factor": 0.0,
                    "periods_with_profit": 0,
                    "periods_with_loss": 0,
                    "aggregate_profit_factor": 0.0,
                    "min_trades_required": self.min_trades_per_window,
                },
                "report": {
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "AL Brooks",
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "summary_stats": {
                        "total_periods": 0,
                        "successful_periods": 0,
                        "success_rate": 0.0,
                        "total_pnl": 0.0,
                        "avg_pnl": 0.0,
                        "median_pnl": 0.0,
                        "std_pnl": 0.0,
                        "max_pnl": 0.0,
                        "min_pnl": 0.0,
                        "avg_win_rate": 0.0,
                        "median_win_rate": 0.0,
                        "avg_profit_factor": 0.0,
                        "median_profit_factor": 0.0,
                        "periods_with_profit": 0,
                        "periods_with_loss": 0,
                        "aggregate_profit_factor": 0.0,
                        "min_trades_required": self.min_trades_per_window,
                    },
                    "detailed_results": [],
                },
            }

        # Executa para cada período
        for period in periods:
            result = self.run_single_period(data, period)
            self.results.append(result)

        # Calcula estatísticas agregadas
        self.calculate_summary_stats()

        # Se summary_stats estiver vazio, inicializa com valores padrão
        if not self.summary_stats:
            self.summary_stats = {
                "total_periods": len(self.results),
                "successful_periods": 0,
                "success_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "median_pnl": 0.0,
                "std_pnl": 0.0,
                "max_pnl": 0.0,
                "min_pnl": 0.0,
                "avg_win_rate": 0.0,
                "median_win_rate": 0.0,
                "avg_profit_factor": 0.0,
                "median_profit_factor": 0.0,
                "periods_with_profit": 0,
                "periods_with_loss": 0,
                "aggregate_profit_factor": 0.0,
                "min_trades_required": self.min_trades_per_window,
            }

        # Gera relatório
        report = self.generate_report()

        logger.info("Validação walk-forward concluída!")

        return {"periods": periods, "results": self.results, "summary_stats": self.summary_stats, "report": report}

    def calculate_summary_stats(self) -> None:
        """
        Calcula estatísticas agregadas dos resultados
        """
        # Considera um período bem sucedido se a otimização funcionou e houve trades na validação
        successful_periods = [
            r
            for r in self.results
            if r.get("optimization_success")
            and r.get("validation_success")
            and r.get("validation_trades", 0) >= self.min_trades_per_window
        ]

        if not successful_periods:
            logger.warning("Nenhum período com trades na validação")
            return

        # Métricas de performance
        pnls = [r["validation_pnl"] for r in successful_periods]
        win_rates = [r["validation_win_rate"] for r in successful_periods]
        profit_factors = [r["validation_profit_factor"] for r in successful_periods]

        total_profit = sum(r.get("validation_profit", 0.0) for r in successful_periods)
        total_loss = sum(r.get("validation_loss", 0.0) for r in successful_periods)

        aggregate_profit_factor = float("inf") if total_loss == 0 else total_profit / total_loss if total_loss else 0.0

        self.summary_stats = {
            "total_periods": len(self.results),
            "successful_periods": len(successful_periods),
            "success_rate": len(successful_periods) / len(self.results) if self.results else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "median_pnl": np.median(pnls),
            "std_pnl": np.std(pnls),
            "max_pnl": max(pnls),
            "min_pnl": min(pnls),
            "avg_win_rate": np.mean(win_rates),
            "median_win_rate": np.median(win_rates),
            "avg_profit_factor": np.mean(profit_factors),
            "median_profit_factor": np.median(profit_factors),
            "periods_with_profit": sum(1 for p in pnls if p > 0),
            "periods_with_loss": sum(1 for p in pnls if p < 0),
            "aggregate_profit_factor": aggregate_profit_factor,
            "min_trades_required": self.min_trades_per_window,
        }

        logger.info(f"Estatísticas agregadas:")
        logger.info(
            f"  - Períodos bem sucedidos: {self.summary_stats['successful_periods']}/{self.summary_stats['total_periods']}"
        )
        logger.info(f"  - P&L Total: ${self.summary_stats['total_pnl']:.2f}")
        logger.info(f"  - P&L Médio: ${self.summary_stats['avg_pnl']:.2f}")
        logger.info(f"  - Win Rate Médio: {self.summary_stats['avg_win_rate']:.2%}")
        logger.info(f"  - Profit Factor Médio: {self.summary_stats['avg_profit_factor']:.2f}")

    def generate_report(self) -> Dict[str, Any]:
        """
        Gera relatório completo da validação walk-forward

        Returns:
            Dicionário com o relatório
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "strategy": "AL Brooks",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "summary_stats": self.summary_stats,
            "detailed_results": [],
            "min_trades_per_window": self.min_trades_per_window,
        }

        # Adiciona resultados detalhados
        for result in self.results:
            detailed_result = {
                "period_number": result["period"]["period_number"],
                "optimization_period": f"Candles {result['period']['optimization_start']} to {result['period']['optimization_end']}",
                "validation_period": f"Candles {result['period']['validation_start']} to {result['period']['validation_end']}",
                "optimization_success": result.get("optimization_success", False),
                "validation_success": result.get("validation_success", False),
                "best_score": result.get("best_score", 0),
                "validation_pnl": result.get("validation_pnl", 0),
                "validation_trades": result.get("validation_trades", 0),
                "validation_win_rate": result.get("validation_win_rate", 0),
                "validation_profit_factor": result.get("validation_profit_factor", 0),
                "validation_profit": result.get("validation_profit", 0),
                "validation_loss": result.get("validation_loss", 0),
                "best_params": result.get("best_params", {}),
            }

            if "error" in result:
                detailed_result["error"] = result["error"]

            report["detailed_results"].append(detailed_result)

        # Salva relatório
        report_path = f"reports/walk_forward/{self.symbol}_{self.timeframe}_report.json"

        # Cria diretório se não existir
        import os

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Relatório salvo em: {report_path}")

        # Gera gráfico de performance
        self.generate_performance_chart()

        return report

    def generate_performance_chart(self) -> None:
        """
        Gera gráfico de performance ao longo dos períodos
        """
        successful_results = [
            r
            for r in self.results
            if r.get("validation_success") and r.get("validation_trades", 0) >= self.min_trades_per_window
        ]

        if not successful_results:
            logger.warning("Nenhum resultado bem sucedido para gerar gráfico")
            return

        # Extrai dados
        periods = [r["period"]["period_number"] for r in successful_results]
        pnls = [r["validation_pnl"] for r in successful_results]
        win_rates = [r["validation_win_rate"] for r in successful_results]

        # Cria figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Gráfico de P&L
        ax1.bar(periods, pnls, color=["green" if p > 0 else "red" for p in pnls])
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax1.set_xlabel("Período")
        ax1.set_ylabel("P&L ($)")
        ax1.set_title(f"Performance por Período - {self.symbol} {self.timeframe}")
        ax1.grid(True, alpha=0.3)

        # Adiciona valores nas barras
        for i, (period, pnl) in enumerate(zip(periods, pnls)):
            ax1.text(
                period,
                pnl + (max(pnls) * 0.01 if pnl > 0 else min(pnls) * 0.01),
                f"${pnl:.0f}",
                ha="center",
                va="bottom" if pnl > 0 else "top",
            )

        # Gráfico de Win Rate
        ax2.plot(periods, win_rates, "o-", linewidth=2, markersize=6)
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="50%")
        ax2.set_xlabel("Período")
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Win Rate por Período")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Adiciona valores nos pontos
        for period, win_rate in zip(periods, win_rates):
            ax2.text(period, win_rate + 0.01, f"{win_rate:.1%}", ha="center", va="bottom")

        plt.tight_layout()

        # Salva gráfico
        chart_path = f"reports/charts/walk_forward_{self.symbol}_{self.timeframe}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Gráfico salvo em: {chart_path}")

    def simplified_optimization(self, data: pd.DataFrame, n_trials: int = 50) -> tuple[Dict[str, Any], float]:
        """
        Executa otimização simplificada com busca em grade

        Args:
            data: DataFrame com dados históricos
            n_trials: Número de combinações de parâmetros para testar

        Returns:
            Tupla com (melhores parâmetros, melhor score)
        """
        import random

        # Espaço de busca para parâmetros
        param_space = {
            "ema_fast_period": range(5, 15),
            "ema_medium_period": range(15, 30),
            "ema_slow_period": range(30, 60),
            "risk_reward_ratio": [1.5, 1.8, 2.0, 2.5, 3.0],
            "max_avg_deviation_pct": [0.1, 0.2, 0.3, 0.5],
        }

        best_score = -float("inf")
        best_params = None

        # Gera combinações aleatórias
        for _ in range(n_trials):
            # Seleciona parâmetros aleatórios com restrições
            ema_fast = random.choice(param_space["ema_fast_period"])
            ema_medium = random.choice([p for p in param_space["ema_medium_period"] if p > ema_fast])
            ema_slow = random.choice([p for p in param_space["ema_slow_period"] if p > ema_medium])

            params = {
                "ema_fast_period": ema_fast,
                "ema_medium_period": ema_medium,
                "ema_slow_period": ema_slow,
                "risk_reward_ratio": random.choice(param_space["risk_reward_ratio"]),
                "max_avg_deviation_pct": random.choice(param_space["max_avg_deviation_pct"]),
            }

            try:
                # Executa backtest com os parâmetros
                trades, total_pnl, _ = backtest_al_brooks_inside_bar(
                    data.copy(),
                    ema_fast_period=params["ema_fast_period"],
                    ema_medium_period=params["ema_medium_period"],
                    ema_slow_period=params["ema_slow_period"],
                    risk_reward_ratio=params["risk_reward_ratio"],
                    max_avg_deviation_pct=params["max_avg_deviation_pct"],
                    lot_size=self.lot_size,
                )

                # Calcula score baseado em Profit Factor e P&L
                closed_trades = [t for t in trades if "pnl" in t and t["pnl"] != 0]

                if not closed_trades:
                    continue

                winning_trades = [t for t in closed_trades if t["pnl"] > 0]
                losing_trades = [t for t in closed_trades if t["pnl"] <= 0]

                total_profit = sum(t["pnl"] for t in winning_trades)
                total_loss = abs(sum(t["pnl"] for t in losing_trades))

                if total_loss == 0:
                    profit_factor = 100.0  # Evita divisão por zero
                else:
                    profit_factor = total_profit / total_loss

                # Score ponderado: Profit Factor com peso maior para P&L positivo
                score = profit_factor * 1.0 if total_pnl > 0 else profit_factor * 0.5

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                # Ignora combinações que causam erro
                continue

        if best_params is None:
            # Retorna parâmetros padrão se nenhuma combinação funcionou
            best_params = {
                "ema_fast_period": 14,
                "ema_medium_period": 26,
                "ema_slow_period": 50,
                "risk_reward_ratio": 1.6,
                "max_avg_deviation_pct": 0.1,
            }
            best_score = 0.0

        return best_params, best_score


def main():
    """
    Função principal para executar validação walk-forward
    """
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Validation para Estratégia AL Brooks")
    parser.add_argument("--config", type=str, help="Caminho do arquivo de configuração")
    parser.add_argument("--opt-window", type=int, default=180, help="Tamanho da janela de otimização (dias)")
    parser.add_argument("--val-window", type=int, default=90, help="Tamanho da janela de validação (dias)")
    parser.add_argument("--step-size", type=int, default=30, help="Passo entre períodos (dias)")
    parser.add_argument(
        "--min-trades",
        type=int,
        default=15,
        help="Número mínimo de trades na janela de validação para considerar o período válido",
    )

    args = parser.parse_args()

    # Executa validação
    validator = WalkForwardValidator(min_trades_per_window=args.min_trades)
    results = validator.run_walk_forward(
        optimization_window=args.opt_window, validation_window=args.val_window, step_size=args.step_size
    )

    # Imprime resumo
    print("\n" + "=" * 60)
    print("RESUMO DA VALIDAÇÃO WALK-FORWARD")
    print("=" * 60)
    print(f"Estratégia: AL Brooks")
    print(f"Símbolo: {validator.symbol}")
    print(f"Timeframe: {validator.timeframe}")
    print(f"Trades mínimos por janela: {validator.min_trades_per_window}")
    print(f"Períodos totais: {results['summary_stats']['total_periods']}")
    print(f"Períodos bem sucedidos: {results['summary_stats']['successful_periods']}")
    print(f"Taxa de sucesso: {results['summary_stats']['success_rate']:.2%}")
    print(f"P&L Total: ${results['summary_stats']['total_pnl']:.2f}")
    print(f"P&L Médio: ${results['summary_stats']['avg_pnl']:.2f}")
    print(f"Win Rate Médio: {results['summary_stats']['avg_win_rate']:.2%}")
    print(f"Profit Factor Médio: {results['summary_stats']['avg_profit_factor']:.2f}")
    pf_agg = results["summary_stats"].get("aggregate_profit_factor", float("nan"))
    if np.isfinite(pf_agg):
        print(f"Profit Factor Agregado: {pf_agg:.2f}")
    else:
        print("Profit Factor Agregado: inf")
    print("=" * 60)


if __name__ == "__main__":
    main()
