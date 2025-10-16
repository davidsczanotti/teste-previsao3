from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
from torch.distributions import Categorical

from .config import DeepTripleRsiConfig
from .env import TripleRsiEnv
from .train import TransformerActorCritic, MLPActorCritic, EnsembleActorCritic
from ...utils.data_loader import load_data_range


class Backtester:
    """
    Advanced backtester with walk-forward validation and comprehensive metrics.
    """

    def __init__(self, config: Optional[DeepTripleRsiConfig] = None):
        self.cfg = config or DeepTripleRsiConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: str, input_dim: int, output_dim: int) -> torch.nn.Module:
        """Load trained model from path."""
        # Determine model type from config
        if self.cfg.use_transformer:
            model = TransformerActorCritic(
                input_dim=input_dim,
                hidden_dim=self.cfg.transformer_dim,
                output_dim=output_dim,
                num_layers=self.cfg.transformer_layers,
                num_heads=self.cfg.transformer_heads,
                dropout=self.cfg.dropout
            )
        else:
            model = MLPActorCritic(
                input_dim=input_dim,
                hidden_dims=self.cfg.mlp_hidden_sizes,
                output_dim=output_dim,
                use_skip=self.cfg.use_skip_connections
            )

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def run_backtest(self, model_path: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run backtest for a specific date range.
        """
        # Load data
        df = load_data_range(self.cfg.symbol, self.cfg.interval, start_date, end_date)

        # Create environment with loaded data
        env = TripleRsiEnv(config=self.cfg, df_primary=df)
        env.reset(seed=42)

        # Load model
        model = self.load_model(model_path, input_dim=env.observation_size, output_dim=env.action_size)

        # Run simulation
        obs = env.reset(seed=42)
        done = False
        trades = []

        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)

            with torch.no_grad():
                action_dist, _ = model(obs_tensor)
                action = action_dist.probs.argmax().item()  # Greedy action

            res = env.step(action)
            obs = res.obs
            done = res.done

            if res.info.get("trade"):
                timestamp = None
                if df is not None and len(df) > 0 and env._i < len(df):
                    timestamp = df.iloc[env._i]['Date']

                price = 0.0
                if env._prices is not None and env._i < len(env._prices):
                    price = float(env._prices[env._i])

                trades.append({
                    'timestamp': timestamp,
                    'action': res.info["trade"],
                    'price': price,
                    'portfolio_value': float(env.portfolio_values[-1])
                })

        # Calculate comprehensive metrics
        stats = env.get_portfolio_stats()
        stats['trades'] = trades
        stats['total_trades'] = len([t for t in trades if isinstance(t.get('action'), str) and 'OPEN' in t['action']])

        return stats

    def walk_forward_validation(self, model_path: str, start_date: str, end_date: str,
                              validation_splits: int = 5) -> Dict[str, Any]:
        """
        Perform walk-forward validation to assess model robustness.
        """
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days

        # Calculate split points
        split_size = total_days // validation_splits
        results = []

        for i in range(validation_splits):
            train_end = start_dt + timedelta(days=(i + 1) * split_size)
            val_start = train_end
            val_end = min(val_start + timedelta(days=split_size), end_dt)

            if val_start >= end_dt:
                break

            print(f"WFA Split {i+1}: Train until {train_end.date()}, Validate {val_start.date()} to {val_end.date()}")

            # Run backtest for this validation period
            val_result = self.run_backtest(
                model_path,
                val_start.strftime("%Y-%m-%d %H:%M:%S"),
                val_end.strftime("%Y-%m-%d %H:%M:%S")
            )
            val_result['split'] = i + 1
            val_result['validation_period'] = f"{val_start.date()} to {val_end.date()}"
            results.append(val_result)

        # Aggregate results
        aggregated = self._aggregate_wfa_results(results)
        return {
            'individual_splits': results,
            'aggregated': aggregated
        }

    def _aggregate_wfa_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate walk-forward validation results."""
        if not results:
            return {}

        # Extract key metrics
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        sortino_ratios = [r.get('sortino_ratio', 0) for r in results]
        calmar_ratios = [r.get('calmar_ratio', 0) for r in results]
        total_returns = [r.get('total_return', 0) for r in results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in results]

        return {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'sharpe_stability': np.mean(sharpe_ratios) / (np.std(sharpe_ratios) + 1e-8),

            'mean_sortino': np.mean(sortino_ratios),
            'std_sortino': np.std(sortino_ratios),

            'mean_calmar': np.mean(calmar_ratios),
            'std_calmar': np.std(calmar_ratios),

            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),

            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': max(max_drawdowns),

            'sharpe_consistency': sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios),
            'return_consistency': sum(1 for r in total_returns if r > 0) / len(total_returns)
        }

    def monte_carlo_simulation(self, model_path: str, start_date: str, end_date: str,
                             num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations to assess strategy robustness.
        """
        results = []

        for i in range(num_simulations):
            # Add random noise to simulate different market conditions
            cfg_sim = self.cfg.__dict__.copy()
            # Add small random variations to key parameters
            cfg_sim['fee_rate'] = self.cfg.fee_rate * (1 + np.random.normal(0, 0.1))
            cfg_sim['slippage_bps'] = max(0, self.cfg.slippage_bps + np.random.normal(0, 0.5))

            config_sim = DeepTripleRsiConfig(**cfg_sim)
            env = TripleRsiEnv(config=config_sim)
            env.reset(seed=i)

            # Load model
            model = self.load_model(model_path)

            # Run simulation
            obs = env.reset(seed=i)
            done = False

            while not done:
                obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)

                with torch.no_grad():
                    action_dist, _ = model(obs_tensor)
                    action = action_dist.probs.argmax().item()

                res = env.step(action)
                obs = res.obs
                done = res.done

            stats = env.get_portfolio_stats()
            stats['simulation'] = i + 1
            results.append(stats)

        # Analyze Monte Carlo results
        returns = [r['total_return'] for r in results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in results]

        return {
            'simulations': results,
            'return_distribution': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95),
                'var_95': np.percentile(returns, 5),  # VaR at 95% confidence
                'expected_shortfall': np.mean([r for r in returns if r <= np.percentile(returns, 5)])
            },
            'sharpe_distribution': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'positive_sharpe_prob': sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios)
            },
            'drawdown_distribution': {
                'mean_max_drawdown': np.mean(max_drawdowns),
                'worst_max_drawdown': np.max(max_drawdowns),
                'prob_drawdown_over_15pct': sum(1 for d in max_drawdowns if d > 0.15) / len(max_drawdowns)
            }
        }


def run_comprehensive_backtest(model_path: str, start_date: str = "2023-01-01 00:00:00",
                              end_date: str = "2024-01-01 00:00:00") -> Dict[str, Any]:
    """
    Run comprehensive backtest with multiple validation methods.
    """
    backtester = Backtester()

    print("Running comprehensive backtest...")
    print(f"Model: {model_path}")
    print(f"Period: {start_date} to {end_date}")

    # 1. Basic backtest
    print("\n1. Running basic backtest...")
    basic_results = backtester.run_backtest(model_path, start_date, end_date)
    print(f"Basic backtest - Sharpe: {basic_results.get('sharpe_ratio', 0):.3f}, "
          f"Return: {basic_results.get('total_return', 0):.3f}, "
          f"Max DD: {basic_results.get('max_drawdown', 0):.3f}")

    # 2. Walk-forward validation
    print("\n2. Running walk-forward validation...")
    wfa_results = backtester.walk_forward_validation(model_path, start_date, end_date)
    print(f"WFA - Mean Sharpe: {wfa_results['aggregated'].get('mean_sharpe', 0):.3f}, "
          f"Sharpe Stability: {wfa_results['aggregated'].get('sharpe_stability', 0):.3f}")

    # 3. Monte Carlo simulation
    print("\n3. Running Monte Carlo simulation...")
    mc_results = backtester.monte_carlo_simulation(model_path, start_date, end_date, num_simulations=500)
    print(f"MC - Expected Return: {mc_results['return_distribution']['mean']:.3f}, "
          f"VaR 95%: {mc_results['return_distribution']['var_95']:.3f}")

    return {
        'basic_backtest': basic_results,
        'walk_forward_validation': wfa_results,
        'monte_carlo_simulation': mc_results,
        'summary': {
            'sharpe_ratio': basic_results.get('sharpe_ratio', 0),
            'total_return': basic_results.get('total_return', 0),
            'max_drawdown': basic_results.get('max_drawdown', 0),
            'sharpe_stability': wfa_results['aggregated'].get('sharpe_stability', 0),
            'return_consistency': wfa_results['aggregated'].get('return_consistency', 0),
            'expected_return_mc': mc_results['return_distribution']['mean'],
            'var_95_mc': mc_results['return_distribution']['var_95'],
            'prob_positive_sharpe': mc_results['sharpe_distribution']['positive_sharpe_prob']
        }
    }


if __name__ == "__main__":
    # Example usage
    model_path = "reports/agents/triple_rsi_deep/BTCUSDT_1m_ppo.pt"

    if os.path.exists(model_path):
        results = run_comprehensive_backtest(model_path)
        print("\n=== BACKTEST SUMMARY ===")
        for key, value in results['summary'].items():
            print(f"{key}: {value:.4f}")
    else:
        print(f"Model not found: {model_path}")
        print("Please train the model first using: python -m src.strategies.triple_rsi_deep.train")
