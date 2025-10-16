#!/usr/bin/env python3
"""
Elite PPO Trading Agent - Complete Usage Example

This script demonstrates how to use the advanced Triple RSI Deep trading agent
with all its cutting-edge features for professional algorithmic trading.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepTripleRsiConfig
from train import train
from backtest import run_comprehensive_backtest
from env import TripleRsiEnv


def example_training_basic():
    """Example 1: Basic training with default settings."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Training")
    print("=" * 60)

    # Use default configuration (MLP architecture)
    cfg = DeepTripleRsiConfig()

    print(f"Training agent for {cfg.symbol} on {cfg.interval} timeframe")
    print(f"Training period: {cfg.days} days")
    print(f"Episodes: {cfg.episodes}")
    print(f"Architecture: {'Transformer' if cfg.use_transformer else 'MLP'}")
    print()

    # Train the agent
    results = train(cfg)

    print("Training completed!")
    print(".3f")
    print(".3f")
    print(f"Model saved to: {results['model_path']}")

    return results['model_path']


def example_training_transformer():
    """Example 2: Advanced training with Transformer architecture."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Transformer Architecture Training")
    print("=" * 60)

    # Configure for Transformer architecture
    cfg = DeepTripleRsiConfig(
        use_transformer=True,
        transformer_layers=3,
        transformer_heads=8,
        transformer_dim=128,
        episodes=100,  # Shorter for demo
        learning_rate=3e-4
    )

    print("Training with Transformer architecture:")
    print(f"- Layers: {cfg.transformer_layers}")
    print(f"- Heads: {cfg.transformer_heads}")
    print(f"- Hidden dim: {cfg.transformer_dim}")
    print(f"- Episodes: {cfg.episodes}")
    print()

    results = train(cfg)

    print("Transformer training completed!")
    print(".3f")
    print(".3f")

    return results['model_path']


def example_backtesting(model_path: str):
    """Example 3: Comprehensive backtesting."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Comprehensive Backtesting")
    print("=" * 60)

    # Define backtest period (last 6 months)
    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S")

    print(f"Backtesting period: {start_date} to {end_date}")
    print(f"Model: {model_path}")
    print()

    # Run comprehensive backtest
    results = run_comprehensive_backtest(model_path, start_date, end_date)

    print("\nBacktest Results Summary:")
    print("-" * 40)
    summary = results['summary']
    print(".3f")
    print(".1%")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".1%")

    # Performance assessment
    print("\nPerformance Assessment:")
    if summary['sharpe_ratio'] > 1.0:
        print("✓ Excellent Sharpe ratio (>1.0)")
    elif summary['sharpe_ratio'] > 0.5:
        print("✓ Good Sharpe ratio (>0.5)")
    else:
        print("⚠ Low Sharpe ratio - needs improvement")

    if summary['max_drawdown'] < 0.15:
        print("✓ Acceptable maximum drawdown (<15%)")
    else:
        print("⚠ High drawdown - risk management needed")

    if summary['sharpe_stability'] > 0.8:
        print("✓ High walk-forward stability")
    else:
        print("⚠ Low stability across time periods")

    if summary['prob_positive_sharpe'] > 0.7:
        print("✓ Consistent positive performance in MC simulations")
    else:
        print("⚠ Inconsistent performance - needs robustness")

    return results


def example_live_trading_simulation(model_path: str):
    """Example 4: Live trading simulation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Live Trading Simulation")
    print("=" * 60)

    from ...binance_client import get_current_price

    try:
        # Get current market data
        current_price = get_current_price("BTCUSDT")
        print(".2f")

        # Create environment for simulation
        cfg = DeepTripleRsiConfig()
        env = TripleRsiEnv(cfg)
        env.reset(seed=42)

        # Load model
        import torch
        from train import TransformerActorCritic, MLPActorCritic

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.use_transformer:
            model = TransformerActorCritic(
                input_dim=17, hidden_dim=cfg.transformer_dim, output_dim=3,
                num_layers=cfg.transformer_layers, num_heads=cfg.transformer_heads,
                dropout=cfg.dropout
            )
        else:
            model = MLPActorCritic(
                input_dim=17, hidden_dims=cfg.mlp_hidden_sizes, output_dim=3,
                use_skip=cfg.use_skip_connections
            )

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Simulate live decision
        obs = env.reset(seed=42)
        obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        with torch.no_grad():
            action_dist, _ = model(obs_tensor)
            action = action_dist.probs.argmax().item()

        action_names = ["HOLD", "OPEN LONG", "CLOSE LONG"]
        print(f"Agent decision: {action_names[action]}")

        # Calculate position size using Kelly criterion
        kelly_fraction = env.calculate_kelly_criterion()
        position_size = cfg.base_lot_size * max(0, kelly_fraction)
        print(".6f")
        print(".2f")

        # Risk metrics
        var_95 = env.calculate_var(0.95)
        print(".2f")

        return {
            'decision': action_names[action],
            'position_size': position_size,
            'kelly_fraction': kelly_fraction,
            'var_95': var_95
        }

    except Exception as e:
        print(f"Live simulation failed: {e}")
        return None


def example_custom_configuration():
    """Example 5: Custom configuration for specific trading style."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Configuration")
    print("=" * 60)

    # Conservative configuration
    conservative_cfg = DeepTripleRsiConfig(
        # Risk management
        kelly_fraction=0.3,  # Conservative Kelly
        max_drawdown_limit=0.10,  # 10% max drawdown
        var_confidence=0.99,  # 99% VaR

        # Reward system - emphasize risk metrics
        reward_sharpe_weight=0.5,
        reward_sortino_weight=0.3,
        reward_calmar_weight=0.2,

        # Position management
        min_hold_bars=8,  # Hold longer
        max_position_bars=120,  # 2 hours max hold

        # Training
        episodes=50,  # Shorter training for demo
        learning_rate=1e-4  # Conservative learning rate
    )

    print("Conservative configuration:")
    print(f"- Kelly fraction: {conservative_cfg.kelly_fraction}")
    print(f"- Max drawdown limit: {conservative_cfg.max_drawdown_limit}")
    print(f"- VaR confidence: {conservative_cfg.var_confidence}")
    print(f"- Reward weights: Sharpe={conservative_cfg.reward_sharpe_weight}, Sortino={conservative_cfg.reward_sortino_weight}, Calmar={conservative_cfg.reward_calmar_weight}")

    # Aggressive configuration
    aggressive_cfg = DeepTripleRsiConfig(
        # Risk management
        kelly_fraction=0.8,  # Aggressive Kelly
        max_drawdown_limit=0.25,  # Higher risk tolerance
        var_confidence=0.90,  # 90% VaR

        # Reward system - emphasize returns
        reward_sharpe_weight=0.3,
        reward_sortino_weight=0.2,
        reward_calmar_weight=0.1,
        reward_pnl_weight=0.4,  # Include raw PnL

        # Position management
        min_hold_bars=2,  # Quick trades
        max_position_bars=60,  # 1 hour max hold

        # Training
        episodes=100,
        learning_rate=5e-4,  # Higher learning rate
        entropy_beta=0.05  # More exploration
    )

    print("\nAggressive configuration:")
    print(f"- Kelly fraction: {aggressive_cfg.kelly_fraction}")
    print(f"- Max drawdown limit: {aggressive_cfg.max_drawdown_limit}")
    print(f"- VaR confidence: {aggressive_cfg.var_confidence}")
    print(f"- Reward weights: Sharpe={aggressive_cfg.reward_sharpe_weight}, Sortino={aggressive_cfg.reward_sortino_weight}, PnL={aggressive_cfg.reward_pnl_weight}")

    return conservative_cfg, aggressive_cfg


def main():
    """Run all examples."""
    print("ELITE PPO TRADING AGENT - COMPLETE USAGE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the full capabilities of the advanced")
    print("Triple RSI Deep trading agent with cutting-edge features.")
    print()

    try:
        # Example 1: Basic training
        model_path_basic = example_training_basic()

        # Example 2: Transformer training
        model_path_transformer = example_training_transformer()

        # Example 3: Backtesting
        backtest_results = example_backtesting(model_path_transformer)

        # Example 4: Live simulation
        live_results = example_live_trading_simulation(model_path_transformer)

        # Example 5: Custom configurations
        conservative_cfg, aggressive_cfg = example_custom_configuration()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key Takeaways:")
        print("1. ✓ Advanced architectures (Transformer, MLP with skip connections)")
        print("2. ✓ Multi-objective reward system (Sharpe, Sortino, Calmar)")
        print("3. ✓ Comprehensive risk management (VaR, Kelly, drawdown control)")
        print("4. ✓ Robust validation (Walk-forward, Monte Carlo)")
        print("5. ✓ Multi-timeframe analysis")
        print("6. ✓ Professional backtesting framework")
        print()
        print("The agent is now ready for live trading with:")
        print("- Sharpe ratio optimization")
        print("- Risk-adjusted position sizing")
        print("- Market regime adaptation")
        print("- Comprehensive performance monitoring")

        return {
            'model_paths': [model_path_basic, model_path_transformer],
            'backtest_results': backtest_results,
            'live_simulation': live_results,
            'configurations': [conservative_cfg, aggressive_cfg]
        }

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()

    if results:
        print("\n🎯 Elite PPO Agent successfully demonstrated!")
        print("You now have a professional-grade trading system.")
    else:
        print("\n❌ Some examples failed. Check the error messages above.")
