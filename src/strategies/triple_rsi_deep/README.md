# PPO Multi-Timeframe Agent

This directory contains an advanced trading agent based on Deep Reinforcement Learning.

## Core Concept

The agent uses the **Proximal Policy Optimization (PPO)** algorithm to make trading decisions. It follows a **Multi-Timeframe Analysis (MTF)** approach, where it operates on a short primary timeframe (1-minute candles) while using a longer secondary timeframe (15-minute candles) to understand the underlying market trend.

This allows the agent to make high-frequency decisions with the context of the broader market structure, preventing it from getting lost in short-term noise.

## Feature Set

The agent's decisions are based on a rich set of features:

-   **Primary Interval (1m):**
    -   Relative Strength Index (RSI) with multiple periods.
    -   Stochastic Oscillator (%K and %D).
    -   Average True Range (ATR) for volatility.
    -   On-Balance Volume (OBV).

-   **Trend Interval (15m):**
    -   Average Directional Index (ADX, DMI+, DMI-) to gauge trend strength.

-   **Time-Based Features:**
    -   Cyclical features representing the day of the week and the hour of the day, allowing the agent to learn daily and weekly patterns.

-   **Agent State:**
    -   Current position (long, short, or flat).
    -   Number of bars held in the current position.

## Reward Philosophy

The agent is optimized to maximize a risk-adjusted return metric, the **Sharpe Ratio**. While the step-by-step rewards are based on realized profit and loss (PnL), the ultimate goal measured at the end of each training episode is to generate stable, consistent returns relative to volatility.

## How to Run

To train the agent, run the following command from the root directory of the project:

```bash
python -m src.strategies.triple_rsi_deep.train
```
