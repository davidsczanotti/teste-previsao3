from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import load_price_history


class Action(IntEnum):
    """Discrete set of trading actions for the environment."""

    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Observation:
    """Snapshot returned by the environment after each transition."""

    features: np.ndarray
    position: int
    balance: float
    equity: float
    price: float
    step: int


@dataclass
class Trade:
    """Record of a completed trade, used for reporting at the end of episodes."""

    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float


class TradingEnvironment:
    """Minimal single-asset trading environment with long-only exposure.

    The agent can hold at most one position of fixed size. Each step it decides
    between the discrete actions ``HOLD``, ``BUY`` or ``SELL``. Observations
    expose a rolling window of percentage returns so that a very small state can
    be used together with a table-based RL algorithm.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 8,
        position_size: float = 0.1,
        starting_balance: float = 1_000.0,
        trading_fee: float = 0.001,
        episode_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Build an environment from pre-loaded candle data.

        Args:
            data: Candle dataframe returned by ``load_price_history``.
            window_size: Number of past returns included in the observation.
            position_size: Quantity of the asset (BTC) traded per position.
            starting_balance: Initial cash balance in quote currency (USDT).
            trading_fee: Fractional fee applied on both buy and sell actions.
            episode_length: Optional cap on steps per episode. When provided the
                environment randomly samples a start index and runs for at most
                this many steps.
            seed: Optional random seed to get reproducible starting points.
        """

        if len(data) < window_size + 2:
            raise ValueError(
                "Quantidade de candles insuficiente para construir a janela de "
                f"observação ({len(data)} < {window_size + 2})."
            )

        self.data = data.reset_index(drop=True).copy()
        self.window_size = window_size
        self.position_size = float(position_size)
        self.starting_balance = float(starting_balance)
        self.fee = float(trading_fee)
        self.episode_length = episode_length
        self.random = random.Random(seed)

        self.max_trade_index = len(self.data) - 2  # precisa de um candle futuro

        self._reset_state(initial_call=True)

    def _reset_state(self, initial_call: bool = False) -> None:
        """Initialize bookkeeping attributes without returning an observation."""

        self.balance = self.starting_balance
        self.position = 0
        self.quantity = 0.0
        self.entry_price = 0.0
        self.entry_cost = 0.0
        self.entry_index = -1
        self.done = False
        self.trades: List[Trade] = []

        start_min = self.window_size
        start_max = self.max_trade_index
        if self.episode_length is not None:
            start_max = max(
                self.window_size,
                self.max_trade_index - self.episode_length,
            )
        if start_max < start_min:
            start_max = start_min
        start_index = start_min if initial_call else self.random.randint(start_min, start_max)

        self.current_index = start_index

        if self.episode_length is None:
            self.episode_end_index = self.max_trade_index
        else:
            self.episode_end_index = min(self.max_trade_index, start_index + self.episode_length)

        self.last_equity = self._equity(self._previous_close_price())

    def reset(self) -> Observation:
        """Start a new episode and return the initial observation."""

        self._reset_state(initial_call=False)
        return self._build_observation()

    def step(self, action: Action | int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Advance the environment one step after applying ``action``.

        Args:
            action: Either an ``Action`` enum member or its integer value.

        Returns:
            observation: New state seen after the transition.
            reward: Difference in total equity between this and the last step.
            done: Flag indicating the episode finished.
            info: Dictionary with human-readable details for logging.
        """

        if self.done:
            raise RuntimeError("A chamada de step() ocorreu após o episódio terminar.")

        action_enum = Action(action)
        execution_price = self._current_price()
        prev_equity = self._equity(execution_price)

        info = self._apply_action(action_enum, execution_price)

        self.current_index += 1
        reached_end = self.current_index >= self.episode_end_index

        if reached_end:
            next_price = execution_price
            self.done = True
        else:
            next_price = self._current_price()

        new_equity = self._equity(next_price)
        reward = new_equity - prev_equity
        self.last_equity = new_equity

        if self.balance <= 0:
            self.done = True

        observation = self._build_observation()
        done_flag = self.done

        info.update(
            {
                "action": action_enum.name,
                "price": execution_price,
                "next_price": next_price,
                "balance": self.balance,
                "equity": new_equity,
                "position": self.position,
            }
        )
        return observation, reward, done_flag, info

    def _apply_action(self, action: Action, price: float) -> Dict[str, Any]:
        """Execute the trading action and return contextual logging metadata."""

        log_details: Dict[str, Any] = {}

        if action is Action.HOLD:
            log_details["event"] = "Aguardando"
            return log_details

        if action is Action.BUY:
            if self.position == 1:
                log_details["event"] = "Compra ignorada (já comprado)"
                return log_details

            gross_cost = price * self.position_size
            total_cost = gross_cost * (1 + self.fee)
            if total_cost > self.balance:
                log_details["event"] = "Compra cancelada (saldo insuficiente)"
                self.done = True  # Termina o episódio por ação inválida
                return log_details

            self.balance -= total_cost
            self.position = 1
            self.quantity = self.position_size
            self.entry_price = price
            self.entry_cost = total_cost
            self.entry_index = self.current_index
            log_details["event"] = "Compra executada"
            log_details["trade"] = {
                "side": "BUY",
                "price": price,
                "qty": self.quantity,
            }
            return log_details

        if action is Action.SELL:
            if self.position == 0:
                log_details["event"] = "Venda ignorada (sem posição)"
                return log_details

            gross_proceeds = price * self.quantity
            net_proceeds = gross_proceeds * (1 - self.fee)
            self.balance += net_proceeds
            pnl = net_proceeds - self.entry_cost
            trade = Trade(
                entry_index=self.entry_index,
                exit_index=self.current_index,
                entry_price=self.entry_price,
                exit_price=price,
                quantity=self.quantity,
                pnl=pnl,
            )
            self.trades.append(trade)
            self.position = 0
            self.quantity = 0.0
            self.entry_price = 0.0
            self.entry_cost = 0.0
            self.entry_index = -1
            log_details["event"] = "Venda executada"
            log_details["trade"] = asdict(trade)
            return log_details

        raise ValueError(f"Ação desconhecida: {action}")

    def _build_observation(self) -> Observation:
        """Construct the observation object for the current step."""

        window_end = self.current_index
        window_start = max(0, window_end - self.window_size)
        returns = self.data["return"].iloc[window_start:window_end].to_numpy(dtype=np.float32)
        if len(returns) < self.window_size:
            returns = np.pad(returns, (self.window_size - len(returns), 0))

        reference_price = self._previous_close_price()

        return Observation(
            features=returns,
            position=self.position,
            balance=self.balance,
            equity=self.last_equity,
            price=reference_price,
            step=window_end,
        )

    def _previous_close_price(self) -> float:
        """Return the latest available close price for rendering and equity."""

        idx = max(0, self.current_index - 1)
        return float(self.data["close"].iloc[idx])

    def _current_price(self) -> float:
        """Return the close price used to execute the current step."""

        return float(self.data["close"].iloc[self.current_index])

    def _equity(self, mark_price: float) -> float:
        """Compute total account equity using ``mark_price`` for open positions."""

        return self.balance + self.quantity * mark_price

    @property
    def trade_log(self) -> List[Trade]:
        """Provide the list of completed trades for diagnostic purposes."""

        return self.trades


def build_environment(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start: Optional[str] = None,
    end: Optional[str] = None,
    window_size: int = 8,
    position_size: float = 0.1,
    starting_balance: float = 1_000.0,
    trading_fee: float = 0.001,
    episode_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> TradingEnvironment:
    """Helper that loads cached data and instantiates ``TradingEnvironment``."""

    data = load_price_history(symbol=symbol, interval=interval, start=start, end=end)
    return TradingEnvironment(
        data=data,
        window_size=window_size,
        position_size=position_size,
        starting_balance=starting_balance,
        trading_fee=trading_fee,
        episode_length=episode_length,
        seed=seed,
    )
