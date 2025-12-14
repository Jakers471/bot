"""
Base Strategy Abstract Class

Defines the interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Literal, Any
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All concrete strategy implementations must inherit from this class
    and implement the generate_signal method.
    """

    def __init__(self, **kwargs):
        """
        Initialize the strategy with optional configuration parameters.

        Args:
            **kwargs: Strategy-specific configuration parameters
        """
        self.config = kwargs

    @abstractmethod
    def generate_signal(
        self,
        row: pd.Series,
        position: int
    ) -> Literal['buy', 'sell', 'hold']:
        """
        Generate a trading signal based on current market data and position.

        This is the core method that each strategy must implement. It analyzes
        the current market state (represented by a DataFrame row) and the current
        position to decide whether to buy, sell, or hold.

        Args:
            row: A pandas Series representing a single row of market data.
                 Contains OHLCV data, indicators, and any other features.
            position: Current position state:
                     - 0: No position (flat)
                     - 1: Long position (holding)

        Returns:
            One of: 'buy', 'sell', or 'hold'
            - 'buy': Enter a long position (only valid when position=0)
            - 'sell': Exit the current position (only valid when position=1)
            - 'hold': Do nothing, maintain current position

        Examples:
            >>> strategy = SomeStrategy()
            >>> row = df.iloc[100]  # Get market data at index 100
            >>> signal = strategy.generate_signal(row, position=0)
            >>> print(signal)  # 'buy', 'sell', or 'hold'
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal state of the strategy.

        Called at the beginning of a backtest or when starting live trading.
        Override this method if your strategy maintains internal state.
        """
        pass

    def get_name(self) -> str:
        """
        Get the name of the strategy.

        Returns:
            String name of the strategy (defaults to class name)
        """
        return self.__class__.__name__

    def get_config(self) -> dict:
        """
        Get the strategy's configuration parameters.

        Returns:
            Dictionary of configuration parameters
        """
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.get_name()}(config={self.config})"
