"""
Strategy modules for the MAR Trading Bot

Provides abstract base class and concrete strategy implementations.
"""

from .base_strategy import BaseStrategy
from .regime_strategy import RegimeStrategy

__all__ = [
    'BaseStrategy',
    'RegimeStrategy',
]
