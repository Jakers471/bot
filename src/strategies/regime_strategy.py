"""
Regime-Based Trading Strategy

This strategy uses market regime detection to generate trading signals.
It buys when the market transitions to an uptrend and sells when the
market transitions to ranging or downtrend conditions.
"""

from typing import Literal, Optional
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

try:
    from src.regime_detector import load_regime_config, detect_regimes
except ImportError:
    from ..regime_detector import load_regime_config, detect_regimes


class RegimeStrategy(BaseStrategy):
    """
    Trading strategy based on market regime detection.

    This strategy generates signals based on regime transitions:
    - BUY: When regime changes to 1 (uptrend) and not in position
    - SELL: When regime changes to 0 (ranging) or 2 (downtrend) and in position
    - HOLD: No regime change or inappropriate position state

    The regime detection is based on moving average analysis including:
    - Percentage of MAs with positive slope
    - Price position within MA bundle
    - MA spread (ribbon width)

    Attributes:
        regime_config_name: Name of the regime configuration to use
                           (e.g., 'default', 'aggressive', 'conservative')
        regime_params: Dictionary of regime detection parameters
        previous_regime: Tracks the previous regime for change detection
    """

    def __init__(self, regime_config_name: str = 'default', **kwargs):
        """
        Initialize the RegimeStrategy.

        Args:
            regime_config_name: Name of the regime configuration to load.
                               Options: 'default', 'aggressive', 'conservative'
                               - 'default': Balanced regime detection
                               - 'aggressive': More sensitive, faster regime changes
                               - 'conservative': Less sensitive, slower regime changes
            **kwargs: Additional configuration parameters (passed to base class)

        Raises:
            FileNotFoundError: If regime config file doesn't exist
            KeyError: If specified config name is not found

        Example:
            >>> strategy = RegimeStrategy('aggressive')
            >>> strategy = RegimeStrategy('conservative')
        """
        super().__init__(**kwargs)
        self.regime_config_name = regime_config_name
        self.regime_params = load_regime_config(regime_config_name)
        self.previous_regime: Optional[int] = None

    def generate_signal(
        self,
        row: pd.Series,
        position: int
    ) -> Literal['buy', 'sell', 'hold']:
        """
        Generate trading signal based on regime changes.

        The strategy logic:
        1. Check if 'regime' column exists in the row
        2. Detect regime transition from previous bar
        3. Generate signal based on transition and current position:
           - Regime change to UPTREND (1) + No position -> BUY
           - Regime change to RANGING (0) or DOWNTREND (2) + In position -> SELL
           - Otherwise -> HOLD

        Args:
            row: Current market data row with 'regime' column.
                 The regime column should contain:
                 - 0: Ranging market
                 - 1: Uptrend
                 - 2: Downtrend
            position: Current position state:
                     - 0: No position (flat)
                     - 1: Long position (in trade)

        Returns:
            Trading signal: 'buy', 'sell', or 'hold'

        Example:
            >>> strategy = RegimeStrategy('default')
            >>> # First call establishes baseline
            >>> signal = strategy.generate_signal(row, position=0)
            >>> # Subsequent calls detect regime changes
            >>> signal = strategy.generate_signal(next_row, position=0)
            >>> # If regime changed to uptrend: signal == 'buy'

        Notes:
            - On first call, sets previous_regime and returns 'hold'
            - Requires 'regime' column in row (can be added via detect_regimes)
            - Only generates buy signal when regime changes to uptrend
            - Generates sell signal when regime changes away from uptrend
        """
        # Check if regime column exists
        if 'regime' not in row:
            raise ValueError(
                "Row must contain 'regime' column. "
                "Use detect_regimes() to add regime detection to your DataFrame."
            )

        current_regime = int(row['regime'])

        # On first bar, initialize previous regime
        if self.previous_regime is None:
            self.previous_regime = current_regime
            return 'hold'

        # Detect regime change
        regime_changed = current_regime != self.previous_regime

        # Generate signal based on regime change and position
        signal = 'hold'

        if regime_changed:
            # Regime changed to UPTREND and we're not in position -> BUY
            if current_regime == 1 and position == 0:
                signal = 'buy'

            # Regime changed to RANGING or DOWNTREND and we're in position -> SELL
            elif current_regime in [0, 2] and position == 1:
                signal = 'sell'

        # Update previous regime for next call
        self.previous_regime = current_regime

        return signal

    def reset(self) -> None:
        """
        Reset the strategy's internal state.

        Clears the previous regime tracking. Should be called at the
        beginning of a backtest or when starting a new trading session.

        Example:
            >>> strategy = RegimeStrategy('default')
            >>> strategy.reset()  # Clear state before backtest
        """
        self.previous_regime = None

    def get_name(self) -> str:
        """
        Get the name of the strategy including config.

        Returns:
            String name in format: 'RegimeStrategy(config_name)'

        Example:
            >>> strategy = RegimeStrategy('aggressive')
            >>> strategy.get_name()
            'RegimeStrategy(aggressive)'
        """
        return f"RegimeStrategy({self.regime_config_name})"

    def get_regime_params(self) -> dict:
        """
        Get the regime detection parameters being used.

        Returns:
            Dictionary of regime detection parameters including:
            - bullish_threshold_up: Uptrend detection threshold
            - bullish_threshold_down: Downtrend detection threshold
            - price_pos_up: Price position for uptrend
            - price_pos_down: Price position for downtrend
            - spread_min: Minimum MA spread
            - min_bars: Minimum regime duration
            - hysteresis: Bars to confirm regime change
            - use_ema_smooth: Whether EMA smoothing is enabled
            - ema_span: EMA smoothing period

        Example:
            >>> strategy = RegimeStrategy('default')
            >>> params = strategy.get_regime_params()
            >>> print(params['bullish_threshold_up'])
            0.75
        """
        return self.regime_params.copy()

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"RegimeStrategy("
            f"config='{self.regime_config_name}', "
            f"params={self.regime_params})"
        )


# === HELPER FUNCTION ===

def apply_regime_detection(
    df: pd.DataFrame,
    regime_config_name: str = 'default',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Helper function to apply regime detection to a DataFrame.

    This is a convenience wrapper around the detect_regimes function
    that automatically loads the specified configuration.

    Args:
        df: DataFrame with required feature columns:
            - bullish_pct_sma: Percentage of MAs with positive slope
            - price_position: Price position in MA bundle (0-1)
            - spread_pct: MA spread percentage
        regime_config_name: Name of the regime configuration to use
                           ('default', 'aggressive', or 'conservative')
        inplace: If True, modify df in place. Otherwise return a copy.

    Returns:
        DataFrame with added 'regime' and 'raw_regime' columns

    Example:
        >>> import pandas as pd
        >>> from src.strategies.regime_strategy import apply_regime_detection
        >>> df = pd.read_parquet('features.parquet')
        >>> df = apply_regime_detection(df, 'default')
        >>> print(df['regime'].value_counts())
        0    1500  # Ranging
        1     800  # Uptrend
        2     700  # Downtrend
    """
    params = load_regime_config(regime_config_name)
    return detect_regimes(df, params, inplace=inplace)


# === USAGE EXAMPLE ===

if __name__ == "__main__":
    """
    Example usage of the RegimeStrategy.
    """
    print("=" * 60)
    print("RegimeStrategy Example")
    print("=" * 60)

    # Create sample data with regime
    print("\nCreating sample market data...")
    np.random.seed(42)
    n = 100

    sample_data = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 50000,
        'high': np.random.randn(n).cumsum() + 50100,
        'low': np.random.randn(n).cumsum() + 49900,
        'close': np.random.randn(n).cumsum() + 50000,
        'bullish_pct_sma': np.random.uniform(0, 1, n),
        'price_position': np.random.uniform(0, 1, n),
        'spread_pct': np.random.uniform(0.1, 1.5, n),
    })

    # Apply regime detection
    print("Applying regime detection with 'default' config...")
    sample_data = apply_regime_detection(sample_data, 'default')

    print(f"\nRegime distribution:")
    print(sample_data['regime'].value_counts().sort_index())

    # Test strategy with different configs
    for config_name in ['default', 'aggressive', 'conservative']:
        print(f"\n{'-' * 60}")
        print(f"Testing with '{config_name}' configuration")
        print(f"{'-' * 60}")

        # Create strategy
        strategy = RegimeStrategy(config_name)
        print(f"Strategy: {strategy.get_name()}")

        # Re-apply regime detection with this config
        df = apply_regime_detection(sample_data, config_name)

        # Simulate trading
        position = 0
        signals = []

        for idx, row in df.iterrows():
            signal = strategy.generate_signal(row, position)
            signals.append(signal)

            # Update position based on signal
            if signal == 'buy':
                position = 1
                print(f"Bar {idx}: BUY signal (regime={int(row['regime'])})")
            elif signal == 'sell':
                position = 0
                print(f"Bar {idx}: SELL signal (regime={int(row['regime'])})")

        # Show summary
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        print(f"\nSummary:")
        print(f"  Total signals: {len(signals)}")
        print(f"  Buy signals: {buy_count}")
        print(f"  Sell signals: {sell_count}")
        print(f"  Hold signals: {signals.count('hold')}")

        # Reset for next test
        strategy.reset()

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nTo use in your code:")
    print("""
from src.strategies import RegimeStrategy
from src.strategies.regime_strategy import apply_regime_detection

# Load your feature data
df = pd.read_parquet('features.parquet')

# Apply regime detection
df = apply_regime_detection(df, 'default')

# Create strategy
strategy = RegimeStrategy('default')

# Generate signals
for idx, row in df.iterrows():
    signal = strategy.generate_signal(row, position)
    # Execute trades based on signal...
    """)
