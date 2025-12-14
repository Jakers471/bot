"""
Market Regime Detection Module

This module provides functionality for detecting market regimes (uptrend, ranging, downtrend)
based on moving average analysis and price position indicators.

Regimes:
    0: Ranging - Market is consolidating, no clear trend
    1: Uptrend - Bullish market conditions
    2: Downtrend - Bearish market conditions

The detection uses three main signals:
    - bullish_pct_sma: Percentage of MAs with positive slope
    - price_position: Where price sits relative to MA bundle (0=below all, 1=above all)
    - spread_pct: Width of MA ribbon (filters noise)

Additional features:
    - EMA smoothing: Reduces noise in detection signals
    - Hysteresis: Prevents regime flickering by requiring confirmation
    - Min bars filter: Ensures regimes have minimum duration
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# === CONFIG MANAGEMENT ===

def get_config_path() -> Path:
    """Get path to regime tuner configs file."""
    base_dir = Path(__file__).parent.parent
    return base_dir / "configs" / "regime_tuner_configs.json"


def load_regime_config(name: str = "default") -> Dict:
    """
    Load a regime detection configuration by name.

    Args:
        name: Name of the configuration to load (e.g., 'default', 'aggressive', 'conservative')

    Returns:
        Dictionary containing regime detection parameters:
            - bullish_threshold_up: Threshold for uptrend detection (% of MAs bullish)
            - bullish_threshold_down: Threshold for downtrend detection (% of MAs bullish)
            - price_pos_up: Price position threshold for uptrend (0-1)
            - price_pos_down: Price position threshold for downtrend (0-1)
            - spread_min: Minimum MA spread to confirm trend
            - min_bars: Minimum bars a regime must last
            - hysteresis: Bars needed to confirm regime change
            - use_ema_smooth: Whether to apply EMA smoothing to signals
            - ema_span: EMA span for signal smoothing

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If requested config name doesn't exist

    Example:
        >>> config = load_regime_config('aggressive')
        >>> config['bullish_threshold_up']
        0.6
    """
    config_file = get_config_path()

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Please ensure {config_file.name} exists in the configs directory."
        )

    with open(config_file, 'r') as f:
        configs = json.load(f)

    if name not in configs:
        available = list(configs.keys())
        raise KeyError(
            f"Config '{name}' not found. Available configs: {available}"
        )

    return configs[name]


def get_available_configs() -> List[str]:
    """
    Get list of available regime configuration names.

    Returns:
        List of configuration names available in the config file.
        Returns empty list if config file doesn't exist.

    Example:
        >>> get_available_configs()
        ['default', 'aggressive', 'conservative']
    """
    config_file = get_config_path()

    if not config_file.exists():
        return []

    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
        return list(configs.keys())
    except (json.JSONDecodeError, IOError):
        return []


# === REGIME DETECTION ===

def detect_regimes(
    df: pd.DataFrame,
    params: Dict,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Detect market regimes based on moving average analysis.

    This function analyzes price action and moving average behavior to classify
    market conditions into three regimes: uptrend (1), ranging (0), and downtrend (2).

    The detection process:
    1. Optional EMA smoothing of input signals
    2. Raw regime detection based on thresholds
    3. Hysteresis filter to reduce flickering
    4. Minimum bars filter to ensure regime stability

    Args:
        df: DataFrame with required columns:
            - bullish_pct_sma: Percentage of MAs with positive slope (0-1)
            - price_position: Price position in MA bundle (0-1)
            - spread_pct: MA spread as percentage
        params: Dictionary with detection parameters:
            - bullish_threshold_up: Uptrend threshold (0.5-1.0)
            - bullish_threshold_down: Downtrend threshold (0.0-0.5)
            - price_pos_up: Price position uptrend threshold (0.5-1.0)
            - price_pos_down: Price position downtrend threshold (0.0-0.5)
            - spread_min: Minimum spread to confirm trend (0.0-2.0)
            - min_bars: Minimum regime duration in bars (1-50)
            - hysteresis: Bars to confirm regime change (1-10)
            - use_ema_smooth: Apply EMA smoothing to signals (bool)
            - ema_span: EMA span for smoothing (2-20)
        inplace: If True, modify df in place. Otherwise return a copy.

    Returns:
        DataFrame with added 'regime' column:
            - 0: Ranging market
            - 1: Uptrend
            - 2: Downtrend
        Also includes 'raw_regime' column showing pre-filtered detections.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_parquet('features.parquet')
        >>> config = load_regime_config('default')
        >>> df_with_regimes = detect_regimes(df, config)
        >>> df_with_regimes['regime'].value_counts()
        0    1500  # Ranging
        1     800  # Uptrend
        2     700  # Downtrend

    Notes:
        - Missing feature columns will be filled with neutral values (0.5 for ratios, 1.0 for spread)
        - Hysteresis prevents rapid regime switching
        - Min bars filter removes very short-lived regimes
    """
    if not inplace:
        df = df.copy()

    # Extract parameters
    bull_up = params['bullish_threshold_up']
    bull_dn = params['bullish_threshold_down']
    price_up = params['price_pos_up']
    price_dn = params['price_pos_down']
    spread_min = params['spread_min']
    min_bars = params['min_bars']
    hyst = params['hysteresis']
    use_ema = params['use_ema_smooth']
    ema_span = params['ema_span']

    # Get feature values with defaults if columns don't exist
    bull_pct = df['bullish_pct_sma'].values if 'bullish_pct_sma' in df.columns else np.full(len(df), 0.5)
    price_pos = df['price_position'].values if 'price_position' in df.columns else np.full(len(df), 0.5)
    spread = df['spread_pct'].values if 'spread_pct' in df.columns else np.full(len(df), 1.0)

    # Apply EMA smoothing if requested
    if use_ema:
        bull_pct = pd.Series(bull_pct).ewm(span=ema_span, adjust=False).mean().values
        price_pos = pd.Series(price_pos).ewm(span=ema_span, adjust=False).mean().values

    # Raw regime detection
    regime = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if bull_pct[i] >= bull_up and price_pos[i] >= price_up and spread[i] >= spread_min:
            regime[i] = 1  # Uptrend
        elif bull_pct[i] <= bull_dn and price_pos[i] <= price_dn and spread[i] >= spread_min:
            regime[i] = 2  # Downtrend
        else:
            regime[i] = 0  # Ranging

    df['raw_regime'] = regime.copy()

    # Apply hysteresis filter
    # Prevents rapid switching by requiring N consecutive bars to confirm change
    smoothed = regime.copy()
    current_regime = smoothed[0]
    count = 0

    for i in range(1, len(smoothed)):
        if smoothed[i] != current_regime:
            count += 1
            if count >= hyst:
                # Change confirmed
                current_regime = smoothed[i]
                count = 0
            else:
                # Keep previous regime
                smoothed[i] = current_regime
        else:
            count = 0

    # Apply minimum bars filter
    # Merges short regimes with previous regime
    final = smoothed.copy()
    i = 0

    while i < len(final):
        # Find end of current regime
        j = i
        while j < len(final) and final[j] == final[i]:
            j += 1

        # If regime is too short and not at start, merge with previous
        if j - i < min_bars and i > 0:
            final[i:j] = final[i-1]

        i = j

    df['regime'] = final

    return df


def validate_params(params: Dict) -> bool:
    """
    Validate regime detection parameters.

    Args:
        params: Parameter dictionary to validate

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid with description of issue
    """
    required_keys = [
        'bullish_threshold_up', 'bullish_threshold_down',
        'price_pos_up', 'price_pos_down', 'spread_min',
        'min_bars', 'hysteresis', 'use_ema_smooth', 'ema_span'
    ]

    # Check all required keys present
    missing = [k for k in required_keys if k not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Validate ranges
    if not 0.5 <= params['bullish_threshold_up'] <= 1.0:
        raise ValueError("bullish_threshold_up must be between 0.5 and 1.0")

    if not 0.0 <= params['bullish_threshold_down'] <= 0.5:
        raise ValueError("bullish_threshold_down must be between 0.0 and 0.5")

    if not 0.5 <= params['price_pos_up'] <= 1.0:
        raise ValueError("price_pos_up must be between 0.5 and 1.0")

    if not 0.0 <= params['price_pos_down'] <= 0.5:
        raise ValueError("price_pos_down must be between 0.0 and 0.5")

    if params['spread_min'] < 0:
        raise ValueError("spread_min must be non-negative")

    if params['min_bars'] < 1:
        raise ValueError("min_bars must be at least 1")

    if params['hysteresis'] < 1:
        raise ValueError("hysteresis must be at least 1")

    if not isinstance(params['use_ema_smooth'], bool):
        raise ValueError("use_ema_smooth must be boolean")

    if params['ema_span'] < 2:
        raise ValueError("ema_span must be at least 2")

    return True


# === USAGE EXAMPLE ===

if __name__ == "__main__":
    """
    Example usage of the regime detector module.
    """
    print("Market Regime Detector Module")
    print("=" * 50)

    # Show available configs
    configs = get_available_configs()
    print(f"\nAvailable configurations: {configs}")

    # Load a config
    if configs:
        config_name = configs[0]
        config = load_regime_config(config_name)
        print(f"\nLoaded '{config_name}' config:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Validate it
        try:
            validate_params(config)
            print("\n[OK] Configuration is valid")
        except ValueError as e:
            print(f"\n[ERROR] Configuration error: {e}")

    print("\n" + "=" * 50)
    print("To use this module in your code:")
    print("""
from src.regime_detector import detect_regimes, load_regime_config

# Load configuration
config = load_regime_config('default')

# Detect regimes
df = detect_regimes(df, config)

# Check regime distribution
print(df['regime'].value_counts())
    """)
