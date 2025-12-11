"""
Moving Average Ribbon (MAR) Calculator

This module calculates the Moving Average Ribbon indicator and derived features.
It processes BTCUSDT price data across multiple timeframes and generates technical
indicators for machine learning model training.

Features Calculated:
-------------------
1. Moving Averages (96 total):
   - SMA (Simple Moving Average): 32 periods (5-36)
   - EMA (Exponential Moving Average): 32 periods (5-36)
   - WMA (Weighted Moving Average): 32 periods (5-36)

2. Spread Features (4 total):
   - spread_sma/ema/wma: Difference between fastest (MA5) and slowest (MA36) MA
   - spread_pct: Spread as percentage of close price

3. Compression/Range Features (9 total):
   - ma_std_{type}: Standard deviation across all 32 MAs (compression metric)
   - ma_range_{type}: Max MA - Min MA for each type (ribbon width)
   - ma_range_pct: Range as percentage of close price
   - compression: Boolean, True if ma_range_pct < 0.5%
   - expansion: Boolean, True if ma_range_pct is increasing

4. Slope Features (99 total):
   - slope_ma{N}_{type}: (MA - MA.shift(5)) / 5 for each MA (momentum/angle)
   - avg_slope_{type}: Average slope across all MAs of that type

5. Price Position Features (3 total):
   - price_position: Where close price is relative to MA bundle (0=below all, 1=above all)
   - cross_above: Boolean, price crossed above the top MA
   - cross_below: Boolean, price crossed below the bottom MA

Total: 211 features + 12 OHLCV columns = 223 columns

Usage:
------
Run as script to process all intervals:
    python mar_calculator.py

Or import and use programmatically:
    from mar_calculator import process_interval, load_features
    process_interval('1h')
    df = load_features('BTCUSDT', '1h')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings

try:
    from src.config import (
        DATA_DIR,
        FEATURES_DIR,
        SYMBOL,
        INTERVALS,
        MA_TYPES,
        MA_START,
        MA_COUNT,
        MA_STEP,
        SLOPE_LOOKBACK,
        COMPRESSION_THRESHOLD,
    )
except ImportError:
    from config import (
        DATA_DIR,
        FEATURES_DIR,
        SYMBOL,
        INTERVALS,
        MA_TYPES,
        MA_START,
        MA_COUNT,
        MA_STEP,
        SLOPE_LOOKBACK,
        COMPRESSION_THRESHOLD,
    )

warnings.filterwarnings('ignore')


def calculate_wma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Weighted Moving Average (WMA).

    WMA gives more weight to recent prices.
    Weight for position i (where 0 is most recent) = (period - i) / sum(1..period)

    Args:
        series: Price series
        period: MA period

    Returns:
        WMA series
    """
    weights = np.arange(1, period + 1)

    def wma_at_point(x):
        if len(x) < period:
            return np.nan
        return np.dot(x[-period:], weights) / weights.sum()

    return series.rolling(window=period).apply(wma_at_point, raw=True)


def calculate_mas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all Moving Averages (SMA, EMA, WMA) for periods 5-36.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all MA columns added
    """
    # Calculate periods (5, 6, 7, ..., 36)
    periods = [MA_START + i * MA_STEP for i in range(MA_COUNT)]

    print(f"  Calculating {len(periods)} MAs for each of {len(MA_TYPES)} types...")

    for ma_type in MA_TYPES:
        print(f"    Processing {ma_type}...")
        for period in periods:
            col_name = f"ma{period}_{ma_type.lower()}"

            if ma_type == "SMA":
                df[col_name] = df['close'].rolling(window=period).mean()
            elif ma_type == "EMA":
                df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
            elif ma_type == "WMA":
                df[col_name] = calculate_wma(df['close'], period)

    return df


def calculate_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate spread features (fastest MA - slowest MA).

    Args:
        df: DataFrame with MA columns

    Returns:
        DataFrame with spread features added
    """
    periods = [MA_START + i * MA_STEP for i in range(MA_COUNT)]
    fastest_period = min(periods)
    slowest_period = max(periods)

    for ma_type in MA_TYPES:
        ma_type_lower = ma_type.lower()
        fastest_col = f"ma{fastest_period}_{ma_type_lower}"
        slowest_col = f"ma{slowest_period}_{ma_type_lower}"

        # Absolute spread
        df[f"spread_{ma_type_lower}"] = df[fastest_col] - df[slowest_col]

    # Spread as percentage of close price
    df['spread_pct'] = (df['spread_sma'].abs() / df['close']) * 100

    return df


def calculate_compression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MA compression/expansion features.

    Args:
        df: DataFrame with MA columns

    Returns:
        DataFrame with compression features added
    """
    periods = [MA_START + i * MA_STEP for i in range(MA_COUNT)]

    for ma_type in MA_TYPES:
        ma_type_lower = ma_type.lower()

        # Get all MA columns for this type
        ma_cols = [f"ma{p}_{ma_type_lower}" for p in periods]
        ma_values = df[ma_cols]

        # Standard deviation across all MAs (compression metric)
        df[f"ma_std_{ma_type_lower}"] = ma_values.std(axis=1)

        # Range (max - min) across all MAs (ribbon width)
        df[f"ma_range_{ma_type_lower}"] = ma_values.max(axis=1) - ma_values.min(axis=1)

    # Range as percentage of close price
    df['ma_range_pct'] = (df['ma_range_sma'] / df['close']) * 100

    # Compression: range < threshold
    df['compression'] = (df['ma_range_pct'] < COMPRESSION_THRESHOLD).astype(int)

    # Expansion: range increasing
    df['expansion'] = (df['ma_range_pct'] > df['ma_range_pct'].shift(1)).astype(int)

    return df


def calculate_slope_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MA slope features (momentum/angle).

    Slope = (MA - MA.shift(lookback)) / lookback

    Args:
        df: DataFrame with MA columns

    Returns:
        DataFrame with slope features added
    """
    periods = [MA_START + i * MA_STEP for i in range(MA_COUNT)]

    for ma_type in MA_TYPES:
        ma_type_lower = ma_type.lower()
        slopes = []

        for period in periods:
            ma_col = f"ma{period}_{ma_type_lower}"
            slope_col = f"slope_ma{period}_{ma_type_lower}"

            # Calculate slope
            df[slope_col] = (
                df[ma_col] - df[ma_col].shift(SLOPE_LOOKBACK)
            ) / SLOPE_LOOKBACK

            slopes.append(slope_col)

        # Average slope across all MAs of this type
        df[f"avg_slope_{ma_type_lower}"] = df[slopes].mean(axis=1)

        # Bullish/Bearish counts (positive slope = bullish/rising, negative = bearish/falling)
        slope_values = df[slopes]
        df[f"bullish_count_{ma_type_lower}"] = (slope_values > 0).sum(axis=1)
        df[f"bearish_count_{ma_type_lower}"] = (slope_values < 0).sum(axis=1)
        df[f"bullish_pct_{ma_type_lower}"] = df[f"bullish_count_{ma_type_lower}"] / len(slopes)

    return df


def calculate_price_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price position relative to MA bundle.

    Args:
        df: DataFrame with MA columns

    Returns:
        DataFrame with price position features added
    """
    periods = [MA_START + i * MA_STEP for i in range(MA_COUNT)]

    # Use SMA for price position (could use any MA type)
    ma_cols = [f"ma{p}_sma" for p in periods]
    ma_values = df[ma_cols]

    # Count how many MAs are below the close price
    mas_below = (ma_values.T < df['close'].values).T.sum(axis=1)

    # Normalize to 0-1 range (0 = below all MAs, 1 = above all MAs)
    df['price_position'] = mas_below / len(ma_cols)

    # Cross above: price crossed above the top MA
    top_ma = ma_values.max(axis=1)
    df['cross_above'] = (
        (df['close'] > top_ma) &
        (df['close'].shift(1) <= top_ma.shift(1))
    ).astype(int)

    # Cross below: price crossed below the bottom MA
    bottom_ma = ma_values.min(axis=1)
    df['cross_below'] = (
        (df['close'] < bottom_ma) &
        (df['close'].shift(1) >= bottom_ma.shift(1))
    ).astype(int)

    return df


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all MAR indicator features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all features
    """
    print("  Calculating Moving Averages...")
    df = calculate_mas(df)

    print("  Calculating spread features...")
    df = calculate_spread_features(df)

    print("  Calculating compression features...")
    df = calculate_compression_features(df)

    print("  Calculating slope features...")
    df = calculate_slope_features(df)

    print("  Calculating price position features...")
    df = calculate_price_position_features(df)

    return df


def process_interval(interval: str) -> None:
    """
    Process a single interval: load data, calculate features, save results.

    Args:
        interval: Time interval (e.g., '1m', '5m', '1h')
    """
    print(f"\nProcessing {interval}...")

    # Load data
    input_path = DATA_DIR / SYMBOL / f"{SYMBOL}_{interval}.parquet"
    print(f"  Loading from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows")

    # Calculate features
    df = calculate_all_features(df)

    # Save results
    output_dir = FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{SYMBOL}_{interval}_features.parquet"
    print(f"  Saving to {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df):,} rows with {len(df.columns)} columns")

    # Show sample of NaN values (expected at start due to rolling windows)
    max_period = MA_START + (MA_COUNT - 1) * MA_STEP
    expected_nans = max_period + SLOPE_LOOKBACK
    print(f"  Expected NaN rows: ~{expected_nans} (due to MA warmup period)")

    # Show feature summary
    feature_cols = [col for col in df.columns if col not in [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]]
    print(f"  Generated {len(feature_cols)} features")


def load_features(symbol: str, interval: str) -> pd.DataFrame:
    """
    Load pre-calculated features from file.
    If features don't exist, load raw data and calculate them.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h')

    Returns:
        DataFrame with all features
    """
    features_path = FEATURES_DIR / f"{symbol}_{interval}_features.parquet"

    if features_path.exists():
        print(f"Loading features from {features_path}")
        return pd.read_parquet(features_path)

    # Features don't exist, calculate from raw data
    raw_path = DATA_DIR / symbol / f"{symbol}_{interval}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"No data found at {raw_path}")

    print(f"Features not found, calculating from {raw_path}...")
    df = pd.read_parquet(raw_path)
    df = calculate_all_features(df)

    # Save for next time
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path, index=False)
    print(f"Saved features to {features_path}")

    return df


def load_raw_data(symbol: str, interval: str) -> pd.DataFrame:
    """
    Load raw OHLCV data without features.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h')

    Returns:
        DataFrame with OHLCV data
    """
    raw_path = DATA_DIR / symbol / f"{symbol}_{interval}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"No data found at {raw_path}")
    return pd.read_parquet(raw_path)


def main():
    """
    Main function to process all intervals.
    """
    print("=" * 60)
    print("MAR Feature Calculator")
    print("=" * 60)
    print(f"Symbol: {SYMBOL}")
    print(f"Intervals: {', '.join(INTERVALS)}")
    print(f"MA Types: {', '.join(MA_TYPES)}")
    print(f"MA Periods: {MA_START} to {MA_START + (MA_COUNT - 1) * MA_STEP} ({MA_COUNT} total)")
    print(f"Total MAs per type: {MA_COUNT}")
    print(f"Total MAs: {MA_COUNT * len(MA_TYPES)}")
    print(f"Slope lookback: {SLOPE_LOOKBACK} bars")
    print(f"Compression threshold: {COMPRESSION_THRESHOLD}%")
    print("=" * 60)

    for interval in INTERVALS:
        try:
            process_interval(interval)
        except Exception as e:
            print(f"ERROR processing {interval}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Feature calculation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
