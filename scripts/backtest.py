#!/usr/bin/env python3
"""
Backtesting Script for MAR Trading Bot

This script backtests a trained model by generating predictions and simulating trades.

Usage:
    python scripts/backtest.py --interval 1h --model models/BTCUSDT_1h_model.joblib
    python scripts/backtest.py --interval 1h  # Auto-detect model path

Results are printed to console and optionally saved to CSV.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SYMBOL, INTERVALS, MODELS_DIR, FEATURES_DIR, LABELS_DIR
from src.trainer import prepare_features, predict
from src.backtester import run_backtest, generate_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Backtest MAR trading bot models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest with specific model file
  python scripts/backtest.py --interval 1h --model models/BTCUSDT_1h_model.joblib

  # Auto-detect model path
  python scripts/backtest.py --interval 1h --symbol BTCUSDT

  # Save results to CSV
  python scripts/backtest.py --interval 1h --output backtest_results.csv

  # Custom capital and position size
  python scripts/backtest.py --interval 1h --capital 50000 --position-size 0.2
        """
    )

    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=INTERVALS,
        help='Time interval (e.g., 1m, 5m, 15m, 1h, 4h, 1d)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (e.g., models/BTCUSDT_1h_model.joblib). If not provided, will auto-detect.'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Trading symbol (default: {SYMBOL})'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital in USD (default: 10000)'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=0.1,
        help='Position size as fraction of capital, e.g., 0.1 = 10%% (default: 0.1)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate per trade (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path for results (optional)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD), optional'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD), optional'
    )

    return parser.parse_args()


def load_data_and_model(args):
    """Load features, labels, and model."""
    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = MODELS_DIR / f"{args.symbol}_{args.interval}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Load features
    features_path = FEATURES_DIR / f"{args.symbol}_{args.interval}_features.csv"

    # Try parquet if CSV doesn't exist
    if not features_path.exists():
        features_path = FEATURES_DIR / f"{args.symbol}_{args.interval}_features.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"Loading features from: {features_path}")

    if features_path.suffix == '.parquet':
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)

    # Load labels if they exist separately
    labels_path = LABELS_DIR / f"{args.symbol}_{args.interval}_labels.csv"
    if labels_path.exists() and 'label' not in df.columns:
        print(f"Loading labels from: {labels_path}")
        labels_df = pd.read_csv(labels_path)
        df['label'] = labels_df['label']

    print(f"Loaded {len(df):,} rows")

    return model, df, model_path


def main():
    """Main backtesting function."""
    args = parse_args()

    print("=" * 70)
    print("MAR TRADING BOT - BACKTESTING")
    print("=" * 70)
    print(f"Symbol:          {args.symbol}")
    print(f"Interval:        {args.interval}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Position Size:   {args.position_size * 100:.1f}%")
    print(f"Commission:      {args.commission * 100:.2f}%")
    print("=" * 70)
    print()

    try:
        # Load model and data
        model, df, model_path = load_data_and_model(args)

        # Filter by date if specified
        if args.start_date or args.end_date:
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
                if args.start_date:
                    df = df[df['open_time'] >= args.start_date]
                    print(f"Filtered to start date: {args.start_date}")
                if args.end_date:
                    df = df[df['open_time'] <= args.end_date]
                    print(f"Filtered to end date: {args.end_date}")
                print(f"Date-filtered data: {len(df):,} rows")

        # Drop rows with NaN (due to MA warmup period)
        df_clean = df.dropna()
        print(f"After removing NaN: {len(df_clean):,} rows")

        if len(df_clean) == 0:
            raise ValueError("No valid data rows after cleaning")

        # Prepare features for prediction
        X, feature_names = prepare_features(df_clean)

        print()
        print("Generating predictions...")
        predictions = predict(model, X)
        df_clean['prediction'] = predictions

        # Show prediction distribution
        print()
        print("Prediction Distribution:")
        print("-" * 40)
        pred_counts = df_clean['prediction'].value_counts().sort_index()
        for pred_class, count in pred_counts.items():
            class_names = {0: 'Ranging', 1: 'Trending Up', 2: 'Trending Down'}
            pct = (count / len(df_clean)) * 100
            print(f"  {class_names.get(pred_class, f'Class {pred_class}')}: {count:6,} ({pct:5.2f}%)")
        print("-" * 40)

        # Check required columns for backtesting
        required_cols = ['open_time', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for backtesting: {missing_cols}")

        # Run backtest
        print()
        print("Running backtest...")
        result = run_backtest(
            df_clean,
            initial_capital=args.capital,
            position_size=args.position_size,
            commission=args.commission
        )

        # Print results
        print()
        report = generate_report(result)
        print(report)

        # Show recent trades
        if result.trades:
            print()
            print("RECENT TRADES (Last 10)")
            print("=" * 70)
            print(f"{'Entry Time':<20} {'Exit Time':<20} {'Side':<6} {'PnL':<12} {'PnL %':<10}")
            print("-" * 70)
            for trade in result.trades[-10:]:
                entry_time = str(trade['entry_time'])[:19] if isinstance(trade['entry_time'], pd.Timestamp) else str(trade['entry_time'])
                exit_time = str(trade['exit_time'])[:19] if isinstance(trade['exit_time'], pd.Timestamp) else str(trade['exit_time'])
                print(
                    f"{entry_time:<20} "
                    f"{exit_time:<20} "
                    f"{trade['side']:<6} "
                    f"${trade['pnl']:>10.2f} "
                    f"{trade['pnl_pct']:>9.2f}%"
                )
            print("=" * 70)

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)

            # Save trades to CSV
            trades_df = pd.DataFrame(result.trades)
            trades_df.to_csv(output_path, index=False)
            print()
            print(f"Results saved to: {output_path}")
            print(f"Trades saved: {len(result.trades)}")

            # Also save equity curve
            equity_path = output_path.parent / f"{output_path.stem}_equity.csv"
            result.equity_curve.to_csv(equity_path, header=['equity'])
            print(f"Equity curve saved to: {equity_path}")

        # Summary recommendations
        print()
        print("BACKTEST SUMMARY")
        print("=" * 70)

        metrics = result.metrics

        # Performance assessment
        if metrics['total_return'] > 0:
            performance = "PROFITABLE"
        else:
            performance = "UNPROFITABLE"

        print(f"Overall Performance: {performance}")
        print(f"Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print()

        if metrics['sharpe_ratio'] > 1.5:
            print("Good Sharpe ratio - risk-adjusted returns are strong")
        elif metrics['sharpe_ratio'] > 0.5:
            print("Moderate Sharpe ratio - acceptable risk-adjusted returns")
        else:
            print("Low Sharpe ratio - consider improving strategy or risk management")

        if metrics['max_drawdown'] > 30:
            print("WARNING: High maximum drawdown - consider reducing position size")

        if metrics['win_rate'] < 40:
            print("Low win rate - ensure profit factor is healthy (profitable trades are larger than losses)")

        print("=" * 70)

        return 0

    except FileNotFoundError as e:
        print()
        print("ERROR: Required files not found")
        print("=" * 70)
        print(str(e))
        print()
        print("Make sure you have:")
        print("  1. Trained a model (run scripts/train.py)")
        print("  2. Generated features (run src/mar_calculator.py)")
        print("=" * 70)
        return 1

    except ValueError as e:
        print()
        print("ERROR: Invalid data")
        print("=" * 70)
        print(str(e))
        print("=" * 70)
        return 1

    except Exception as e:
        print()
        print("ERROR: Backtest failed")
        print("=" * 70)
        print(f"{type(e).__name__}: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
