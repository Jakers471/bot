#!/usr/bin/env python3
"""
Training Script for MAR Trading Bot

This script trains machine learning models to predict market regimes.

Usage:
    python scripts/train.py --interval 1h --model xgboost
    python scripts/train.py --interval 4h --model randomforest --symbol ETHUSDT

Models are saved to models/ directory.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import train_pipeline, get_feature_importance
from src.config import SYMBOL, INTERVALS, MODELS_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MAR trading bot models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost model for 1h interval
  python scripts/train.py --interval 1h --model xgboost

  # Train RandomForest for 4h interval
  python scripts/train.py --interval 4h --model randomforest

  # Train for different symbol
  python scripts/train.py --interval 1h --model xgboost --symbol ETHUSDT

  # Don't save model (just evaluate)
  python scripts/train.py --interval 1h --model xgboost --no-save
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
        default='xgboost',
        choices=['xgboost', 'randomforest'],
        help='Model type to train (default: xgboost)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Trading symbol (default: {SYMBOL})'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the trained model'
    )

    parser.add_argument(
        '--top-features',
        type=int,
        default=20,
        help='Number of top features to display (default: 20)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 70)
    print("MAR TRADING BOT - MODEL TRAINING")
    print("=" * 70)
    print(f"Symbol:          {args.symbol}")
    print(f"Interval:        {args.interval}")
    print(f"Model Type:      {args.model.upper()}")
    print(f"Save Model:      {not args.no_save}")
    print("=" * 70)
    print()

    try:
        # Run training pipeline
        result = train_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            model_type=args.model,
            save=not args.no_save
        )

        print()
        print("=" * 70)
        print("TRAINING METRICS")
        print("=" * 70)
        metrics = result['metrics']
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1 Score:        {metrics['f1']:.4f}")
        print()
        print("Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"                 Predicted")
        print(f"                 Ranging  Trending↑ Trending↓")
        print(f"Actual Ranging     {cm[0][0]:4d}      {cm[0][1]:4d}      {cm[0][2]:4d}")
        print(f"       Trending↑   {cm[1][0]:4d}      {cm[1][1]:4d}      {cm[1][2]:4d}")
        print(f"       Trending↓   {cm[2][0]:4d}      {cm[2][1]:4d}      {cm[2][2]:4d}")
        print("=" * 70)

        # Display top features
        print()
        print(f"TOP {args.top_features} MOST IMPORTANT FEATURES")
        print("=" * 70)
        importance_df = get_feature_importance(
            result['model'],
            result['feature_names']
        )
        print(importance_df.head(args.top_features).to_string(index=False))
        print("=" * 70)

        # Model save info
        if not args.no_save:
            print()
            print("MODEL SAVED")
            print("=" * 70)
            print(f"Location: {result['model_path']}")
            print(f"Size:     {result['model_path'].stat().st_size / 1024:.2f} KB")
            print()
            print("To use this model for backtesting:")
            print(f"  python scripts/backtest.py --interval {args.interval} --model {result['model_path']}")
            print()
            print("To use this model for live trading:")
            print(f"  python scripts/bot.py --symbol {args.symbol} --interval {args.interval} --paper")
            print("=" * 70)

        print()
        print("TRAINING COMPLETE!")
        print()

        return 0

    except FileNotFoundError as e:
        print()
        print("ERROR: Required data files not found")
        print("=" * 70)
        print(str(e))
        print()
        print("Make sure you have:")
        print("  1. Downloaded historical data (run binance_downloader.py)")
        print("  2. Calculated features (run src/mar_calculator.py)")
        print("  3. Generated labels (run src/labeler.py)")
        print("=" * 70)
        return 1

    except ValueError as e:
        print()
        print("ERROR: Invalid data or configuration")
        print("=" * 70)
        print(str(e))
        print("=" * 70)
        return 1

    except Exception as e:
        print()
        print("ERROR: Training failed")
        print("=" * 70)
        print(f"{type(e).__name__}: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
