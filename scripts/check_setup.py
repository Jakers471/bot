#!/usr/bin/env python3
"""
Setup Checker for MAR Trading Bot

This script verifies that all required components are in place
and provides guidance on what steps need to be completed.

Usage:
    python scripts/check_setup.py
    python scripts/check_setup.py --interval 1h --symbol BTCUSDT
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, FEATURES_DIR, MODELS_DIR, LABELS_DIR,
    SYMBOL, INTERVALS
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check MAR trading bot setup'
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        choices=INTERVALS,
        help='Time interval to check (default: 1h)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Trading symbol to check (default: {SYMBOL})'
    )

    return parser.parse_args()


def check_directories():
    """Check if required directories exist."""
    print("Checking directories...")
    print("-" * 60)

    dirs = {
        'Data': DATA_DIR,
        'Features': FEATURES_DIR,
        'Models': MODELS_DIR,
        'Labels': LABELS_DIR,
    }

    all_exist = True
    for name, path in dirs.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {name:12} {path}: {status}")
        if not exists:
            all_exist = False

    print("-" * 60)
    return all_exist


def check_data(symbol, interval):
    """Check if data files exist."""
    print(f"\nChecking data files for {symbol} {interval}...")
    print("-" * 60)

    files = {
        'Raw Data (parquet)': DATA_DIR / symbol / f"{symbol}_{interval}.parquet",
        'Raw Data (csv)': DATA_DIR / symbol / f"{symbol}_{interval}.csv",
        'Features (parquet)': FEATURES_DIR / f"{symbol}_{interval}_features.parquet",
        'Features (csv)': FEATURES_DIR / f"{symbol}_{interval}_features.csv",
        'Labels': LABELS_DIR / f"{symbol}_{interval}_labels.csv",
        'Model': MODELS_DIR / f"{symbol}_{interval}_model.joblib",
    }

    status_map = {}
    for name, path in files.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        size = f"({path.stat().st_size / 1024 / 1024:.2f} MB)" if exists else ""
        print(f"  {name:20} {status:8} {size}")
        status_map[name] = exists

    print("-" * 60)
    return status_map


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\nChecking Python dependencies...")
    print("-" * 60)

    packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations',
        'joblib': 'Model serialization',
        'sklearn': 'Machine learning (scikit-learn)',
        'xgboost': 'XGBoost models',
        'ccxt': 'Cryptocurrency exchange (for live trading)',
        'plotly': 'Visualization (for backtesting)',
    }

    all_installed = True
    for package, description in packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  {package:15} OK     - {description}")
        except ImportError:
            print(f"  {package:15} MISSING - {description}")
            all_installed = False

    print("-" * 60)
    return all_installed


def provide_next_steps(status_map, interval, symbol):
    """Provide guidance on next steps."""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    has_raw_data = status_map.get('Raw Data (parquet)', False) or status_map.get('Raw Data (csv)', False)
    has_features = status_map.get('Features (parquet)', False) or status_map.get('Features (csv)', False)
    has_labels = status_map.get('Labels', False)
    has_model = status_map.get('Model', False)

    steps = []

    if not has_raw_data:
        steps.append({
            'step': 1,
            'name': 'Download Historical Data',
            'command': f'python binance_downloader.py',
            'description': 'Download OHLCV data from Binance'
        })

    if has_raw_data and not has_features:
        steps.append({
            'step': len(steps) + 1,
            'name': 'Calculate Features',
            'command': f'python src/mar_calculator.py',
            'description': 'Calculate MAR indicator features'
        })

    if has_features and not has_labels:
        steps.append({
            'step': len(steps) + 1,
            'name': 'Generate Labels',
            'command': f'python src/labeler.py',
            'description': 'Label data for supervised learning'
        })

    if has_features and has_labels and not has_model:
        steps.append({
            'step': len(steps) + 1,
            'name': 'Train Model',
            'command': f'python scripts/train.py --interval {interval} --model xgboost',
            'description': 'Train machine learning model'
        })

    if has_model:
        steps.append({
            'step': len(steps) + 1,
            'name': 'Backtest Strategy',
            'command': f'python scripts/backtest.py --interval {interval}',
            'description': 'Test model performance on historical data'
        })

        steps.append({
            'step': len(steps) + 1,
            'name': 'Paper Trade',
            'command': f'python scripts/bot.py --symbol {symbol} --interval {interval} --paper',
            'description': 'Test bot in paper trading mode'
        })

        steps.append({
            'step': len(steps) + 1,
            'name': 'Live Trade (when ready!)',
            'command': f'python scripts/bot.py --symbol {symbol} --interval {interval} --live',
            'description': 'Run bot with real money (use caution!)'
        })

    if steps:
        for step in steps:
            print(f"\n{step['step']}. {step['name']}")
            print(f"   {step['description']}")
            print(f"   Command: {step['command']}")
    else:
        print("\nAll components are in place!")
        print("You can proceed with backtesting or trading.")

    print("=" * 60)


def main():
    """Main check function."""
    args = parse_args()

    print()
    print("=" * 60)
    print("MAR TRADING BOT - SETUP CHECKER")
    print("=" * 60)
    print(f"Symbol:   {args.symbol}")
    print(f"Interval: {args.interval}")
    print("=" * 60)
    print()

    # Check directories
    dirs_ok = check_directories()

    # Check dependencies
    deps_ok = check_dependencies()

    # Check data files
    status_map = check_data(args.symbol, args.interval)

    # Provide next steps
    provide_next_steps(status_map, args.interval, args.symbol)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not dirs_ok:
        print("Some directories are missing (they will be created automatically)")

    if not deps_ok:
        print("\nWARNING: Some Python packages are missing")
        print("Install missing packages with:")
        print("  pip install pandas numpy joblib scikit-learn xgboost ccxt plotly")

    print("\nFor detailed documentation, see:")
    print("  scripts/README.md")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
