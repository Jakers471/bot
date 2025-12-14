#!/usr/bin/env python3
"""
Visualize Backtest Results

This script loads backtest results and generates publication-quality charts.

Usage:
    python scripts/visualize_backtest.py --interval 1h
    python scripts/visualize_backtest.py --interval 1h --config default
    python scripts/visualize_backtest.py --results results/backtest_summary_BTCUSDT_1h_20241201_120000.json
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SYMBOL, INTERVALS, FEATURES_DIR
from src.visualizer import plot_all_from_backtest, plot_trades, plot_equity_curve, plot_comparison
from scripts.run_backtest import detect_regimes, load_regime_configs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate visualization charts from backtest results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize latest backtest results for 1h interval
  python scripts/visualize_backtest.py --interval 1h

  # Visualize specific config only
  python scripts/visualize_backtest.py --interval 1h --config aggressive

  # Visualize from specific results file
  python scripts/visualize_backtest.py --results results/backtest_summary_BTCUSDT_1h_20241201.json

  # Custom output directory
  python scripts/visualize_backtest.py --interval 1h --output charts/custom
        """
    )

    parser.add_argument(
        '--interval',
        type=str,
        choices=INTERVALS,
        help='Time interval (required if --results not specified)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Trading symbol (default: {SYMBOL})'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Specific config to visualize (default: all configs)'
    )

    parser.add_argument(
        '--results',
        type=str,
        help='Path to specific backtest results JSON file'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for charts (default: charts/SYMBOL_INTERVAL_TIMESTAMP)'
    )

    parser.add_argument(
        '--no-mas',
        action='store_true',
        help='Hide moving averages on trade charts (faster rendering for large datasets)'
    )

    parser.add_argument(
        '--max-bars',
        type=int,
        default=1000,
        help='Maximum bars to show on trade charts (default: 1000, use -1 for all)'
    )

    return parser.parse_args()


def load_latest_results(symbol: str, interval: str) -> dict:
    """Load the most recent backtest results for given symbol and interval."""
    results_dir = PROJECT_ROOT / 'results'

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Run: python scripts/run_backtest.py --interval {interval}"
        )

    # Find all matching summary files
    pattern = f"backtest_summary_{symbol}_{interval}_*.json"
    summary_files = sorted(results_dir.glob(pattern), reverse=True)

    if not summary_files:
        raise FileNotFoundError(
            f"No backtest results found for {symbol} {interval}\n"
            f"Run: python scripts/run_backtest.py --interval {interval}"
        )

    # Return the most recent
    return summary_files[0]


def load_individual_results(summary_path: Path) -> dict:
    """Load individual result files referenced in summary."""
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    metadata = summary.get('metadata', {})
    symbol = metadata.get('symbol', SYMBOL)
    interval = metadata.get('interval', '1h')
    timestamp = metadata.get('timestamp', '')

    results_dir = summary_path.parent
    results = {}

    # Load each config's full results
    for config_name in summary.get('configs', {}).keys():
        pattern = f"backtest_{symbol}_{interval}_{config_name}_{timestamp}.json"
        result_files = list(results_dir.glob(pattern))

        if result_files:
            with open(result_files[0], 'r') as f:
                results[config_name] = json.load(f)

    return results, metadata


def main():
    """Main visualization execution."""
    args = parse_args()

    print("=" * 100)
    print("MAR TRADING BOT - BACKTEST VISUALIZATION")
    print("=" * 100)

    try:
        # Determine which results to load
        if args.results:
            summary_path = Path(args.results)
            if not summary_path.exists():
                raise FileNotFoundError(f"Results file not found: {summary_path}")
        else:
            if not args.interval:
                raise ValueError("Must specify --interval or --results")
            summary_path = load_latest_results(args.symbol, args.interval)

        print(f"Loading results from: {summary_path}")

        # Load results
        results_dict, metadata = load_individual_results(summary_path)

        if not results_dict:
            raise ValueError(f"No individual results found for {summary_path}")

        symbol = metadata.get('symbol', args.symbol)
        interval = metadata.get('interval', args.interval)
        timestamp = metadata.get('timestamp', '')

        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Configs: {list(results_dict.keys())}")

        # Filter to specific config if requested
        if args.config:
            if args.config not in results_dict:
                raise ValueError(
                    f"Config '{args.config}' not found. "
                    f"Available: {list(results_dict.keys())}"
                )
            results_dict = {args.config: results_dict[args.config]}
            print(f"Filtering to config: {args.config}")

        # Load features data with regimes
        features_path = FEATURES_DIR / f"{symbol}_{interval}_features.parquet"
        if not features_path.exists():
            features_path = FEATURES_DIR / f"{symbol}_{interval}_features.csv"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_path}\n"
                f"Run: python src/mar_calculator.py"
            )

        print(f"Loading features from: {features_path}")
        if features_path.suffix == '.parquet':
            df = pd.read_parquet(features_path)
        else:
            df = pd.read_csv(features_path)

        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)
        print(f"Loaded {len(df):,} bars")

        # Apply regime detection for the first config (for visualization)
        first_config_name = list(results_dict.keys())[0]
        configs = load_regime_configs()

        if first_config_name not in configs:
            print(f"WARNING: Config '{first_config_name}' not found, using default")
            config_params = configs.get('default', {})
        else:
            config_params = configs[first_config_name]

        print(f"Applying regime detection with config: {first_config_name}")
        df_with_regime = detect_regimes(df, config_params)

        # Limit bars for visualization if requested
        if args.max_bars > 0 and len(df_with_regime) > args.max_bars:
            print(f"Limiting to last {args.max_bars} bars for visualization")
            df_display = df_with_regime.iloc[-args.max_bars:].reset_index(drop=True)

            # Adjust trade indices to match display dataframe
            offset = len(df_with_regime) - len(df_display)
            for config_name, result in results_dict.items():
                trades = result.get('trades', [])
                filtered_trades = []

                for trade in trades:
                    entry_idx = trade.get('entry_idx')
                    exit_idx = trade.get('exit_idx')

                    # Only include trades that are visible in display range
                    if entry_idx is not None and entry_idx >= offset:
                        trade_copy = trade.copy()
                        trade_copy['entry_idx'] = entry_idx - offset
                        if exit_idx is not None:
                            trade_copy['exit_idx'] = exit_idx - offset
                        filtered_trades.append(trade_copy)

                result['trades'] = filtered_trades
        else:
            df_display = df_with_regime

        # Load buy-and-hold benchmark if available
        buy_and_hold = None
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                # Try to find BH data in metadata or first result
                if results_dict:
                    first_result = list(results_dict.values())[0]
                    if 'buy_and_hold' in first_result:
                        buy_and_hold = first_result['buy_and_hold']

        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = PROJECT_ROOT / 'charts' / f"{symbol}_{interval}_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Generate all charts
        print("\n" + "=" * 100)
        plot_all_from_backtest(
            df=df_display,
            results_dict=results_dict,
            output_dir=output_dir,
            buy_and_hold=buy_and_hold,
            show_mas=not args.no_mas
        )

        print("\n" + "=" * 100)
        print("VISUALIZATION COMPLETE!")
        print("=" * 100)
        print(f"\nCharts saved to: {output_dir.absolute()}")

        # List generated files
        chart_files = sorted(output_dir.glob('*.png'))
        if chart_files:
            print(f"\nGenerated {len(chart_files)} charts:")
            for chart_file in chart_files:
                print(f"  - {chart_file.name}")

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    except ValueError as e:
        print(f"\nERROR: {e}")
        return 1

    except Exception as e:
        print(f"\nERROR: Visualization failed")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
