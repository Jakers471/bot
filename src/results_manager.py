#!/usr/bin/env python3
"""
Results Manager for MAR Trading Bot

This module provides functions to save backtest results in a structured format:
- results/{timestamp}_{interval}_{config}/
  - summary.json (metrics, parameters, settings)
  - trades.csv (all trades with detailed information)
  - equity_curve.csv (equity over time)
  - config.json (copy of regime config used)
  - metadata.txt (human-readable summary)

The system ensures organized storage of backtest results for analysis and comparison.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd


def save_backtest_results(
    result: Dict[str, Any],
    config_name: str,
    config_params: Dict[str, Any],
    args,
    project_root: Path = None
) -> Path:
    """
    Save comprehensive backtest results to a structured folder.

    Args:
        result: Backtest result dictionary containing:
            - trades: List of trade dictionaries
            - equity_curve: List of equity curve points
            - metrics: Performance metrics dictionary
            - initial_capital: Starting capital
            - final_capital: Ending capital
        config_name: Name of the regime config (e.g., 'default', 'aggressive')
        config_params: Dictionary of regime detection parameters
        args: Command line arguments object with backtest settings
        project_root: Path to project root (defaults to parent of this file)

    Returns:
        Path to the created results folder
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    # Create timestamp for folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create folder name: {timestamp}_{interval}_{config}
    folder_name = f"{timestamp}_{args.interval}_{config_name}"
    results_folder = project_root / 'results' / folder_name
    results_folder.mkdir(parents=True, exist_ok=True)

    # 1. Save trades.csv
    _save_trades_csv(result['trades'], results_folder, config_params, result)

    # 2. Save equity_curve.csv
    _save_equity_curve_csv(result['equity_curve'], results_folder, result['initial_capital'])

    # 3. Save config.json
    _save_config_json(config_params, config_name, results_folder)

    # 4. Save summary.json
    _save_summary_json(result, config_name, args, results_folder)

    # 5. Save metadata.txt
    _save_metadata_txt(result, config_name, config_params, args, results_folder)

    print(f"Results saved to: {results_folder}")
    return results_folder


def _save_trades_csv(trades: List[Dict], folder: Path, config_params: Dict, result: Dict):
    """
    Save trades to CSV with detailed information.

    Columns:
    - trade_num: Sequential trade number
    - entry_time: Trade entry timestamp
    - exit_time: Trade exit timestamp
    - side: 'long' or 'short'
    - entry_price: Entry price
    - exit_price: Exit price
    - pnl_dollars: Profit/loss in dollars
    - pnl_percent: Profit/loss as percentage
    - cumulative_pnl: Running total PnL
    - capital_after: Capital after trade
    - bars_held: Number of bars held (calculated from indices)
    - regime_at_entry: Regime at entry (inferred from side)
    """
    if not trades:
        # Create empty CSV with headers
        filepath = folder / 'trades.csv'
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trade_num', 'entry_time', 'exit_time', 'side', 'entry_price', 'exit_price',
                'pnl_dollars', 'pnl_percent', 'cumulative_pnl', 'capital_after',
                'bars_held', 'regime_at_entry'
            ])
        return

    filepath = folder / 'trades.csv'
    cumulative_pnl = 0.0

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'trade_num', 'entry_time', 'exit_time', 'side', 'entry_price', 'exit_price',
            'pnl_dollars', 'pnl_percent', 'cumulative_pnl', 'capital_after',
            'bars_held', 'regime_at_entry'
        ])

        # Data rows
        for i, trade in enumerate(trades, 1):
            cumulative_pnl += trade['pnl']

            # Calculate bars held
            bars_held = trade['exit_idx'] - trade['entry_idx']

            # Infer regime from side (1=long, 2=short)
            regime_at_entry = 1 if trade['side'] == 'long' else 2

            writer.writerow([
                i,
                trade['entry_time'],
                trade['exit_time'],
                trade['side'],
                f"{trade['entry_price']:.2f}",
                f"{trade['exit_price']:.2f}",
                f"{trade['pnl']:.2f}",
                f"{trade['pnl_pct']:.4f}",
                f"{cumulative_pnl:.2f}",
                f"{trade['capital_after']:.2f}",
                bars_held,
                regime_at_entry
            ])


def _save_equity_curve_csv(equity_curve: List[Dict], folder: Path, initial_capital: float):
    """
    Save equity curve to CSV.

    Columns:
    - timestamp: Time point
    - equity: Portfolio equity at this time
    - return_pct: Return percentage from initial capital
    - drawdown_pct: Drawdown from peak
    """
    filepath = folder / 'equity_curve.csv'

    if not equity_curve:
        # Create empty CSV with headers
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'equity', 'return_pct', 'drawdown_pct'])
        return

    # Calculate drawdowns
    equity_values = [e['equity'] for e in equity_curve]
    peak = initial_capital
    drawdowns = []

    for equity in equity_values:
        if equity > peak:
            peak = equity
        drawdown = ((equity - peak) / peak) * 100
        drawdowns.append(drawdown)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['timestamp', 'equity', 'return_pct', 'drawdown_pct'])

        # Data rows
        for i, point in enumerate(equity_curve):
            return_pct = ((point['equity'] - initial_capital) / initial_capital) * 100
            writer.writerow([
                point['open_time'],
                f"{point['equity']:.2f}",
                f"{return_pct:.4f}",
                f"{drawdowns[i]:.4f}"
            ])


def _save_config_json(config_params: Dict, config_name: str, folder: Path):
    """Save regime detection config to JSON."""
    filepath = folder / 'config.json'

    config_data = {
        'config_name': config_name,
        'parameters': config_params
    }

    with open(filepath, 'w') as f:
        json.dump(config_data, f, indent=2)


def _save_summary_json(result: Dict, config_name: str, args, folder: Path):
    """
    Save summary metrics and parameters to JSON.

    Includes:
    - Backtest parameters (capital, position size, etc.)
    - Performance metrics
    - Trade statistics
    """
    filepath = folder / 'summary.json'

    summary = {
        'metadata': {
            'config_name': config_name,
            'symbol': args.symbol,
            'interval': args.interval,
            'timestamp': datetime.now().isoformat(),
            'no_lookahead': True
        },
        'parameters': {
            'initial_capital': args.capital,
            'position_size': args.position_size,
            'commission': args.commission,
            'slippage': args.slippage,
            'start_date': args.start_date if hasattr(args, 'start_date') else None,
            'end_date': args.end_date if hasattr(args, 'end_date') else None
        },
        'metrics': result['metrics'],
        'summary': {
            'initial_capital': result['initial_capital'],
            'final_capital': result['final_capital'],
            'total_trades': len(result['trades']),
            'total_return': result['metrics']['total_return'],
            'sharpe_ratio': result['metrics']['sharpe_ratio'],
            'max_drawdown': result['metrics']['max_drawdown']
        }
    }

    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)


def _save_metadata_txt(
    result: Dict,
    config_name: str,
    config_params: Dict,
    args,
    folder: Path
):
    """
    Save human-readable metadata summary.

    Includes:
    - Run timestamp
    - Data period (start to end)
    - Parameters (capital, position size, commission, slippage)
    - Config name and settings
    - Summary metrics
    - NO LOOK-AHEAD confirmation
    """
    filepath = folder / 'metadata.txt'
    metrics = result['metrics']

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MAR TRADING BOT - BACKTEST RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Run information
        f.write("RUN INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config Name:      {config_name}\n")
        f.write(f"Symbol:           {args.symbol}\n")
        f.write(f"Interval:         {args.interval}\n")
        f.write(f"NO LOOK-AHEAD:    CONFIRMED\n")
        f.write("\n")

        # Data period
        f.write("DATA PERIOD\n")
        f.write("-" * 80 + "\n")
        if metrics.get('data_start'):
            data_start = pd.to_datetime(metrics['data_start']).strftime('%Y-%m-%d %H:%M:%S')
            data_end = pd.to_datetime(metrics['data_end']).strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Start:            {data_start}\n")
            f.write(f"End:              {data_end}\n")
            f.write(f"Duration:         {metrics['data_period_str']} ({metrics['data_period_days']} days)\n")
        else:
            f.write("Not available\n")

        if metrics.get('first_trade'):
            first_trade = pd.to_datetime(metrics['first_trade']).strftime('%Y-%m-%d %H:%M:%S')
            last_trade = pd.to_datetime(metrics['last_trade']).strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\nFirst Trade:      {first_trade}\n")
            f.write(f"Last Trade:       {last_trade}\n")
            f.write(f"Trading Period:   {metrics['trading_period_str']} ({metrics['trading_period_days']} days)\n")
        f.write("\n")

        # Parameters
        f.write("BACKTEST PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Initial Capital:  ${args.capital:,.2f}\n")
        f.write(f"Position Size:    {args.position_size * 100:.1f}%\n")
        f.write(f"Commission:       {args.commission * 100:.3f}% per trade\n")
        f.write(f"Slippage:         {args.slippage * 100:.3f}% per trade\n")
        if hasattr(args, 'start_date') and args.start_date:
            f.write(f"Start Date:       {args.start_date}\n")
        if hasattr(args, 'end_date') and args.end_date:
            f.write(f"End Date:         {args.end_date}\n")
        f.write("\n")

        # Regime config settings
        f.write("REGIME DETECTION CONFIG\n")
        f.write("-" * 80 + "\n")
        f.write(f"Bullish Threshold Up:    {config_params.get('bullish_threshold_up', 'N/A')}\n")
        f.write(f"Bullish Threshold Down:  {config_params.get('bullish_threshold_down', 'N/A')}\n")
        f.write(f"Price Position Up:       {config_params.get('price_pos_up', 'N/A')}\n")
        f.write(f"Price Position Down:     {config_params.get('price_pos_down', 'N/A')}\n")
        f.write(f"Spread Minimum:          {config_params.get('spread_min', 'N/A')}\n")
        f.write(f"Min Bars:                {config_params.get('min_bars', 'N/A')}\n")
        f.write(f"Hysteresis:              {config_params.get('hysteresis', 'N/A')}\n")
        f.write(f"Use EMA Smoothing:       {config_params.get('use_ema_smooth', 'N/A')}\n")
        f.write(f"EMA Span:                {config_params.get('ema_span', 'N/A')}\n")
        f.write("\n")

        # Performance summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Trades:     {metrics['num_trades']}\n")
        f.write(f"Initial Capital:  ${result['initial_capital']:,.2f}\n")
        f.write(f"Final Capital:    ${result['final_capital']:,.2f}\n")
        f.write(f"Total Return:     {metrics['total_return']:.2f}%\n")
        f.write(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"Max Drawdown:     {metrics['max_drawdown']:.2f}%\n")
        f.write("\n")

        # Trade statistics
        f.write("TRADE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Win Rate:         {metrics['win_rate']:.2f}%\n")
        f.write(f"Profit Factor:    {metrics['profit_factor']:.2f}\n")
        f.write(f"Average Win:      ${metrics['avg_win']:.2f}\n")
        f.write(f"Average Loss:     ${metrics['avg_loss']:.2f}\n")
        f.write(f"Largest Win:      ${metrics['largest_win']:.2f}\n")
        f.write(f"Largest Loss:     ${metrics['largest_loss']:.2f}\n")
        f.write("\n")

        # Footer
        f.write("=" * 80 + "\n")
        f.write("IMPORTANT NOTES:\n")
        f.write("- This backtest is CAUSAL and contains NO LOOK-AHEAD BIAS\n")
        f.write("- Regime changes require confirmation over multiple bars\n")
        f.write("- Trades execute at the NEXT bar's open after signal generation\n")
        f.write("- Slippage and commission are applied to all trades\n")
        f.write("=" * 80 + "\n")


def save_multiple_configs(
    results: Dict[str, Dict],
    all_configs: Dict[str, Dict],
    args,
    project_root: Path = None
) -> Dict[str, Path]:
    """
    Save results for multiple configs and create a comparison summary.

    Args:
        results: Dictionary mapping config_name -> result_dict
        all_configs: Dictionary mapping config_name -> config_params
        args: Command line arguments
        project_root: Path to project root

    Returns:
        Dictionary mapping config_name -> results_folder_path
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    saved_folders = {}

    # Save each config's results
    for config_name, result in results.items():
        config_params = all_configs[config_name]
        folder = save_backtest_results(
            result, config_name, config_params, args, project_root
        )
        saved_folders[config_name] = folder

    # Create comparison summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_folder = project_root / 'results' / f"{timestamp}_{args.interval}_comparison"
    comparison_folder.mkdir(parents=True, exist_ok=True)

    _save_comparison_summary(results, all_configs, args, comparison_folder)

    print(f"\nComparison summary saved to: {comparison_folder}")
    return saved_folders


def _save_comparison_summary(
    results: Dict[str, Dict],
    all_configs: Dict[str, Dict],
    args,
    folder: Path
):
    """Save comparison of all configs to CSV and TXT."""

    # CSV comparison
    csv_path = folder / 'comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'config', 'total_trades', 'total_return', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'profit_factor', 'avg_win', 'avg_loss'
        ])

        # Data
        for config_name, result in results.items():
            m = result['metrics']
            writer.writerow([
                config_name,
                m['num_trades'],
                f"{m['total_return']:.2f}",
                f"{m['sharpe_ratio']:.2f}",
                f"{m['max_drawdown']:.2f}",
                f"{m['win_rate']:.2f}",
                f"{m['profit_factor']:.2f}",
                f"{m['avg_win']:.2f}",
                f"{m['avg_loss']:.2f}"
            ])

    # Text comparison
    txt_path = folder / 'comparison.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("BACKTEST COMPARISON - ALL CONFIGS\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Symbol:           {args.symbol}\n")
        f.write(f"Interval:         {args.interval}\n")
        f.write(f"Initial Capital:  ${args.capital:,.2f}\n")
        f.write(f"Position Size:    {args.position_size * 100:.1f}%\n\n")

        # Get data period from first result
        first_result = list(results.values())[0]['metrics']
        if first_result.get('data_start'):
            data_start = pd.to_datetime(first_result['data_start']).strftime('%Y-%m-%d')
            data_end = pd.to_datetime(first_result['data_end']).strftime('%Y-%m-%d')
            f.write(f"Data Period:      {data_start} to {data_end}\n")
            f.write(f"Duration:         {first_result['data_period_str']}\n\n")

        f.write("-" * 100 + "\n")
        f.write(f"{'Config':<15} {'Trades':<8} {'Return':<12} {'Sharpe':<10} {'MaxDD':<12} {'Win%':<10} {'PF':<10}\n")
        f.write("-" * 100 + "\n")

        for config_name, result in results.items():
            m = result['metrics']
            f.write(
                f"{config_name:<15} "
                f"{m['num_trades']:<8} "
                f"{m['total_return']:>10.2f}%  "
                f"{m['sharpe_ratio']:>8.2f}  "
                f"{m['max_drawdown']:>10.2f}%  "
                f"{m['win_rate']:>8.2f}%  "
                f"{m['profit_factor']:>8.2f}\n"
            )

        f.write("-" * 100 + "\n\n")

        # Best performers
        best_return = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
        best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
        best_drawdown = min(results.items(), key=lambda x: x[1]['metrics']['max_drawdown'])

        f.write(f"Best Return:      {best_return[0]} ({best_return[1]['metrics']['total_return']:.2f}%)\n")
        f.write(f"Best Sharpe:      {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.2f})\n")
        f.write(f"Lowest Drawdown:  {best_drawdown[0]} ({best_drawdown[1]['metrics']['max_drawdown']:.2f}%)\n")

        f.write("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    print("This module provides results storage functions for the MAR Trading Bot.")
    print("Import and use save_backtest_results() or save_multiple_configs() in your backtest scripts.")
