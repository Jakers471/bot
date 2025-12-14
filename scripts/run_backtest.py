#!/usr/bin/env python3
"""
Run Backtest Script for MAR Trading Bot

This script runs backtests on BTCUSDT data using different regime detection configs
(default, conservative, aggressive) and compares their performance.

Usage:
    python scripts/run_backtest.py --interval 1h
    python scripts/run_backtest.py --interval 1h --start-date 2024-01-01 --end-date 2024-06-01
    python scripts/run_backtest.py --interval 4h --capital 50000
    python scripts/run_backtest.py --interval 1h --show-trades 10
    python scripts/run_backtest.py --interval 1h --no-save
    python scripts/run_backtest.py --interval 1h --output-dir custom_results

Results are saved to results/ directory as JSON files.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SYMBOL, INTERVALS, FEATURES_DIR
from src.results_manager import save_backtest_results, save_multiple_configs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtests with different regime detection configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on 1h data with all configs
  python scripts/run_backtest.py --interval 1h

  # Run backtest with date range
  python scripts/run_backtest.py --interval 1h --start-date 2024-01-01 --end-date 2024-06-01

  # Custom capital and position size
  python scripts/run_backtest.py --interval 1h --capital 50000 --position-size 0.2

  # Run on specific config only
  python scripts/run_backtest.py --interval 1h --config aggressive
        """
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        choices=INTERVALS,
        help='Time interval (default: 1h)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Trading symbol (default: {SYMBOL})'
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

    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital in USD (default: 10000)'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=None,
        help='Position size as fraction of capital (default: auto-calculated from risk)'
    )

    parser.add_argument(
        '--risk-per-trade',
        type=float,
        default=0.02,
        help='Max risk per trade as fraction of account (default: 0.02 = 2%%)'
    )

    parser.add_argument(
        '--stop-loss-pct',
        type=float,
        default=0.03,
        help='Stop loss as fraction of entry price (default: 0.03 = 3%%)'
    )

    parser.add_argument(
        '--risk-reward',
        type=float,
        default=3.0,
        help='Risk/reward ratio for take profit (default: 3.0 = 1:3)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate per trade (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--config',
        type=str,
        choices=['default', 'conservative', 'aggressive', 'all'],
        default='all',
        help='Which config to backtest (default: all)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0005,
        help='Slippage rate per trade (default: 0.0005 = 0.05%%)'
    )

    parser.add_argument(
        '--show-trades',
        type=int,
        default=5,
        help='Number of recent trades to display (default: 5, use 0 to hide)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save results to disk'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results/)'
    )

    return parser.parse_args()


def calculate_buy_and_hold(df: pd.DataFrame, initial_capital: float, position_size: float,
                           commission: float, slippage: float,
                           risk_per_trade: float = 0.02, stop_loss_pct: float = 0.03) -> dict:
    """
    Calculate buy-and-hold benchmark for comparison.

    Args:
        df: DataFrame with OHLCV data
        initial_capital: Starting capital
        position_size: Fraction of capital to invest (None = auto-calculate)
        commission: Commission rate
        slippage: Slippage rate
        risk_per_trade: For auto-calculating position size
        stop_loss_pct: For auto-calculating position size

    Returns:
        Dictionary with buy-and-hold metrics
    """
    if len(df) < 2:
        return {'total_return': 0.0, 'max_drawdown': 0.0}

    # Auto-calculate position size if not provided
    if position_size is None:
        position_size = min(risk_per_trade / stop_loss_pct, 1.0)

    # Buy at first bar's open (with slippage and commission)
    entry_price = df.iloc[0]['open'] * (1 + slippage)
    exit_price = df.iloc[-1]['close'] * (1 - slippage)

    # Calculate return
    invested = initial_capital * position_size
    shares = invested / entry_price
    final_value = shares * exit_price
    costs = invested * (commission + slippage) + final_value * (commission + slippage)
    net_return = (final_value - invested - costs) / initial_capital * 100

    # Calculate max drawdown
    # Simulate equity curve
    equity_curve = []
    for i, row in df.iterrows():
        current_value = shares * row['close']
        total_equity = (initial_capital - invested) + current_value
        equity_curve.append(total_equity)

    equity_series = pd.Series(equity_curve)
    cumulative_max = equity_series.expanding().max()
    drawdown = (equity_series - cumulative_max) / cumulative_max * 100
    max_drawdown = abs(drawdown.min())

    return {
        'total_return': float(net_return),
        'max_drawdown': float(max_drawdown),
        'entry_price': float(entry_price),
        'exit_price': float(exit_price)
    }


def load_regime_configs():
    """Load regime detection configurations from JSON file."""
    config_path = PROJECT_ROOT / 'configs' / 'regime_tuner_configs.json'

    if not config_path.exists():
        print(f"WARNING: Config file not found at {config_path}")
        print("Using default configurations...")
        return {
            "default": {
                "bullish_threshold_up": 0.75,
                "bullish_threshold_down": 0.35,
                "price_pos_up": 0.65,
                "price_pos_down": 0.35,
                "spread_min": 0.3,
                "min_bars": 15,
                "hysteresis": 2,
                "use_ema_smooth": True,
                "ema_span": 5
            },
            "conservative": {
                "bullish_threshold_up": 0.8,
                "bullish_threshold_down": 0.2,
                "price_pos_up": 0.75,
                "price_pos_down": 0.25,
                "spread_min": 0.5,
                "min_bars": 20,
                "hysteresis": 5,
                "use_ema_smooth": True,
                "ema_span": 10
            },
            "aggressive": {
                "bullish_threshold_up": 0.6,
                "bullish_threshold_down": 0.4,
                "price_pos_up": 0.55,
                "price_pos_down": 0.45,
                "spread_min": 0.2,
                "min_bars": 5,
                "hysteresis": 1,
                "use_ema_smooth": True,
                "ema_span": 3
            }
        }

    with open(config_path, 'r') as f:
        return json.load(f)


def detect_regimes(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Detect market regimes based on MAR indicators.

    IMPORTANT: This implementation is CAUSAL (no look-ahead bias).
    - Regime changes are only confirmed after min_bars consecutive bars
    - All filters look backward, never forward

    Args:
        df: DataFrame with features
        params: Regime detection parameters

    Returns:
        DataFrame with 'regime' column added (0=ranging, 1=trending_up, 2=trending_down)
    """
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

    # Get indicator values (with fallbacks)
    bull_pct = df['bullish_pct_sma'].values if 'bullish_pct_sma' in df.columns else np.full(len(df), 0.5)
    price_pos = df['price_position'].values if 'price_position' in df.columns else np.full(len(df), 0.5)
    spread = df['spread_pct'].values if 'spread_pct' in df.columns else np.full(len(df), 1.0)

    # Apply EMA smoothing if enabled (EMA is causal - only uses past data)
    if use_ema:
        bull_pct = pd.Series(bull_pct).ewm(span=ema_span, adjust=False).mean().values
        price_pos = pd.Series(price_pos).ewm(span=ema_span, adjust=False).mean().values

    # Detect raw regimes (point-in-time, no look-ahead)
    raw_regime = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if bull_pct[i] >= bull_up and price_pos[i] >= price_up and spread[i] >= spread_min:
            raw_regime[i] = 1  # Uptrend
        elif bull_pct[i] <= bull_dn and price_pos[i] <= price_dn and spread[i] >= spread_min:
            raw_regime[i] = 2  # Downtrend
        else:
            raw_regime[i] = 0  # Ranging

    # CAUSAL regime confirmation (no look-ahead)
    # A regime change is only confirmed after 'confirmation_bars' consecutive bars
    # This combines hysteresis and min_bars into a single causal filter
    confirmation_bars = max(hyst, min_bars)

    confirmed_regime = np.zeros(len(df), dtype=int)
    current_regime = 0  # Start with ranging
    pending_regime = None
    pending_count = 0

    for i in range(len(df)):
        raw = raw_regime[i]

        if raw == current_regime:
            # Same as current confirmed regime, stay in it
            pending_regime = None
            pending_count = 0
            confirmed_regime[i] = current_regime
        elif pending_regime is None:
            # New potential regime change, start counting
            pending_regime = raw
            pending_count = 1
            confirmed_regime[i] = current_regime  # Stay in current until confirmed
        elif raw == pending_regime:
            # Continues the pending regime
            pending_count += 1
            if pending_count >= confirmation_bars:
                # Confirmed! Switch to new regime
                current_regime = pending_regime
                confirmed_regime[i] = current_regime
                pending_regime = None
                pending_count = 0
            else:
                # Not yet confirmed, stay in current
                confirmed_regime[i] = current_regime
        else:
            # Different from both current and pending, reset pending
            pending_regime = raw
            pending_count = 1
            confirmed_regime[i] = current_regime

    df['regime'] = confirmed_regime
    df['raw_regime'] = raw_regime  # Keep raw for debugging
    return df


def simulate_regime_strategy(df: pd.DataFrame, initial_capital: float, position_size: float,
                             commission: float, slippage: float = 0.0005,
                             risk_per_trade: float = 0.02, stop_loss_pct: float = 0.03,
                             risk_reward: float = 3.0) -> dict:
    """
    Simulate trading based on regime predictions with proper risk management.

    IMPORTANT: NO LOOK-AHEAD BIAS
    - Signal generated at bar N's close
    - Trade executed at bar N+1's OPEN (realistic execution)
    - Slippage applied to entry and exit

    RISK MANAGEMENT:
    - Stop-loss: Exit if price moves stop_loss_pct against position
    - Take-profit: Exit if price moves (stop_loss_pct * risk_reward) in favor
    - Position sizing: risk_per_trade / stop_loss_pct (if position_size is None)
    - Max loss per trade: 2% of account (default)

    Regime 0 (ranging): Close position
    Regime 1 (trending_up): Go LONG
    Regime 2 (trending_down): Go SHORT

    Args:
        df: DataFrame with regime predictions
        initial_capital: Starting capital
        position_size: Fraction of capital per trade (None = auto-calculate from risk)
        commission: Commission rate (e.g., 0.001 = 0.1%)
        slippage: Slippage rate (e.g., 0.0005 = 0.05%)
        risk_per_trade: Max risk per trade as fraction of account (default 0.02 = 2%)
        stop_loss_pct: Stop loss distance as fraction of entry price (default 0.03 = 3%)
        risk_reward: Risk/reward ratio for take profit (default 3.0 = 1:3)

    Returns:
        Dictionary with trades, equity curve, and metrics
    """
    trades = []
    equity = initial_capital
    equity_curve = []
    total_commission_paid = 0.0
    total_slippage_cost = 0.0

    # Exit type counters
    exit_counts = {'stop_loss': 0, 'take_profit': 0, 'regime_change': 0, 'end_of_data': 0}

    # Calculate take profit distance
    take_profit_pct = stop_loss_pct * risk_reward

    current_position = None  # None, 'long', or 'short'
    entry_price = None
    entry_time = None
    entry_idx = None
    stop_loss_price = None
    take_profit_price = None
    trade_position_size = None  # Position size for current trade

    # We need to track the PREVIOUS bar's regime to know when signal was generated
    # Then execute on CURRENT bar's open
    pending_action = None  # ('enter', side) or ('exit',) or ('reverse', side)

    df_list = df.reset_index(drop=True)

    def calculate_position_size(current_equity):
        """Calculate position size based on risk parameters."""
        if position_size is not None:
            return position_size
        # Risk-based position sizing: risk_per_trade / stop_loss_pct
        # If risking 2% of account with 3% stop, position = 2%/3% = 66.7%
        calculated_size = risk_per_trade / stop_loss_pct
        # Cap at 100% of equity
        return min(calculated_size, 1.0)

    def close_position(row_idx, exit_price_raw, exit_reason):
        """Helper to close current position and record trade."""
        nonlocal current_position, entry_price, entry_time, entry_idx
        nonlocal stop_loss_price, take_profit_price, trade_position_size
        nonlocal equity, total_commission_paid, total_slippage_cost

        row = df_list.iloc[row_idx]
        timestamp = row['open_time']

        # Apply slippage to exit
        if current_position == 'long':
            exit_price = exit_price_raw * (1 - slippage)
            price_change = (exit_price - entry_price) / entry_price
        else:  # short
            exit_price = exit_price_raw * (1 + slippage)
            price_change = (entry_price - exit_price) / entry_price

        # Calculate PnL
        pnl_pct = price_change - (2 * commission) - (2 * slippage)
        trade_capital = equity * trade_position_size
        pnl = trade_capital * pnl_pct

        # Track costs
        trade_commission = trade_capital * commission * 2
        trade_slippage = trade_capital * slippage * 2
        total_commission_paid += trade_commission
        total_slippage_cost += trade_slippage

        equity += pnl

        # Calculate trade duration in bars
        duration_bars = row_idx - entry_idx

        trade = {
            'entry_time': str(entry_time),
            'exit_time': str(timestamp),
            'entry_idx': int(entry_idx),
            'exit_idx': int(row_idx),
            'duration_bars': int(duration_bars),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'stop_loss': float(stop_loss_price) if stop_loss_price else None,
            'take_profit': float(take_profit_price) if take_profit_price else None,
            'side': current_position,
            'exit_reason': exit_reason,
            'pnl': float(pnl),
            'pnl_pct': float(pnl_pct * 100),
            'position_size': float(trade_position_size),
            'commission_paid': float(trade_commission),
            'slippage_cost': float(trade_slippage),
            'capital_before': float(equity - pnl),
            'capital_after': float(equity)
        }
        trades.append(trade)
        exit_counts[exit_reason] += 1

        # Reset position state
        current_position = None
        entry_price = None
        entry_time = None
        entry_idx = None
        stop_loss_price = None
        take_profit_price = None
        trade_position_size = None

        return trade

    for i in range(len(df_list)):
        row = df_list.iloc[i]
        timestamp = row['open_time']
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        regime = row['regime']

        # CHECK STOP-LOSS / TAKE-PROFIT FIRST (before any new actions)
        # This uses the current bar's high/low to check if SL/TP was hit
        if current_position is not None:
            sl_hit = False
            tp_hit = False

            if current_position == 'long':
                # For long: SL hit if low <= stop_loss_price, TP hit if high >= take_profit_price
                if low_price <= stop_loss_price:
                    sl_hit = True
                elif high_price >= take_profit_price:
                    tp_hit = True
            else:  # short
                # For short: SL hit if high >= stop_loss_price, TP hit if low <= take_profit_price
                if high_price >= stop_loss_price:
                    sl_hit = True
                elif low_price <= take_profit_price:
                    tp_hit = True

            # Process SL/TP exits (SL takes priority if both hit in same bar)
            if sl_hit:
                close_position(i, stop_loss_price, 'stop_loss')
                pending_action = None  # Clear any pending action
            elif tp_hit:
                close_position(i, take_profit_price, 'take_profit')
                pending_action = None  # Clear any pending action

        # EXECUTE pending action from previous bar's signal (at current bar's OPEN)
        if pending_action is not None and current_position is None:
            action_type = pending_action[0]

            if action_type == 'enter':
                # Enter new position at this bar's open
                side = pending_action[1]
                trade_position_size = calculate_position_size(equity)

                if side == 'long':
                    entry_price = open_price * (1 + slippage)
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                else:  # short
                    entry_price = open_price * (1 - slippage)
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)

                entry_time = timestamp
                entry_idx = i
                current_position = side

            pending_action = None

        elif pending_action is not None and current_position is not None:
            action_type = pending_action[0]

            if action_type == 'exit':
                # Exit position at this bar's open (regime change)
                close_position(i, open_price, 'regime_change')

            elif action_type == 'reverse':
                # Exit current position first
                close_position(i, open_price, 'regime_change')

                # Enter new position
                new_side = pending_action[1]
                trade_position_size = calculate_position_size(equity)

                if new_side == 'long':
                    entry_price = open_price * (1 + slippage)
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                else:  # short
                    entry_price = open_price * (1 - slippage)
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)

                entry_time = timestamp
                entry_idx = i
                current_position = new_side

            pending_action = None

        # GENERATE signal for next bar (based on current bar's close/regime)
        # Only generate if not already in the desired position
        if regime == 1:
            desired_position = 'long'
        elif regime == 2:
            desired_position = 'short'
        else:
            desired_position = None

        # Check if position needs to change
        if current_position is None:
            if desired_position is not None:
                pending_action = ('enter', desired_position)
        elif current_position != desired_position:
            if desired_position is None:
                pending_action = ('exit',)
            else:
                pending_action = ('reverse', desired_position)

        # Record equity at this point
        equity_curve.append({
            'open_time': str(timestamp),
            'equity': float(equity)
        })

    # Close any remaining position at the end
    if current_position is not None:
        exit_price = df_list.iloc[-1]['close']

        if current_position == 'long':
            price_change = (exit_price - entry_price) / entry_price
        else:
            price_change = (entry_price - exit_price) / entry_price

        pnl_pct = price_change - (2 * commission) - (2 * slippage)
        trade_capital = equity * trade_position_size
        pnl = trade_capital * pnl_pct

        trade_commission = trade_capital * commission * 2
        trade_slippage = trade_capital * slippage * 2
        total_commission_paid += trade_commission
        total_slippage_cost += trade_slippage

        equity += pnl
        duration_bars = len(df_list) - 1 - entry_idx

        trade = {
            'entry_time': str(entry_time),
            'exit_time': str(df_list.iloc[-1]['open_time']),
            'entry_idx': int(entry_idx),
            'exit_idx': len(df_list) - 1,
            'duration_bars': int(duration_bars),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'stop_loss': float(stop_loss_price) if stop_loss_price else None,
            'take_profit': float(take_profit_price) if take_profit_price else None,
            'side': current_position,
            'exit_reason': 'end_of_data',
            'pnl': float(pnl),
            'pnl_pct': float(pnl_pct * 100),
            'position_size': float(trade_position_size),
            'commission_paid': float(trade_commission),
            'slippage_cost': float(trade_slippage),
            'capital_before': float(equity - pnl),
            'capital_after': float(equity)
        }
        trades.append(trade)
        exit_counts['end_of_data'] += 1

        if equity_curve:
            equity_curve[-1]['equity'] = float(equity)

    # Calculate metrics
    metrics = calculate_metrics(trades, equity_curve, initial_capital, df,
                                total_commission_paid, total_slippage_cost)

    # Add exit breakdown to metrics
    metrics['exit_counts'] = exit_counts

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'metrics': metrics,
        'initial_capital': float(initial_capital),
        'final_capital': float(equity),
        'total_commission_paid': float(total_commission_paid),
        'total_slippage_cost': float(total_slippage_cost),
        'risk_params': {
            'risk_per_trade': float(risk_per_trade),
            'stop_loss_pct': float(stop_loss_pct),
            'take_profit_pct': float(take_profit_pct),
            'risk_reward': float(risk_reward)
        }
    }


def calculate_metrics(trades: list, equity_curve: list, initial_capital: float, df: pd.DataFrame = None,
                     total_commission: float = 0.0, total_slippage: float = 0.0) -> dict:
    """Calculate performance metrics including time periods and costs."""

    # Time period calculations
    data_start = None
    data_end = None
    data_period_days = 0
    data_period_str = "N/A"
    trading_period_days = 0
    trading_period_str = "N/A"

    if df is not None and len(df) > 0:
        data_start = pd.to_datetime(df['open_time'].iloc[0])
        data_end = pd.to_datetime(df['open_time'].iloc[-1])
        data_period = data_end - data_start
        data_period_days = data_period.days

        # Format as human readable
        years = data_period_days // 365
        months = (data_period_days % 365) // 30
        days = data_period_days % 30
        parts = []
        if years > 0: parts.append(f"{years}y")
        if months > 0: parts.append(f"{months}m")
        if days > 0 or not parts: parts.append(f"{days}d")
        data_period_str = " ".join(parts)

    if len(trades) == 0:
        return {
            'num_trades': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_trade_duration_bars': 0.0,
            'total_commission_paid': float(total_commission),
            'total_slippage_cost': float(total_slippage),
            'net_profit': 0.0,
            'data_start': str(data_start) if data_start else None,
            'data_end': str(data_end) if data_end else None,
            'data_period_days': data_period_days,
            'data_period_str': data_period_str,
            'trading_period_days': 0,
            'trading_period_str': "N/A",
            'first_trade': None,
            'last_trade': None
        }

    # Basic trade statistics
    num_trades = len(trades)
    final_capital = equity_curve[-1]['equity']
    total_return = ((final_capital - initial_capital) / initial_capital) * 100

    # Win rate
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    win_rate = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average win/loss
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

    # Largest win/loss
    largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
    largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0

    # Sharpe ratio (annualized)
    equity_values = [initial_capital] + [e['equity'] for e in equity_curve]
    returns = pd.Series(equity_values).pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    equity_series = pd.Series([e['equity'] for e in equity_curve])
    cumulative_max = equity_series.expanding().max()
    drawdown = (equity_series - cumulative_max) / cumulative_max * 100
    max_drawdown = abs(drawdown.min())

    # Trading period (first trade to last trade)
    first_trade = pd.to_datetime(trades[0]['entry_time'])
    last_trade = pd.to_datetime(trades[-1]['exit_time'])
    trading_period = last_trade - first_trade
    trading_period_days = trading_period.days

    # Format trading period as human readable
    years = trading_period_days // 365
    months = (trading_period_days % 365) // 30
    days = trading_period_days % 30
    parts = []
    if years > 0: parts.append(f"{years}y")
    if months > 0: parts.append(f"{months}m")
    if days > 0 or not parts: parts.append(f"{days}d")
    trading_period_str = " ".join(parts)

    # Calculate average trade duration in bars
    durations = [t['duration_bars'] for t in trades if 'duration_bars' in t]
    avg_trade_duration_bars = np.mean(durations) if durations else 0.0

    # Calculate net profit (total PnL after all costs)
    net_profit = final_capital - initial_capital

    return {
        'num_trades': int(num_trades),
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.99,
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'largest_win': float(largest_win),
        'largest_loss': float(largest_loss),
        'avg_trade_duration_bars': float(avg_trade_duration_bars),
        'total_commission_paid': float(total_commission),
        'total_slippage_cost': float(total_slippage),
        'net_profit': float(net_profit),
        'data_start': str(data_start) if data_start else None,
        'data_end': str(data_end) if data_end else None,
        'data_period_days': int(data_period_days),
        'data_period_str': data_period_str,
        'trading_period_days': int(trading_period_days),
        'trading_period_str': trading_period_str,
        'first_trade': str(first_trade),
        'last_trade': str(last_trade)
    }


def print_box_header(title: str, width: int = 120):
    """Print a boxed header with ASCII characters for Windows compatibility."""
    line = "=" * width

    padding = (width - len(title)) // 2
    print(f"\n{line}")
    print(f"{' ' * padding}{title}")
    print(f"{line}")


def print_trade_summary(trades: list, show_n: int = 5):
    """Print a summary of recent trades, best trades, and worst trades."""
    if not trades or show_n <= 0:
        return

    print_box_header("TRADE DETAILS")

    # Exit reason abbreviations
    exit_abbrev = {
        'stop_loss': 'SL',
        'take_profit': 'TP',
        'regime_change': 'REG',
        'end_of_data': 'END'
    }

    # Recent trades
    print(f"\nLast {min(show_n, len(trades))} Trades:")
    print("-" * 130)
    print(f"{'Time':<20} {'Side':<6} {'Entry':<12} {'Exit':<12} {'SL':<12} {'TP':<12} {'Exit':<5} {'PnL $':<12} {'PnL %':<10}")
    print("-" * 130)

    for trade in trades[-show_n:]:
        entry_time = pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')
        side = trade['side'].upper()
        entry_price = f"${trade['entry_price']:,.2f}"
        exit_price = f"${trade['exit_price']:,.2f}"
        sl_price = f"${trade.get('stop_loss', 0):,.2f}" if trade.get('stop_loss') else "N/A"
        tp_price = f"${trade.get('take_profit', 0):,.2f}" if trade.get('take_profit') else "N/A"
        exit_reason = exit_abbrev.get(trade.get('exit_reason', ''), '?')
        pnl_dollars = f"${trade['pnl']:,.2f}"
        pnl_pct = f"{trade['pnl_pct']:+.2f}%"

        print(f"{entry_time:<20} {side:<6} {entry_price:<12} {exit_price:<12} {sl_price:<12} {tp_price:<12} {exit_reason:<5} {pnl_dollars:<12} {pnl_pct:<10}")

    # Best and worst trades
    if len(trades) > 1:
        print("\nBest & Worst Trades:")
        print("-" * 130)

        best_trade = max(trades, key=lambda t: t['pnl'])
        worst_trade = min(trades, key=lambda t: t['pnl'])

        best_exit = exit_abbrev.get(best_trade.get('exit_reason', ''), '?')
        worst_exit = exit_abbrev.get(worst_trade.get('exit_reason', ''), '?')

        print(f"Best:  {pd.to_datetime(best_trade['entry_time']).strftime('%Y-%m-%d')} | "
              f"{best_trade['side'].upper():<6} | ${best_trade['pnl']:>10,.2f} ({best_trade['pnl_pct']:+.2f}%) | "
              f"{best_trade['duration_bars']} bars | Exit: {best_exit}")
        print(f"Worst: {pd.to_datetime(worst_trade['entry_time']).strftime('%Y-%m-%d')} | "
              f"{worst_trade['side'].upper():<6} | ${worst_trade['pnl']:>10,.2f} ({worst_trade['pnl_pct']:+.2f}%) | "
              f"{worst_trade['duration_bars']} bars | Exit: {worst_exit}")

        # Average duration
        avg_duration = np.mean([t['duration_bars'] for t in trades])
        print(f"\nAverage Trade Duration: {avg_duration:.1f} bars")


def print_costs_breakdown(result: dict):
    """Print detailed breakdown of trading costs."""
    print_box_header("COSTS BREAKDOWN")

    total_comm = result.get('total_commission_paid', 0)
    total_slip = result.get('total_slippage_cost', 0)
    total_costs = total_comm + total_slip
    num_trades = result['metrics']['num_trades']

    print(f"\nTotal Commission Paid: ${total_comm:>12,.2f} (${total_comm/num_trades if num_trades > 0 else 0:>8,.2f} per trade)")
    print(f"Total Slippage Cost:   ${total_slip:>12,.2f} (${total_slip/num_trades if num_trades > 0 else 0:>8,.2f} per trade)")
    print("-" * 45)
    print(f"Total Trading Costs:   ${total_costs:>12,.2f}")
    print(f"\nNet Profit (after costs): ${result['metrics']['net_profit']:>12,.2f}")


def print_comparison_table(results: dict, bh_result: dict = None):
    """Print a formatted comparison table of all configs."""
    print_box_header("BACKTEST RESULTS COMPARISON (NO LOOK-AHEAD)")

    # Get time period info from first result
    first_result = list(results.values())[0]['metrics']
    data_start = first_result.get('data_start', 'N/A')
    data_end = first_result.get('data_end', 'N/A')
    data_period = first_result.get('data_period_str', 'N/A')

    # Format dates nicely
    if data_start and data_start != 'N/A':
        data_start = pd.to_datetime(data_start).strftime('%Y-%m-%d')
    if data_end and data_end != 'N/A':
        data_end = pd.to_datetime(data_end).strftime('%Y-%m-%d')

    print(f"\nData Period: {data_start} to {data_end} ({data_period})")

    # Show buy-and-hold benchmark
    if bh_result:
        print(f"Buy & Hold:  {bh_result['total_return']:.2f}% return, {bh_result['max_drawdown']:.2f}% max drawdown")

    # Header
    print(f"\n{'Config':<15} {'Trades':<8} {'Return':<12} {'vs B&H':<12} {'Sharpe':<10} {'MaxDD':<12} {'Win%':<10} {'PF':<10}")
    print("-" * 120)

    # Rows for each config
    for config_name, result in results.items():
        m = result['metrics']
        alpha = m['total_return'] - bh_result['total_return'] if bh_result else 0
        alpha_str = f"{alpha:+.2f}%" if bh_result else "N/A"
        print(
            f"{config_name:<15} "
            f"{m['num_trades']:<8} "
            f"{m['total_return']:>10.2f}%  "
            f"{alpha_str:>10}  "
            f"{m['sharpe_ratio']:>8.2f}  "
            f"{m['max_drawdown']:>10.2f}%  "
            f"{m['win_rate']:>8.2f}%  "
            f"{m['profit_factor']:>8.2f}"
        )

    print("-" * 120)

    # Find best performers
    best_return = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    best_drawdown = min(results.items(), key=lambda x: x[1]['metrics']['max_drawdown'])

    print(f"\nBest Return:      {best_return[0]} ({best_return[1]['metrics']['total_return']:.2f}%)")
    print(f"Best Sharpe:      {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.2f})")
    print(f"Lowest Drawdown:  {best_drawdown[0]} ({best_drawdown[1]['metrics']['max_drawdown']:.2f}%)")

    # Sanity check warning
    if bh_result:
        best_strategy_return = best_return[1]['metrics']['total_return']
        best_strategy_dd = best_return[1]['metrics']['max_drawdown']
        bh_return = bh_result['total_return']
        bh_dd = bh_result['max_drawdown']

        # Red flag checks
        if best_strategy_return > bh_return * 1.5 and best_strategy_dd < bh_dd * 0.1:
            print("\n*** WARNING: Results may be too good. Verify no look-ahead bias. ***")
        elif best_strategy_dd < 5 and bh_dd > 30:
            print("\n*** WARNING: Drawdown suspiciously low vs buy-and-hold. ***")


def print_summary_table(result: dict, config_name: str, bh_result: dict = None):
    """Print detailed summary table for a single config."""
    m = result['metrics']

    print_box_header(f"{config_name.upper()} - PERFORMANCE SUMMARY")

    print("\nPERFORMANCE METRICS:")
    print("-" * 60)
    print(f"{'Total Return:':<30} {m['total_return']:>10.2f}%")
    print(f"{'Net Profit:':<30} ${m['net_profit']:>10,.2f}")
    print(f"{'Sharpe Ratio:':<30} {m['sharpe_ratio']:>10.2f}")
    print(f"{'Max Drawdown:':<30} {m['max_drawdown']:>10.2f}%")
    print(f"{'Win Rate:':<30} {m['win_rate']:>10.2f}%")
    print(f"{'Profit Factor:':<30} {m['profit_factor']:>10.2f}")

    if bh_result:
        alpha = m['total_return'] - bh_result['total_return']
        print(f"{'Alpha vs Buy & Hold:':<30} {alpha:>10.2f}%")

    print("\nTRADE STATISTICS:")
    print("-" * 60)
    print(f"{'Total Trades:':<30} {m['num_trades']:>10}")
    print(f"{'Average Win:':<30} ${m['avg_win']:>10,.2f}")
    print(f"{'Average Loss:':<30} ${m['avg_loss']:>10,.2f}")
    print(f"{'Largest Win:':<30} ${m['largest_win']:>10,.2f}")
    print(f"{'Largest Loss:':<30} ${m['largest_loss']:>10,.2f}")
    print(f"{'Avg Trade Duration:':<30} {m['avg_trade_duration_bars']:>10.1f} bars")

    # Exit breakdown
    if 'exit_counts' in m:
        print("\nEXIT BREAKDOWN:")
        print("-" * 60)
        ec = m['exit_counts']
        total_exits = sum(ec.values())
        if total_exits > 0:
            print(f"{'Stop-Loss Exits:':<30} {ec.get('stop_loss', 0):>6} ({ec.get('stop_loss', 0)/total_exits*100:>5.1f}%)")
            print(f"{'Take-Profit Exits:':<30} {ec.get('take_profit', 0):>6} ({ec.get('take_profit', 0)/total_exits*100:>5.1f}%)")
            print(f"{'Regime Change Exits:':<30} {ec.get('regime_change', 0):>6} ({ec.get('regime_change', 0)/total_exits*100:>5.1f}%)")
            print(f"{'End of Data:':<30} {ec.get('end_of_data', 0):>6} ({ec.get('end_of_data', 0)/total_exits*100:>5.1f}%)")

    # Risk parameters used
    if 'risk_params' in result:
        rp = result['risk_params']
        print("\nRISK PARAMETERS:")
        print("-" * 60)
        print(f"{'Risk Per Trade:':<30} {rp['risk_per_trade']*100:>10.1f}%")
        print(f"{'Stop Loss:':<30} {rp['stop_loss_pct']*100:>10.1f}%")
        print(f"{'Take Profit:':<30} {rp['take_profit_pct']*100:>10.1f}%")
        print(f"{'Risk/Reward:':<30} {'1:' + str(int(rp['risk_reward'])):>10}")

    print("\nCOSTS & FEES:")
    print("-" * 60)
    total_comm = result.get('total_commission_paid', 0)
    total_slip = result.get('total_slippage_cost', 0)
    total_costs = total_comm + total_slip
    print(f"{'Total Commission:':<30} ${total_comm:>10,.2f}")
    print(f"{'Total Slippage:':<30} ${total_slip:>10,.2f}")
    print(f"{'Total Costs:':<30} ${total_costs:>10,.2f}")
    if m['num_trades'] > 0:
        print(f"{'Cost per Trade:':<30} ${total_costs/m['num_trades']:>10,.2f}")

    print("\nTIME PERIOD:")
    print("-" * 60)
    print(f"{'Data Period:':<30} {m['data_period_str']}")
    print(f"{'Trading Period:':<30} {m['trading_period_str']}")
    if m.get('first_trade'):
        print(f"{'First Trade:':<30} {pd.to_datetime(m['first_trade']).strftime('%Y-%m-%d %H:%M')}")
    if m.get('last_trade'):
        print(f"{'Last Trade:':<30} {pd.to_datetime(m['last_trade']).strftime('%Y-%m-%d %H:%M')}")


def save_results(results: dict, configs_to_run: dict, args) -> Path:
    """
    Save backtest results using the new results_manager module.

    This saves results in a structured folder format with CSV files,
    JSON configs, and human-readable metadata.

    Returns:
        Path to the output directory
    """
    print("\nUsing structured results storage system...")

    # Determine output directory
    if args.output_dir != 'results':
        # Custom output directory
        results_dir = Path(args.output_dir)
    else:
        results_dir = PROJECT_ROOT / args.output_dir

    # Save using the new results manager
    saved_folders = save_multiple_configs(
        results=results,
        all_configs=configs_to_run,
        args=args,
        project_root=PROJECT_ROOT
    )

    # Also save old-style JSON files for backward compatibility
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save individual config results (old format)
    for config_name, result in results.items():
        filename = f"backtest_{args.symbol}_{args.interval}_{config_name}_{timestamp}.json"
        filepath = results_dir / filename

        # Add metadata
        result['metadata'] = {
            'symbol': args.symbol,
            'interval': args.interval,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'initial_capital': args.capital,
            'position_size': args.position_size,
            'commission': args.commission,
            'slippage': args.slippage,
            'config_name': config_name,
            'timestamp': timestamp,
            'no_lookahead': True
        }

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

    # Save comparison summary (old format)
    summary_filename = f"backtest_summary_{args.symbol}_{args.interval}_{timestamp}.json"
    summary_filepath = results_dir / summary_filename

    summary = {
        'metadata': {
            'symbol': args.symbol,
            'interval': args.interval,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'initial_capital': args.capital,
            'position_size': args.position_size,
            'commission': args.commission,
            'slippage': args.slippage,
            'timestamp': timestamp,
            'no_lookahead': True
        },
        'configs': {
            config_name: result['metrics']
            for config_name, result in results.items()
        }
    }

    with open(summary_filepath, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBackward-compatible JSON files also saved to: {results_dir}")

    # Return the primary output folder from results manager
    if saved_folders:
        return saved_folders[0] if isinstance(saved_folders, list) else saved_folders
    return results_dir


def main():
    """Main backtest execution."""
    args = parse_args()

    print_box_header("MAR TRADING BOT - REGIME BACKTEST (NO LOOK-AHEAD)")
    print(f"\nSymbol:          {args.symbol}")
    print(f"Interval:        {args.interval}")
    print(f"Initial Capital: ${args.capital:,.2f}")

    # Risk Management Display
    take_profit_pct = args.stop_loss_pct * args.risk_reward
    if args.position_size is not None:
        print(f"Position Size:   {args.position_size * 100:.1f}% (manual)")
    else:
        auto_pos_size = args.risk_per_trade / args.stop_loss_pct
        print(f"Position Size:   {min(auto_pos_size, 1.0) * 100:.1f}% (auto: {args.risk_per_trade*100:.1f}% risk / {args.stop_loss_pct*100:.1f}% SL)")

    print(f"\n--- RISK MANAGEMENT ---")
    print(f"Risk Per Trade:  {args.risk_per_trade * 100:.1f}% of account (max ${args.capital * args.risk_per_trade:,.2f} loss)")
    print(f"Stop Loss:       {args.stop_loss_pct * 100:.1f}% from entry")
    print(f"Take Profit:     {take_profit_pct * 100:.1f}% from entry (1:{args.risk_reward:.0f} R:R)")
    print(f"-----------------------")

    print(f"\nCommission:      {args.commission * 100:.3f}% per trade")
    print(f"Slippage:        {args.slippage * 100:.3f}% per trade")
    if args.start_date:
        print(f"Start Date:      {args.start_date}")
    if args.end_date:
        print(f"End Date:        {args.end_date}")
    print(f"Show Trades:     {args.show_trades if args.show_trades > 0 else 'Hidden'}")

    try:
        # Load features data
        features_path = FEATURES_DIR / f"{args.symbol}_{args.interval}_features.parquet"
        if not features_path.exists():
            features_path = FEATURES_DIR / f"{args.symbol}_{args.interval}_features.csv"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_path}\n"
                f"Run: python src/mar_calculator.py"
            )

        print(f"\nLoading features from: {features_path}")
        if features_path.suffix == '.parquet':
            df = pd.read_parquet(features_path)
        else:
            df = pd.read_csv(features_path)

        # Ensure open_time is datetime
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)

        print(f"Loaded {len(df):,} rows")

        # Show data date range
        data_start = df['open_time'].min()
        data_end = df['open_time'].max()
        print(f"Data range: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")

        # Filter by date if specified
        if args.start_date or args.end_date:
            if args.start_date:
                df = df[df['open_time'] >= args.start_date]
                print(f"Filtered to start date: {args.start_date}")
            if args.end_date:
                df = df[df['open_time'] <= args.end_date]
                print(f"Filtered to end date: {args.end_date}")
            print(f"Date-filtered data: {len(df):,} rows")

        # Drop rows with NaN
        df_clean = df.dropna()
        print(f"After removing NaN: {len(df_clean):,} rows")

        if len(df_clean) == 0:
            raise ValueError("No valid data rows after cleaning")

        # Load regime configs
        print("\nLoading regime detection configs...")
        all_configs = load_regime_configs()

        # Determine which configs to run
        if args.config == 'all':
            configs_to_run = all_configs
        else:
            if args.config not in all_configs:
                raise ValueError(f"Config '{args.config}' not found. Available: {list(all_configs.keys())}")
            configs_to_run = {args.config: all_configs[args.config]}

        print(f"Running backtests for: {list(configs_to_run.keys())}")

        # Calculate buy-and-hold benchmark FIRST
        print_box_header("BUY-AND-HOLD BENCHMARK")
        print("\nCalculating benchmark...")
        bh_result = calculate_buy_and_hold(
            df_clean,
            initial_capital=args.capital,
            position_size=args.position_size,
            commission=args.commission,
            slippage=args.slippage,
            risk_per_trade=args.risk_per_trade,
            stop_loss_pct=args.stop_loss_pct
        )
        print(f"Entry Price:     ${bh_result['entry_price']:,.2f}")
        print(f"Exit Price:      ${bh_result['exit_price']:,.2f}")
        print(f"Total Return:    {bh_result['total_return']:.2f}%")
        print(f"Max Drawdown:    {bh_result['max_drawdown']:.2f}%")

        # Run backtests for each config
        results = {}
        for config_name, config_params in configs_to_run.items():
            print_box_header(f"BACKTEST: {config_name.upper()}")

            # Detect regimes
            print("\nDetecting regimes (CAUSAL - no look-ahead)...")
            df_with_regime = detect_regimes(df_clean, config_params)

            # Show regime distribution
            regime_counts = df_with_regime['regime'].value_counts().sort_index()
            print("\nRegime Distribution:")
            regime_names = {0: 'Ranging', 1: 'Trending Up', 2: 'Trending Down'}
            for regime_id, count in regime_counts.items():
                pct = (count / len(df_with_regime)) * 100
                print(f"  {regime_names.get(regime_id, f'Regime {regime_id}')}: {count:6,} ({pct:5.2f}%)")

            # Run backtest (with slippage, executes on NEXT bar's open)
            print("\nSimulating trades (entry on NEXT bar open, with SL/TP)...")
            result = simulate_regime_strategy(
                df_with_regime,
                initial_capital=args.capital,
                position_size=args.position_size,
                commission=args.commission,
                slippage=args.slippage,
                risk_per_trade=args.risk_per_trade,
                stop_loss_pct=args.stop_loss_pct,
                risk_reward=args.risk_reward
            )

            results[config_name] = result

            # Print detailed summary table for this config
            print_summary_table(result, config_name, bh_result)

            # Print trade details if requested
            if args.show_trades > 0 and result['trades']:
                print_trade_summary(result['trades'], args.show_trades)

            # Print costs breakdown
            print_costs_breakdown(result)

        # Print comparison table
        if len(results) > 1:
            print_comparison_table(results, bh_result)

        # Save results
        if not args.no_save:
            print_box_header("SAVING RESULTS")
            output_folder = save_results(results, configs_to_run, args)
            print(f"\nResults saved to: {output_folder}")
        else:
            print("\nSkipping save (--no-save flag set)")

        print_box_header("BACKTEST COMPLETE!")

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    except ValueError as e:
        print(f"\nERROR: {e}")
        return 1

    except Exception as e:
        print(f"\nERROR: Backtest failed")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
