"""
Trade Visualization Module for MAR Trading Bot

Generates publication-quality PNG charts for backtest results:
- Trade entries/exits on price charts with regime coloring
- Equity curves with drawdown analysis
- Multi-strategy comparisons with benchmarks

All charts use dark theme matching the notebook style.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter
from pathlib import Path
from typing import Dict, List, Optional, Union


# Dark theme color palette
COLORS = {
    'background': '#1a1a1a',
    'panel': '#1e1e1e',
    'grid': '#333333',
    'text': '#e0e0e0',
    'regime_up': '#004a00',      # Dark green
    'regime_range': '#4a4a00',   # Dark yellow
    'regime_down': '#4a0000',    # Dark red
    'candle_up': '#26a69a',      # Teal
    'candle_down': '#ef5350',    # Red
    'ma_positive': '#00ff00',    # Bright green
    'ma_negative': '#ff0000',    # Bright red
    'entry_long': '#00ff00',     # Green
    'entry_short': '#ff0000',    # Red
    'exit': '#ffff00',           # Yellow
    'equity': '#007acc',         # Blue
    'drawdown': '#ff4444',       # Bright red
    'benchmark': '#888888',      # Gray
}


def plot_trades(
    df: pd.DataFrame,
    trades: List[Dict],
    config_name: str,
    output_path: Union[str, Path],
    show_mas: bool = True,
    regime_column: str = 'regime'
) -> None:
    """
    Plot candlestick chart with trade entries/exits and regime backgrounds.

    Args:
        df: DataFrame with OHLCV data and regime column
        trades: List of trade dictionaries with entry/exit info
        config_name: Name of config for title
        output_path: Path to save PNG file
        show_mas: Whether to show moving average lines
        regime_column: Name of regime column in df (default: 'regime')
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['panel'])

    # Ensure datetime index
    if 'open_time' in df.columns:
        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)

    times = np.arange(len(df))

    # Draw regime backgrounds
    if regime_column in df.columns:
        regime_colors = {
            0: COLORS['regime_range'],
            1: COLORS['regime_up'],
            2: COLORS['regime_down']
        }

        current_regime = df.iloc[0][regime_column]
        block_start = 0

        for i in range(1, len(df)):
            regime = df.iloc[i][regime_column]
            if regime != current_regime or i == len(df) - 1:
                end_idx = i if regime != current_regime else i + 1
                ax.axvspan(
                    block_start, end_idx,
                    alpha=0.5,
                    color=regime_colors.get(current_regime, COLORS['regime_range']),
                    zorder=0
                )
                current_regime = regime
                block_start = i

    # Draw moving averages with color-coded slopes
    if show_mas:
        ma_columns = [col for col in df.columns if col.startswith('ma') and col.endswith('_sma')]

        for ma_col in ma_columns:
            if ma_col not in df.columns:
                continue

            values = df[ma_col].values

            # Calculate slopes
            slopes = np.zeros(len(values))
            slopes[1:] = values[1:] - values[:-1]
            slopes[0] = slopes[1] if len(slopes) > 1 else 0

            # Create line segments
            points = np.array([times, values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color by slope
            colors = np.where(slopes[1:] >= 0, COLORS['ma_positive'], COLORS['ma_negative'])

            lc = LineCollection(
                segments,
                colors=colors,
                linewidths=1.2,
                alpha=0.7,
                zorder=1
            )
            ax.add_collection(lc)

    # Draw candlesticks
    df_up = df[df['close'] >= df['open']]
    df_down = df[df['close'] < df['open']]

    # Bullish candles
    ax.bar(
        df_up.index, df_up['close'] - df_up['open'],
        width=0.6, bottom=df_up['open'],
        color=COLORS['candle_up'], zorder=2
    )
    ax.bar(
        df_up.index, df_up['high'] - df_up['close'],
        width=0.1, bottom=df_up['close'],
        color=COLORS['candle_up'], zorder=2
    )
    ax.bar(
        df_up.index, df_up['low'] - df_up['open'],
        width=0.1, bottom=df_up['open'],
        color=COLORS['candle_up'], zorder=2
    )

    # Bearish candles
    ax.bar(
        df_down.index, df_down['close'] - df_down['open'],
        width=0.6, bottom=df_down['open'],
        color=COLORS['candle_down'], zorder=2
    )
    ax.bar(
        df_down.index, df_down['high'] - df_down['open'],
        width=0.1, bottom=df_down['open'],
        color=COLORS['candle_down'], zorder=2
    )
    ax.bar(
        df_down.index, df_down['low'] - df_down['close'],
        width=0.1, bottom=df_down['close'],
        color=COLORS['candle_down'], zorder=2
    )

    # Mark trade entries and exits
    for trade in trades:
        entry_idx = trade.get('entry_idx')
        exit_idx = trade.get('exit_idx')
        side = trade.get('side')
        pnl = trade.get('pnl', 0)

        if entry_idx is not None and entry_idx < len(df):
            entry_price = trade.get('entry_price', df.iloc[entry_idx]['close'])
            marker = '^' if side == 'long' else 'v'
            color = COLORS['entry_long'] if side == 'long' else COLORS['entry_short']

            ax.scatter(
                entry_idx, entry_price,
                marker=marker, s=150, c=color,
                edgecolors='white', linewidths=1.5,
                zorder=5, label='Entry' if trade == trades[0] else None
            )

        if exit_idx is not None and exit_idx < len(df):
            exit_price = trade.get('exit_price', df.iloc[exit_idx]['close'])

            ax.scatter(
                exit_idx, exit_price,
                marker='s', s=100, c=COLORS['exit'],
                edgecolors='white', linewidths=1.5,
                zorder=5, label='Exit' if trade == trades[0] else None
            )

            # Add PnL annotation
            pnl_text = f"+${pnl:.0f}" if pnl > 0 else f"-${abs(pnl):.0f}"
            text_color = COLORS['entry_long'] if pnl > 0 else COLORS['entry_short']

            ax.annotate(
                pnl_text,
                xy=(exit_idx, exit_price),
                xytext=(0, 15 if pnl > 0 else -15),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                color=text_color,
                weight='bold',
                zorder=6
            )

    # Formatting
    ax.set_xlim(0, len(df))
    ax.set_ylim(df['low'].min() * 0.998, df['high'].max() * 1.002)

    # X-axis: Show dates
    if 'open_time' in df.columns:
        step = max(1, len(df) // 10)
        tick_positions = range(0, len(df), step)
        tick_labels = [
            pd.Timestamp(df['open_time'].iloc[i]).strftime('%m/%d/%y')
            for i in tick_positions
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', color=COLORS['text'])

    # Y-axis: Format as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.tick_params(colors=COLORS['text'])

    # Calculate total return for title
    if trades:
        initial_capital = trades[0].get('capital_before', 10000)
        final_capital = trades[-1].get('capital_after', initial_capital)
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
    else:
        total_return = 0

    # Date range for title
    if 'open_time' in df.columns:
        start_date = pd.Timestamp(df['open_time'].iloc[0]).strftime('%Y-%m-%d')
        end_date = pd.Timestamp(df['open_time'].iloc[-1]).strftime('%Y-%m-%d')
        date_range = f"{start_date} to {end_date}"
    else:
        date_range = f"{len(df)} bars"

    # Title
    ax.set_title(
        f'{config_name} | {date_range} | Return: {total_return:+.2f}% | Trades: {len(trades)}',
        color=COLORS['text'],
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'])

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS['regime_up'], alpha=0.6, label='Uptrend'),
        mpatches.Patch(color=COLORS['regime_range'], alpha=0.6, label='Ranging'),
        mpatches.Patch(color=COLORS['regime_down'], alpha=0.6, label='Downtrend'),
    ]

    if show_mas:
        legend_handles.extend([
            plt.Line2D([0], [0], color=COLORS['ma_positive'], lw=2, label='MA+'),
            plt.Line2D([0], [0], color=COLORS['ma_negative'], lw=2, label='MA-'),
        ])

    legend_handles.extend([
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['entry_long'],
                   markersize=10, label='Long Entry', linestyle='None'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['entry_short'],
                   markersize=10, label='Short Entry', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['exit'],
                   markersize=8, label='Exit', linestyle='None'),
    ])

    ax.legend(
        handles=legend_handles,
        loc='upper left',
        facecolor=COLORS['panel'],
        edgecolor=COLORS['grid'],
        framealpha=0.9,
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLORS['background'])
    plt.close(fig)

    print(f"Saved trade chart to: {output_path}")


def plot_equity_curve(
    equity_curve: List[Dict],
    config_name: str,
    output_path: Union[str, Path],
    initial_capital: float = 10000
) -> None:
    """
    Plot equity curve with drawdown analysis.

    Args:
        equity_curve: List of dicts with 'open_time' and 'equity' keys
        config_name: Name of config for title
        output_path: Path to save PNG file
        initial_capital: Starting capital for calculating returns
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9),
        facecolor=COLORS['background'],
        gridspec_kw={'height_ratios': [3, 1]}
    )

    for ax in [ax1, ax2]:
        ax.set_facecolor(COLORS['panel'])

    # Extract equity values
    equity_values = [initial_capital] + [e['equity'] for e in equity_curve]
    times = pd.to_datetime([e['open_time'] for e in equity_curve])

    # Calculate high watermark and drawdown
    equity_series = pd.Series(equity_values[1:], index=times)
    high_watermark = equity_series.expanding().max()
    drawdown = (equity_series - high_watermark) / high_watermark * 100

    # Find max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()

    # Plot equity curve
    ax1.plot(times, equity_series.values, color=COLORS['equity'], linewidth=2, label='Equity')
    ax1.plot(times, high_watermark.values, color=COLORS['text'],
             linewidth=1, linestyle='--', alpha=0.5, label='High Watermark')

    # Annotate max drawdown
    if max_dd_idx in equity_series.index:
        max_dd_equity = equity_series.loc[max_dd_idx]
        ax1.scatter(
            max_dd_idx, max_dd_equity,
            color=COLORS['drawdown'], s=100, zorder=5,
            edgecolors='white', linewidths=1.5
        )
        ax1.annotate(
            f'Max DD: {max_dd_value:.2f}%',
            xy=(max_dd_idx, max_dd_equity),
            xytext=(10, -30),
            textcoords='offset points',
            color=COLORS['drawdown'],
            fontweight='bold',
            fontsize=10,
            arrowprops=dict(
                arrowstyle='->',
                color=COLORS['drawdown'],
                lw=1.5
            )
        )

    # Format equity axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.tick_params(colors=COLORS['text'])
    ax1.grid(True, alpha=0.2, color=COLORS['grid'])
    ax1.legend(loc='upper left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'])

    # Calculate total return
    final_equity = equity_series.iloc[-1]
    total_return = ((final_equity - initial_capital) / initial_capital) * 100

    ax1.set_title(
        f'{config_name} Equity Curve | Return: {total_return:+.2f}% | Max DD: {abs(max_dd_value):.2f}%',
        color=COLORS['text'],
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Plot drawdown
    ax2.fill_between(times, 0, drawdown.values,
                     color=COLORS['drawdown'], alpha=0.3, label='Drawdown')
    ax2.plot(times, drawdown.values, color=COLORS['drawdown'], linewidth=1.5)
    ax2.axhline(y=0, color=COLORS['text'], linewidth=1, linestyle='-', alpha=0.3)

    # Format drawdown axis
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax2.tick_params(colors=COLORS['text'])
    ax2.grid(True, alpha=0.2, color=COLORS['grid'])
    ax2.set_xlabel('Date', color=COLORS['text'], fontsize=12)
    ax2.legend(loc='lower left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'])

    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLORS['background'])
    plt.close(fig)

    print(f"Saved equity curve to: {output_path}")


def plot_comparison(
    results_dict: Dict[str, Dict],
    output_path: Union[str, Path],
    buy_and_hold: Optional[Dict] = None
) -> None:
    """
    Compare equity curves of multiple configs on one chart.

    Args:
        results_dict: Dict mapping config_name to result dict (with 'equity_curve' and 'metrics')
        output_path: Path to save PNG file
        buy_and_hold: Optional dict with buy-and-hold benchmark data
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['panel'])

    # Color palette for different configs
    config_colors = [
        '#007acc', '#00ff00', '#ff00ff', '#ffaa00', '#00ffff',
        '#ff0088', '#88ff00', '#0088ff', '#ff8800', '#8800ff'
    ]

    legend_entries = []

    # Plot each config's equity curve
    for idx, (config_name, result) in enumerate(results_dict.items()):
        equity_curve = result.get('equity_curve', [])
        metrics = result.get('metrics', {})
        initial_capital = result.get('initial_capital', 10000)

        if not equity_curve:
            continue

        # Extract equity values
        equity_values = [initial_capital] + [e['equity'] for e in equity_curve]
        times = pd.to_datetime([e['open_time'] for e in equity_curve])

        # Get color
        color = config_colors[idx % len(config_colors)]

        # Calculate final return
        final_return = metrics.get('total_return', 0)

        # Plot line
        ax.plot(
            times,
            equity_values[1:],
            color=color,
            linewidth=2.5,
            label=f'{config_name}: {final_return:+.2f}%',
            alpha=0.9
        )

    # Plot buy-and-hold benchmark if provided
    if buy_and_hold and results_dict:
        # Use first config's equity curve for timeline
        first_result = list(results_dict.values())[0]
        equity_curve = first_result.get('equity_curve', [])
        initial_capital = first_result.get('initial_capital', 10000)

        if equity_curve:
            times = pd.to_datetime([e['open_time'] for e in equity_curve])

            # Calculate buy-and-hold equity over time (linear interpolation)
            bh_return = buy_and_hold.get('total_return', 0)
            bh_equity = initial_capital * (1 + bh_return / 100)

            # Linear equity growth from initial to final
            bh_equity_curve = np.linspace(initial_capital, bh_equity, len(times))

            ax.plot(
                times,
                bh_equity_curve,
                color=COLORS['benchmark'],
                linewidth=2,
                linestyle='--',
                label=f'Buy & Hold: {bh_return:+.2f}%',
                alpha=0.7
            )

    # Formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.tick_params(colors=COLORS['text'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'])

    # Title
    ax.set_title(
        'Strategy Comparison - Equity Curves',
        color=COLORS['text'],
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Labels
    ax.set_xlabel('Date', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Equity ($)', color=COLORS['text'], fontsize=12)

    # Legend
    ax.legend(
        loc='upper left',
        facecolor=COLORS['panel'],
        edgecolor=COLORS['grid'],
        framealpha=0.9,
        fontsize=10
    )

    # Format x-axis
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLORS['background'])
    plt.close(fig)

    print(f"Saved comparison chart to: {output_path}")


def plot_all_from_backtest(
    df: pd.DataFrame,
    results_dict: Dict[str, Dict],
    output_dir: Union[str, Path],
    buy_and_hold: Optional[Dict] = None,
    show_mas: bool = True
) -> None:
    """
    Generate all visualization charts from backtest results.

    Convenience function that creates:
    - Individual trade charts for each config
    - Individual equity curves for each config
    - Comparison chart of all configs

    Args:
        df: DataFrame with OHLCV and regime data
        results_dict: Dict mapping config_name to result dict
        output_dir: Directory to save all charts
        buy_and_hold: Optional buy-and-hold benchmark data
        show_mas: Whether to show moving averages on trade charts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualization charts...")
    print("=" * 80)

    # Generate individual charts for each config
    for config_name, result in results_dict.items():
        trades = result.get('trades', [])
        equity_curve = result.get('equity_curve', [])
        initial_capital = result.get('initial_capital', 10000)

        # Trade chart
        trade_chart_path = output_dir / f'{config_name}_trades.png'
        plot_trades(
            df=df,
            trades=trades,
            config_name=config_name,
            output_path=trade_chart_path,
            show_mas=show_mas
        )

        # Equity curve
        equity_chart_path = output_dir / f'{config_name}_equity.png'
        plot_equity_curve(
            equity_curve=equity_curve,
            config_name=config_name,
            output_path=equity_chart_path,
            initial_capital=initial_capital
        )

    # Generate comparison chart if multiple configs
    if len(results_dict) > 1:
        comparison_path = output_dir / 'comparison.png'
        plot_comparison(
            results_dict=results_dict,
            output_path=comparison_path,
            buy_and_hold=buy_and_hold
        )

    print("=" * 80)
    print(f"All charts saved to: {output_dir}")


if __name__ == '__main__':
    """
    Example usage for testing the visualizer module.
    """
    print("Trade Visualization Module")
    print("Import this module and use the following functions:")
    print("  - plot_trades(df, trades, config_name, output_path)")
    print("  - plot_equity_curve(equity_curve, config_name, output_path)")
    print("  - plot_comparison(results_dict, output_path, buy_and_hold)")
    print("  - plot_all_from_backtest(df, results_dict, output_dir, buy_and_hold)")
