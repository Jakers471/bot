"""
Backtesting module for trading strategies based on ML predictions.

This module simulates trading based on model signals:
- prediction=1 (trending_up) -> go LONG
- prediction=2 (trending_down) -> go SHORT
- prediction=0 (ranging) -> close position (go flat)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import INITIAL_CAPITAL, POSITION_SIZE, COMMISSION


@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: List[Dict]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    initial_capital: float
    final_capital: float

    def __repr__(self):
        return (
            f"BacktestResult(\n"
            f"  Total Trades: {self.metrics['num_trades']}\n"
            f"  Total Return: {self.metrics['total_return']:.2f}%\n"
            f"  Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
            f"  Max Drawdown: {self.metrics['max_drawdown']:.2f}%\n"
            f"  Win Rate: {self.metrics['win_rate']:.2f}%\n"
            f")"
        )


def run_backtest(
    df_with_predictions: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    position_size: float = POSITION_SIZE,
    commission: float = COMMISSION
) -> BacktestResult:
    """
    Run backtest on DataFrame with predictions.

    Args:
        df_with_predictions: DataFrame with columns [open_time, open, high, low, close, prediction]
        initial_capital: Starting capital
        position_size: Fraction of capital to use per trade (0.1 = 10%)
        commission: Commission rate per trade (0.001 = 0.1%)

    Returns:
        BacktestResult object with trades, equity curve, and metrics
    """
    # Validate input
    required_cols = ['open_time', 'open', 'high', 'low', 'close', 'prediction']
    missing_cols = [col for col in required_cols if col not in df_with_predictions.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df_with_predictions.copy()
    df = df.sort_values('open_time').reset_index(drop=True)

    # Initialize tracking variables
    trades = []
    equity = initial_capital
    equity_curve = []

    current_position = None  # None, 'long', or 'short'
    entry_price = None
    entry_time = None
    entry_idx = None

    # Simulate trading
    for idx, row in df.iterrows():
        timestamp = row['open_time']
        price = row['close']
        prediction = row['prediction']

        # Determine desired position based on prediction
        if prediction == 1:
            desired_position = 'long'
        elif prediction == 2:
            desired_position = 'short'
        else:  # prediction == 0 (ranging)
            desired_position = None

        # Handle position changes
        if current_position is None:
            # No position, enter if signal present
            if desired_position is not None:
                current_position = desired_position
                entry_price = price
                entry_time = timestamp
                entry_idx = idx

        elif current_position != desired_position:
            # Position exists but needs to change - close current position
            exit_price = price
            exit_time = timestamp

            # Calculate PnL
            if current_position == 'long':
                price_change = (exit_price - entry_price) / entry_price
            else:  # short
                price_change = (entry_price - exit_price) / entry_price

            # Account for commission (entry + exit)
            pnl_pct = price_change - (2 * commission)

            # Calculate dollar PnL
            trade_capital = equity * position_size
            pnl = trade_capital * pnl_pct
            equity += pnl

            # Record trade
            trade = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_idx': entry_idx,
                'exit_idx': idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': current_position,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,  # Convert to percentage
                'capital_before': equity - pnl,
                'capital_after': equity
            }
            trades.append(trade)

            # Enter new position if signal present
            if desired_position is not None:
                current_position = desired_position
                entry_price = price
                entry_time = timestamp
                entry_idx = idx
            else:
                current_position = None
                entry_price = None
                entry_time = None
                entry_idx = None

        # Record equity at this point
        equity_curve.append({
            'open_time': timestamp,
            'equity': equity
        })

    # Close any remaining position at the end
    if current_position is not None:
        exit_price = df.iloc[-1]['close']
        exit_time = df.iloc[-1]['open_time']

        if current_position == 'long':
            price_change = (exit_price - entry_price) / entry_price
        else:  # short
            price_change = (entry_price - exit_price) / entry_price

        pnl_pct = price_change - (2 * commission)
        trade_capital = equity * position_size
        pnl = trade_capital * pnl_pct
        equity += pnl

        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_idx': entry_idx,
            'exit_idx': len(df) - 1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': current_position,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'capital_before': equity - pnl,
            'capital_after': equity
        }
        trades.append(trade)

        # Update final equity in curve
        equity_curve[-1]['equity'] = equity

    # Create equity curve Series
    equity_df = pd.DataFrame(equity_curve)
    equity_series = pd.Series(
        equity_df['equity'].values,
        index=equity_df['open_time']
    )

    # Calculate metrics
    metrics = calculate_metrics(trades, equity_series, initial_capital)

    return BacktestResult(
        trades=trades,
        equity_curve=equity_series,
        metrics=metrics,
        initial_capital=initial_capital,
        final_capital=equity
    )


def calculate_metrics(
    trades: List[Dict],
    equity_curve: pd.Series,
    initial_capital: float
) -> Dict[str, float]:
    """
    Calculate performance metrics from trades and equity curve.

    Args:
        trades: List of trade dictionaries
        equity_curve: Series of portfolio value over time
        initial_capital: Starting capital

    Returns:
        Dictionary of performance metrics
    """
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
            'largest_loss': 0.0
        }

    # Basic trade statistics
    num_trades = len(trades)
    total_pnl = sum(t['pnl'] for t in trades)
    final_capital = equity_curve.iloc[-1]
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

    # Sharpe ratio (annualized, assuming daily returns)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    cumulative_max = equity_curve.expanding().max()
    drawdown = (equity_curve - cumulative_max) / cumulative_max * 100
    max_drawdown = abs(drawdown.min())

    return {
        'num_trades': num_trades,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss
    }


def plot_equity_curve(result: BacktestResult) -> go.Figure:
    """
    Plot equity curve from backtest result.

    Args:
        result: BacktestResult object

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Equity curve
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))

    # Add horizontal line for initial capital
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        annotation_position="right"
    )

    # Calculate drawdown
    cumulative_max = pd.Series(result.equity_curve).expanding().max()
    drawdown = (result.equity_curve - cumulative_max) / cumulative_max * 100

    # Add drawdown on secondary y-axis
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown %',
        line=dict(color='red', width=1),
        yaxis='y2',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))

    fig.update_layout(
        title=f'Equity Curve - Total Return: {result.metrics["total_return"]:.2f}%',
        xaxis_title='Time',
        yaxis_title='Portfolio Value ($)',
        yaxis2=dict(
            title='Drawdown (%)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=600
    )

    return fig


def plot_trades_on_chart(df: pd.DataFrame, trades: List[Dict]) -> go.Figure:
    """
    Plot candlestick chart with entry/exit markers.

    Args:
        df: DataFrame with OHLC data
        trades: List of trade dictionaries

    Returns:
        Plotly figure with candlestick chart and trade markers
    """
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['open_time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))

    # Plot trade entries and exits
    for trade in trades:
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']

        # Entry marker
        entry_color = 'green' if trade['side'] == 'long' else 'red'
        entry_symbol = 'triangle-up' if trade['side'] == 'long' else 'triangle-down'

        fig.add_trace(go.Scatter(
            x=[df.iloc[entry_idx]['open_time']],
            y=[trade['entry_price']],
            mode='markers',
            marker=dict(
                symbol=entry_symbol,
                size=12,
                color=entry_color,
                line=dict(width=2, color='white')
            ),
            name=f"{trade['side'].upper()} Entry",
            showlegend=False,
            hovertemplate=f"Entry: {trade['side'].upper()}<br>Price: {trade['entry_price']:.2f}<extra></extra>"
        ))

        # Exit marker
        exit_color = 'darkgreen' if trade['pnl'] > 0 else 'darkred'

        fig.add_trace(go.Scatter(
            x=[df.iloc[exit_idx]['open_time']],
            y=[trade['exit_price']],
            mode='markers',
            marker=dict(
                symbol='x',
                size=10,
                color=exit_color,
                line=dict(width=2)
            ),
            name=f"Exit (PnL: {trade['pnl']:.2f})",
            showlegend=False,
            hovertemplate=f"Exit<br>Price: {trade['exit_price']:.2f}<br>PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)<extra></extra>"
        ))

    fig.update_layout(
        title='Trade Entries and Exits on Price Chart',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='closest',
        template='plotly_white',
        height=600,
        xaxis_rangeslider_visible=False
    )

    return fig


def generate_report(result: BacktestResult) -> str:
    """
    Generate a text summary report of backtest results.

    Args:
        result: BacktestResult object

    Returns:
        Formatted string report
    """
    metrics = result.metrics

    report = f"""
{'='*60}
BACKTEST PERFORMANCE REPORT
{'='*60}

CAPITAL
--------
Initial Capital:        ${result.initial_capital:,.2f}
Final Capital:          ${result.final_capital:,.2f}
Net Profit/Loss:        ${result.final_capital - result.initial_capital:,.2f}

RETURNS
-------
Total Return:           {metrics['total_return']:.2f}%
Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}
Maximum Drawdown:       {metrics['max_drawdown']:.2f}%

TRADE STATISTICS
----------------
Total Trades:           {metrics['num_trades']}
Win Rate:               {metrics['win_rate']:.2f}%
Profit Factor:          {metrics['profit_factor']:.2f}

Average Win:            ${metrics['avg_win']:.2f}
Average Loss:           ${metrics['avg_loss']:.2f}
Largest Win:            ${metrics['largest_win']:.2f}
Largest Loss:           ${metrics['largest_loss']:.2f}

TRADE BREAKDOWN
---------------
Winning Trades:         {int(metrics['num_trades'] * metrics['win_rate'] / 100)}
Losing Trades:          {metrics['num_trades'] - int(metrics['num_trades'] * metrics['win_rate'] / 100)}

{'='*60}
"""

    return report


def main():
    """
    Demonstration of backtester with sample data and predictions.
    """
    print("Backtester Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_bars = 1000

    # Generate sample price data (random walk)
    initial_price = 50000
    returns = np.random.randn(n_bars) * 0.01  # 1% volatility
    prices = initial_price * np.exp(np.cumsum(returns))

    # Create OHLC data
    df = pd.DataFrame({
        'open_time': pd.date_range(start='2024-01-01', periods=n_bars, freq='1h'),
        'open': prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
        'close': prices
    })

    # Generate sample predictions
    # Strategy: simple momentum-based
    df['returns'] = df['close'].pct_change()
    df['momentum'] = df['returns'].rolling(10).mean()

    # Convert momentum to predictions
    df['prediction'] = 0  # Default to ranging
    df.loc[df['momentum'] > 0.001, 'prediction'] = 1  # Trending up
    df.loc[df['momentum'] < -0.001, 'prediction'] = 2  # Trending down

    # Fill NaN predictions with 0
    df['prediction'] = df['prediction'].fillna(0).astype(int)

    print(f"\nGenerated {len(df)} bars of sample data")
    print(f"Prediction distribution:")
    print(df['prediction'].value_counts().sort_index())

    # Run backtest
    print("\nRunning backtest...")
    result = run_backtest(
        df,
        initial_capital=10000,
        position_size=0.1,
        commission=0.001
    )

    # Print report
    print(generate_report(result))

    # Show first few trades
    if result.trades:
        print("\nFirst 5 Trades:")
        print("-" * 100)
        print(f"{'Entry Time':<20} {'Exit Time':<20} {'Side':<6} {'Entry':<10} {'Exit':<10} {'PnL':<10} {'PnL %':<10}")
        print("-" * 100)
        for trade in result.trades[:5]:
            print(
                f"{str(trade['entry_time']):<20} "
                f"{str(trade['exit_time']):<20} "
                f"{trade['side']:<6} "
                f"${trade['entry_price']:<9.2f} "
                f"${trade['exit_price']:<9.2f} "
                f"${trade['pnl']:<9.2f} "
                f"{trade['pnl_pct']:<9.2f}%"
            )

    # Generate plots
    print("\nGenerating plots...")

    # Equity curve
    equity_fig = plot_equity_curve(result)
    equity_fig.write_html('backtest_equity_curve.html')
    print("Saved equity curve to: backtest_equity_curve.html")

    # Trade chart (use subset for clarity)
    trades_fig = plot_trades_on_chart(df.iloc[:200],
                                      [t for t in result.trades if t['entry_idx'] < 200])
    trades_fig.write_html('backtest_trades_chart.html')
    print("Saved trades chart to: backtest_trades_chart.html")

    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
