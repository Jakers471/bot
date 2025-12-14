# CLAUDE.md - Project Guidelines for MAR Trading Bot

## REALISTIC BACKTESTING ONLY

This project uses **exclusively realistic backtesting** with no look-ahead bias. All casual/simplified backtests have been removed.

### What is Look-Ahead Bias?
Using future information to make decisions that would only be available in the past. This makes backtests show unrealistic profits.

### Rules That MUST Be Followed:

#### 1. Signal Generation
- Signals are generated using ONLY data available at that point in time
- A signal on bar N can only use data from bars 0 to N (inclusive)
- EMA/SMA calculations are causal (pandas ewm/rolling are OK)

#### 2. Trade Execution Timing
- **Signal detected**: End of bar N (when close price is known)
- **Trade executed**: Open of bar N+1 (next bar)
- NEVER enter at the same bar's close - that's look-ahead

#### 3. Regime Confirmation (Hysteresis/Min Bars)
- Regime changes must be confirmed CAUSALLY
- Cannot know regime duration until it's over
- Confirmation window looks BACKWARD, not forward
- A regime change is confirmed after N consecutive bars of new regime

#### 4. Price Data
- Entry price: Next bar's OPEN (after signal)
- Exit price: Next bar's OPEN (after exit signal)
- Never use close of signal bar for execution

#### 5. Feature Calculations
- All MAs, slopes, indicators use only past/current data
- Normalization must use rolling windows, not full dataset min/max
- Any percentage calculations must be point-in-time

### Slippage & Costs
- Commission: 0.1% per trade (entry + exit)
- Slippage: 0.05% per trade (conservative estimate)
- These compound - don't underestimate

### Benchmarks Required
Every backtest MUST include:
1. Buy-and-hold benchmark for comparison
2. Date range clearly stated
3. Number of bars/candles used

### Red Flags (Results Too Good)
If you see these, suspect look-ahead:
- Max drawdown < 10% over multi-year BTC data
- Sharpe ratio > 2.0 without HFT
- Win rate > 60% on trend-following
- Returns >> buy-and-hold with tiny drawdown

### Code Review Checklist
Before committing backtest changes:
- [ ] No future data in signal generation
- [ ] Trade entry on NEXT bar open
- [ ] Regime filters are causal (backward-looking only)
- [ ] Slippage included
- [ ] Buy-and-hold benchmark included
- [ ] Results sanity-checked against buy-and-hold

## Project Structure

```
bot/
├── src/                    # Core modules
│   ├── config.py          # Central configuration
│   ├── mar_calculator.py  # Feature engineering (MAs, indicators)
│   ├── regime_detector.py # Regime detection (MUST be causal)
│   ├── strategies/        # Trading strategies
│   └── backtester/        # Backtesting engine
├── scripts/
│   └── run_backtest.py    # Main backtest runner
├── configs/               # JSON configurations
├── results/               # Backtest outputs
├── notebooks/             # Jupyter analysis
└── data/                  # Price data (BTCUSDT)
```

## Running Backtests

```bash
# Run all 3 configs
python scripts/run_backtest.py --interval 1h

# Specific config
python scripts/run_backtest.py --interval 1h --config aggressive

# Show more trade details
python scripts/run_backtest.py --interval 1h --show-trades 10

# Custom costs
python scripts/run_backtest.py --interval 1h --commission 0.001 --slippage 0.0005
```

## Generate Charts

```bash
# Visualize latest backtest
python scripts/visualize_backtest.py --interval 1h

# Limit bars for faster rendering
python scripts/visualize_backtest.py --interval 1h --max-bars 500 --no-mas
```

## Results Storage

Each backtest creates a folder: `results/{timestamp}_{interval}_{config}/`
- `trades.csv` - All trades with costs
- `equity_curve.csv` - Portfolio value over time
- `metadata.txt` - Human-readable summary
- `summary.json` - Machine-readable metrics
- `config.json` - Regime detection params used

Charts saved to: `charts/{symbol}_{interval}_{timestamp}/`

## Regime Detection Configs

Three configs in `configs/regime_tuner_configs.json`:
- `default`: Balanced (Bull Up: 0.75, Min Bars: 15)
- `conservative`: Fewer signals, higher confidence (Bull Up: 0.80, Min Bars: 20)
- `aggressive`: More signals, faster reaction (Bull Up: 0.60, Min Bars: 5)

Regimes:
- 0 = Ranging (no position)
- 1 = Uptrend (long)
- 2 = Downtrend (short)

## Costs Applied

Every trade includes:
- Commission: 0.1% entry + 0.1% exit (configurable)
- Slippage: 0.05% entry + 0.05% exit (configurable)
- Execution: Next bar's OPEN price (not signal bar's close)
