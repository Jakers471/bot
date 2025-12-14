# MAR Trading Bot

Bitcoin trading bot using Moving Average Ribbon (MAR) regime detection.

**All backtests are realistic** - no look-ahead bias, proper execution timing, full cost modeling.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest on all configs
python scripts/run_backtest.py --interval 1h

# Generate visualization charts
python scripts/visualize_backtest.py --interval 1h
```

## How It Works

1. **Regime Detection**: Analyzes 32 moving averages to detect market state
   - Uptrend (1): Go long
   - Ranging (0): Close position
   - Downtrend (2): Go short

2. **Realistic Execution**:
   - Signal generated at bar N's close
   - Trade executes at bar N+1's OPEN
   - Commission + slippage applied

3. **Results**: Saved to `results/` with trades, equity curve, and metadata

## Project Structure

```
bot/
├── scripts/
│   ├── run_backtest.py      # Main backtest runner
│   └── visualize_backtest.py # Generate charts
├── src/
│   ├── config.py            # Configuration
│   ├── mar_calculator.py    # Feature engineering
│   ├── regime_detector.py   # Regime detection
│   ├── results_manager.py   # Save results
│   └── visualizer.py        # Chart generation
├── configs/
│   └── regime_tuner_configs.json  # 3 regime configs
├── results/                 # Backtest outputs
├── charts/                  # Visualization PNGs
└── notebooks/
    └── 02_regime_tuner.ipynb  # Interactive tuning
```

## Regime Configs

| Config | Bull Up | Min Bars | Description |
|--------|---------|----------|-------------|
| default | 0.75 | 15 | Balanced |
| conservative | 0.80 | 20 | Fewer trades, higher confidence |
| aggressive | 0.60 | 5 | More trades, faster reaction |

## CLI Options

```bash
python scripts/run_backtest.py \
  --interval 1h \
  --config all \
  --capital 10000 \
  --position-size 0.1 \
  --commission 0.001 \
  --slippage 0.0005 \
  --show-trades 5
```

## Output

Each run saves to `results/{timestamp}_{interval}_{config}/`:
- `trades.csv` - All trades with entry/exit times, prices, PnL, costs
- `equity_curve.csv` - Portfolio value over time
- `metadata.txt` - Human-readable summary
- `summary.json` - Machine-readable metrics

Charts saved to `charts/`:
- `{config}_trades.png` - Price chart with trade markers
- `{config}_equity.png` - Equity curve with drawdown

## No Look-Ahead Guarantee

- Regime changes require N consecutive bars to confirm (no future knowledge)
- Trades execute on NEXT bar's open (can't trade on signal bar's close)
- All indicators use only past/current data
- See `CLAUDE.md` for detailed rules
