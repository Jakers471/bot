# MAR Trading Bot - Quick Start Guide

## Overview

This trading bot uses Moving Average Ribbon (MAR) indicators with machine learning to predict market regimes and execute trades automatically.

## Prerequisites

1. **Python 3.8+** installed
2. **Required packages** installed:
   ```bash
   pip install pandas numpy joblib scikit-learn xgboost ccxt plotly
   ```

3. **Historical data** downloaded
4. **Binance API credentials** (for live trading)

## Quick Start (5 Steps)

### Step 1: Check Your Setup
```bash
python scripts/check_setup.py
```
This will show you what's missing and what needs to be done.

### Step 2: Download Data (if needed)
```bash
python binance_downloader.py
```

### Step 3: Calculate Features
```bash
python src/mar_calculator.py
```

### Step 4: Generate Labels
```bash
python src/labeler.py
```

### Step 5: Train a Model
```bash
python scripts/train.py --interval 1h --model xgboost
```

## Usage Examples

### Training
```bash
# Train XGBoost for 1h interval
python scripts/train.py --interval 1h --model xgboost

# Train RandomForest for 4h interval
python scripts/train.py --interval 4h --model randomforest
```

### Backtesting
```bash
# Basic backtest
python scripts/backtest.py --interval 1h

# With custom settings
python scripts/backtest.py --interval 1h --capital 50000 --position-size 0.2

# Save results
python scripts/backtest.py --interval 1h --output results.csv
```

### Paper Trading
```bash
# Start paper trading
python scripts/bot.py --symbol BTCUSDT --interval 1h --paper

# With custom confidence threshold
python scripts/bot.py --symbol BTCUSDT --interval 1h --paper --min-confidence 0.7
```

### Live Trading
```bash
# Set API credentials (do this once)
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

# Run live bot (REAL MONEY!)
python scripts/bot.py --symbol BTCUSDT --interval 1h --live
```

## File Structure

```
bot/
├── scripts/
│   ├── train.py           # Train models
│   ├── backtest.py        # Backtest strategies
│   ├── bot.py             # Live trading bot
│   ├── check_setup.py     # Setup checker
│   └── README.md          # Detailed documentation
├── src/
│   ├── config.py          # Configuration
│   ├── mar_calculator.py  # Feature calculation
│   ├── labeler.py         # Label generation
│   ├── trainer.py         # Model training
│   └── backtester.py      # Backtesting engine
├── data/                  # Raw OHLCV data
├── features/              # Calculated features
├── labels/                # Generated labels
├── models/                # Trained models
└── QUICKSTART.md          # This file
```

## Key Concepts

### Market Regimes (Predictions)
- **0 = Ranging**: Market is moving sideways → Close positions
- **1 = Trending Up**: Market is trending upward → Go LONG
- **2 = Trending Down**: Market is trending downward → Go SHORT

### MAR Indicators
The bot calculates 96 moving averages (SMA, EMA, WMA) and derives 200+ features including:
- Spread (distance between fast and slow MAs)
- Compression (how tight the MAs are)
- Slopes (momentum of each MA)
- Price position (where price is relative to MAs)

## Important Notes

### Safety First
- **Always start with paper trading**
- Never risk more than you can afford to lose
- Test thoroughly before going live
- Monitor the bot regularly

### Performance Tips
1. Use longer timeframes (1h, 4h) for more reliable signals
2. Start with small position sizes (5-10% of capital)
3. Set appropriate confidence thresholds (0.6-0.8)
4. Backtest across different market conditions
5. Retrain models periodically with fresh data

### Troubleshooting

**"Model not found"**
→ Run `python scripts/train.py --interval 1h --model xgboost`

**"Features file not found"**
→ Run `python src/mar_calculator.py`

**"No labeled data found"**
→ Run `python src/labeler.py`

**"CCXT import error"**
→ Run `pip install ccxt`

## Configuration

Edit `src/config.py` to customize:
- Trading symbols
- Timeframes
- MA parameters
- Position sizing
- Commission rates
- And more...

## Recommended Workflow

1. **Development Phase**
   - Download historical data
   - Calculate features
   - Generate labels
   - Train multiple models
   - Compare performance

2. **Testing Phase**
   - Backtest on historical data
   - Analyze metrics (Sharpe, drawdown, win rate)
   - Paper trade for 1-2 weeks
   - Monitor predictions vs. reality

3. **Production Phase**
   - Start with small capital
   - Use conservative position sizing
   - Monitor daily performance
   - Keep logs of all trades
   - Retrain models monthly

## Performance Metrics

Good model characteristics:
- **Accuracy**: > 50% (better than random)
- **Sharpe Ratio**: > 1.0 (good risk-adjusted returns)
- **Win Rate**: > 40% (with good profit factor)
- **Max Drawdown**: < 20% (controlled risk)

## Support

For detailed documentation:
- `scripts/README.md` - Script usage
- `src/` modules - Code documentation
- Configuration: `src/config.py`

## Disclaimer

- **Trading involves significant risk**
- **Past performance ≠ future results**
- **Use at your own risk**
- **No warranties provided**

---

**Ready to start?** Run `python scripts/check_setup.py` to see what you need!
