# MAR Trading Bot Scripts

This directory contains production-ready scripts for training models, backtesting strategies, and running the live trading bot.

## Scripts Overview

### 1. `train.py` - Model Training
Trains machine learning models to predict market regimes (ranging, trending up, trending down).

**Usage:**
```bash
# Train XGBoost model for 1h interval
python scripts/train.py --interval 1h --model xgboost

# Train RandomForest for 4h interval
python scripts/train.py --interval 4h --model randomforest

# Train for different symbol
python scripts/train.py --interval 1h --model xgboost --symbol ETHUSDT
```

**Features:**
- Loads features and labels from `features/` and `labels/` directories
- Trains model with train/test split
- Displays accuracy, precision, recall, F1 score, and confusion matrix
- Shows top feature importances
- Saves trained model to `models/` directory

**Requirements:**
- Historical data downloaded (run `binance_downloader.py`)
- Features calculated (run `src/mar_calculator.py`)
- Labels generated (run `src/labeler.py`)

---

### 2. `backtest.py` - Strategy Backtesting
Backtests a trained model by generating predictions and simulating trades.

**Usage:**
```bash
# Backtest with auto-detected model
python scripts/backtest.py --interval 1h --symbol BTCUSDT

# Backtest with specific model file
python scripts/backtest.py --interval 1h --model models/BTCUSDT_1h_model.joblib

# Save results to CSV
python scripts/backtest.py --interval 1h --output results/backtest_results.csv

# Custom capital and position size
python scripts/backtest.py --interval 1h --capital 50000 --position-size 0.2

# Backtest specific date range
python scripts/backtest.py --interval 1h --start-date 2024-01-01 --end-date 2024-06-01
```

**Features:**
- Loads trained model and historical data
- Generates predictions for all candles
- Simulates trading with configurable capital and position size
- Calculates performance metrics (returns, Sharpe ratio, max drawdown, win rate)
- Shows recent trades
- Optionally saves trades and equity curve to CSV

**Trading Logic:**
- Prediction = 1 (Trending Up) → Go LONG
- Prediction = 2 (Trending Down) → Go SHORT
- Prediction = 0 (Ranging) → Close position (go flat)

---

### 3. `bot.py` - Live Trading Bot
Runs the trading bot in real-time, fetching live data and executing trades.

**Usage:**
```bash
# Paper trading (recommended for testing)
python scripts/bot.py --symbol BTCUSDT --interval 1h --paper

# Live trading with environment variables
python scripts/bot.py --symbol BTCUSDT --interval 1h --live

# Live trading with explicit API keys
python scripts/bot.py --symbol BTCUSDT --interval 1h --live \
    --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET

# Custom settings
python scripts/bot.py --symbol BTCUSDT --interval 1h --paper \
    --position-size 0.05 --min-confidence 0.7 --check-interval 120
```

**Features:**
- Connects to Binance via CCXT
- Fetches latest candles in real-time
- Calculates MAR features on-the-fly
- Makes predictions using trained model
- Executes trades based on signals
- Supports both paper trading (simulation) and live trading
- Configurable position sizing and confidence thresholds
- Graceful shutdown (Ctrl+C closes open positions)

**Important Notes:**
- **Always start with `--paper` mode** to verify the bot works correctly
- Live trading requires Binance API credentials
- Set environment variables: `BINANCE_API_KEY` and `BINANCE_API_SECRET`
- Bot uses Binance Futures for short positions
- Press Ctrl+C to stop the bot (it will close open positions)

**Environment Variables:**
```bash
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"
export BINANCE_TESTNET="true"  # Optional: use testnet
```

---

## Complete Workflow

### Step 1: Download Data
```bash
python binance_downloader.py
```

### Step 2: Calculate Features
```bash
python src/mar_calculator.py
```

### Step 3: Generate Labels
```bash
python src/labeler.py
```

### Step 4: Train Model
```bash
python scripts/train.py --interval 1h --model xgboost
```

### Step 5: Backtest Strategy
```bash
python scripts/backtest.py --interval 1h --output results/backtest.csv
```

### Step 6: Paper Trade
```bash
python scripts/bot.py --symbol BTCUSDT --interval 1h --paper
```

### Step 7: Live Trade (when ready!)
```bash
# Set API credentials first
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# Run live bot
python scripts/bot.py --symbol BTCUSDT --interval 1h --live
```

---

## Configuration

All configuration is managed in `src/config.py`:

- **Paths**: Data, features, models, labels directories
- **Symbols**: Trading pairs to trade
- **Intervals**: Timeframes to use
- **MAR Settings**: Moving average parameters
- **ML Settings**: Train/test split, random state
- **Backtest Settings**: Initial capital, position size, commission

---

## Error Handling

All scripts include comprehensive error handling:

- **FileNotFoundError**: Missing data, features, or model files
  - Solution: Run prerequisite steps (download data, calculate features, etc.)

- **ValueError**: Invalid data or configuration
  - Solution: Check data quality and configuration settings

- **API Errors**: Binance connection or authentication issues
  - Solution: Verify API credentials and internet connection

---

## Tips for Success

1. **Start with Paper Trading**: Always test in paper mode before risking real money
2. **Backtest Thoroughly**: Ensure strategy is profitable across different time periods
3. **Monitor Performance**: Track metrics like Sharpe ratio and max drawdown
4. **Risk Management**: Use appropriate position sizing (recommend 5-10% of capital)
5. **Stay Updated**: Keep data fresh, retrain models periodically
6. **Test on Testnet**: Use Binance testnet for realistic testing without risk

---

## Troubleshooting

### "Model not found"
Run training script first:
```bash
python scripts/train.py --interval 1h --model xgboost
```

### "Features file not found"
Calculate features first:
```bash
python src/mar_calculator.py
```

### "No labeled data found"
Generate labels first:
```bash
python src/labeler.py
```

### CCXT import error
Install CCXT:
```bash
pip install ccxt
```

### API authentication errors
- Verify API key and secret are correct
- Ensure API key has trading permissions enabled
- Check if IP is whitelisted (if restriction enabled)

---

## Safety Warnings

- **Live trading involves real financial risk**
- **Never risk more than you can afford to lose**
- **Past performance does not guarantee future results**
- **Always use stop losses and proper risk management**
- **Test thoroughly before deploying to production**
- **Monitor the bot regularly - don't leave it unattended**

---

## Support

For issues or questions:
1. Check the documentation in `src/` modules
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Test with minimal position sizes first

---

## License

Use at your own risk. No warranties provided.
