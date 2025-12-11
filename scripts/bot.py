#!/usr/bin/env python3
"""
Live Trading Bot for MAR Strategy

This script runs a live trading bot that:
1. Connects to Binance via CCXT
2. Fetches latest candles
3. Calculates MAR features
4. Loads model and makes predictions
5. Executes trades (or logs for paper trading)

Usage:
    python scripts/bot.py --symbol BTCUSDT --interval 1h --paper
    python scripts/bot.py --symbol BTCUSDT --interval 1h --live --api-key YOUR_KEY --api-secret YOUR_SECRET

IMPORTANT: Always start with --paper mode to verify the bot works correctly!
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import joblib
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SYMBOL, INTERVALS, MODELS_DIR, MA_START, MA_COUNT, MA_STEP
from src.mar_calculator import calculate_all_features
from src.trainer import prepare_features, predict, predict_proba


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MAR Trading Bot - Live Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading (recommended to start)
  python scripts/bot.py --symbol BTCUSDT --interval 1h --paper

  # Live trading with API keys from environment
  python scripts/bot.py --symbol BTCUSDT --interval 1h --live

  # Live trading with explicit API keys
  python scripts/bot.py --symbol BTCUSDT --interval 1h --live --api-key YOUR_KEY --api-secret YOUR_SECRET

  # Custom position size
  python scripts/bot.py --symbol BTCUSDT --interval 1h --paper --position-size 0.05

Environment Variables:
  BINANCE_API_KEY      - Binance API key
  BINANCE_API_SECRET   - Binance API secret
  BINANCE_TESTNET      - Set to 'true' to use testnet (paper trading)

WARNING: Live trading involves real money. Always test thoroughly in paper mode first!
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOL,
        help=f'Trading symbol (default: {SYMBOL})'
    )

    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=INTERVALS,
        help='Time interval (e.g., 1m, 5m, 15m, 1h, 4h, 1d)'
    )

    parser.add_argument(
        '--paper',
        action='store_true',
        help='Paper trading mode (no real trades, just logging)'
    )

    parser.add_argument(
        '--live',
        action='store_true',
        help='Live trading mode (REAL MONEY - use with caution!)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Binance API key (or set BINANCE_API_KEY env var)'
    )

    parser.add_argument(
        '--api-secret',
        type=str,
        help='Binance API secret (or set BINANCE_API_SECRET env var)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (auto-detected if not provided)'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=0.1,
        help='Position size as fraction of capital (default: 0.1 = 10%%)'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=60,
        help='Seconds between checks (default: 60)'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum prediction confidence (0-1) to take action (default: 0.5)'
    )

    return parser.parse_args()


def initialize_exchange(args):
    """Initialize CCXT exchange connection."""
    try:
        import ccxt
    except ImportError:
        print()
        print("ERROR: CCXT not installed")
        print("Install with: pip install ccxt")
        sys.exit(1)

    # Get API credentials
    api_key = args.api_key or os.getenv('BINANCE_API_KEY')
    api_secret = args.api_secret or os.getenv('BINANCE_API_SECRET')

    if args.live:
        if not api_key or not api_secret:
            print()
            print("ERROR: Live trading requires API credentials")
            print("Provide via --api-key and --api-secret or set environment variables:")
            print("  BINANCE_API_KEY")
            print("  BINANCE_API_SECRET")
            sys.exit(1)

        # Initialize live exchange
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures for short positions
            }
        })

        print("Initialized LIVE Binance connection")
        print("WARNING: This will execute REAL trades with REAL money!")

    else:
        # Paper trading - read-only connection
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        print("Initialized Binance connection (Paper Trading Mode)")

    return exchange


def fetch_latest_candles(exchange, symbol, interval, limit=200):
    """
    Fetch latest candles from exchange.

    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        interval: Timeframe (e.g., '1h')
        limit: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {limit} latest candles for {symbol} {interval}...")

    # Fetch candles
    candles = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

    # Convert to DataFrame
    df = pd.DataFrame(
        candles,
        columns=['open_time', 'open', 'high', 'low', 'close', 'volume']
    )

    # Convert timestamp to datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

    return df


def load_model(args):
    """Load trained model."""
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = MODELS_DIR / f"{args.symbol}_{args.interval}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    return model, model_path


def get_current_position(exchange, symbol, paper_mode=True, paper_position=None):
    """
    Get current position for the symbol.

    Returns:
        dict with 'side' (None, 'long', 'short') and 'size'
    """
    if paper_mode:
        return paper_position or {'side': None, 'size': 0}

    # For live trading, query actual position from exchange
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['contracts']) > 0:
                side = 'long' if pos['side'] == 'long' else 'short'
                return {'side': side, 'size': float(pos['contracts'])}
        return {'side': None, 'size': 0}
    except Exception as e:
        print(f"Warning: Could not fetch position: {e}")
        return {'side': None, 'size': 0}


def execute_trade(exchange, symbol, action, size, price, paper_mode=True):
    """
    Execute a trade.

    Args:
        exchange: CCXT exchange
        symbol: Trading pair
        action: 'buy', 'sell', 'close'
        size: Position size
        price: Current price
        paper_mode: If True, just log the trade

    Returns:
        Order result or None
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if paper_mode:
        print()
        print("=" * 70)
        print(f"[PAPER TRADE] {timestamp}")
        print(f"  Action: {action.upper()}")
        print(f"  Symbol: {symbol}")
        print(f"  Size:   {size}")
        print(f"  Price:  ${price:,.2f}")
        print("=" * 70)
        return None

    # Live trading execution
    try:
        print()
        print("=" * 70)
        print(f"[LIVE TRADE] {timestamp}")
        print(f"  Executing {action.upper()} order...")

        if action == 'buy':
            order = exchange.create_market_buy_order(symbol, size)
        elif action == 'sell':
            order = exchange.create_market_sell_order(symbol, size)
        elif action == 'close':
            # Close position
            order = exchange.create_market_order(symbol, 'sell', size)

        print(f"  Order executed: {order['id']}")
        print(f"  Status: {order['status']}")
        print("=" * 70)

        return order

    except Exception as e:
        print(f"ERROR executing trade: {e}")
        print("=" * 70)
        return None


def trading_loop(args, exchange, model):
    """Main trading loop."""
    print()
    print("=" * 70)
    print("STARTING TRADING LOOP")
    print("=" * 70)
    print(f"Symbol:           {args.symbol}")
    print(f"Interval:         {args.interval}")
    print(f"Mode:             {'PAPER TRADING' if args.paper else 'LIVE TRADING'}")
    print(f"Position Size:    {args.position_size * 100:.1f}%")
    print(f"Min Confidence:   {args.min_confidence * 100:.1f}%")
    print(f"Check Interval:   {args.check_interval}s")
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop")
    print()

    # Convert symbol format for CCXT (BTCUSDT -> BTC/USDT)
    ccxt_symbol = f"{args.symbol[:-4]}/{args.symbol[-4:]}"

    # Paper trading state
    paper_position = {'side': None, 'size': 0}

    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"[{timestamp}] Iteration {iteration}")
            print("-" * 70)

            try:
                # Fetch latest data
                df = fetch_latest_candles(
                    exchange,
                    ccxt_symbol,
                    args.interval,
                    limit=200  # Need enough for MA calculation
                )

                print(f"Fetched {len(df)} candles, latest: {df['open_time'].iloc[-1]}")

                # Calculate features
                print("Calculating MAR features...")
                df_features = calculate_all_features(df.copy())

                # Get latest row (current candle)
                latest = df_features.iloc[-1:].copy()

                # Check for NaN values
                if latest.isnull().any().any():
                    print("WARNING: Latest candle has NaN values (possibly insufficient data)")
                    print("Skipping this iteration...")
                    time.sleep(args.check_interval)
                    continue

                # Make prediction
                X, feature_names = prepare_features(latest)
                prediction = predict(model, X)[0]
                probabilities = predict_proba(model, X)[0]

                # Get confidence (max probability)
                confidence = probabilities.max()

                class_names = {0: 'Ranging', 1: 'Trending Up', 2: 'Trending Down'}
                predicted_class = class_names[prediction]

                print(f"Prediction: {predicted_class} (confidence: {confidence:.2%})")
                print(f"Probabilities: Ranging={probabilities[0]:.2%}, "
                      f"Trending Up={probabilities[1]:.2%}, "
                      f"Trending Down={probabilities[2]:.2%}")

                # Get current price
                current_price = float(latest['close'].iloc[0])
                print(f"Current Price: ${current_price:,.2f}")

                # Get current position
                current_position = get_current_position(
                    exchange,
                    ccxt_symbol,
                    paper_mode=args.paper,
                    paper_position=paper_position
                )

                print(f"Current Position: {current_position['side'] or 'None'}")

                # Trading logic
                action_taken = False

                # Only act if confidence exceeds threshold
                if confidence >= args.min_confidence:

                    # Determine desired position based on prediction
                    if prediction == 1:  # Trending Up
                        desired_position = 'long'
                    elif prediction == 2:  # Trending Down
                        desired_position = 'short'
                    else:  # Ranging
                        desired_position = None

                    # Execute trades based on position changes
                    if current_position['side'] != desired_position:

                        # Close current position if exists
                        if current_position['side'] is not None:
                            execute_trade(
                                exchange,
                                ccxt_symbol,
                                'close',
                                current_position['size'],
                                current_price,
                                paper_mode=args.paper
                            )
                            action_taken = True

                            if args.paper:
                                paper_position = {'side': None, 'size': 0}

                        # Open new position if signal present
                        if desired_position is not None:
                            # Calculate position size
                            # In paper mode, assume $10,000 capital
                            capital = 10000 if args.paper else None  # Would need to fetch actual balance

                            if capital:
                                position_value = capital * args.position_size
                                position_size = position_value / current_price

                                action = 'buy' if desired_position == 'long' else 'sell'

                                execute_trade(
                                    exchange,
                                    ccxt_symbol,
                                    action,
                                    position_size,
                                    current_price,
                                    paper_mode=args.paper
                                )
                                action_taken = True

                                if args.paper:
                                    paper_position = {
                                        'side': desired_position,
                                        'size': position_size
                                    }
                else:
                    print(f"Confidence {confidence:.2%} below threshold {args.min_confidence:.2%}, no action taken")

                if not action_taken:
                    print("No action taken")

                print("-" * 70)
                print()

            except Exception as e:
                print(f"ERROR in iteration: {e}")
                import traceback
                traceback.print_exc()
                print()

            # Wait before next iteration
            print(f"Waiting {args.check_interval}s until next check...")
            print()
            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("SHUTTING DOWN")
        print("=" * 70)

        # Close any open positions
        current_position = get_current_position(
            exchange,
            ccxt_symbol,
            paper_mode=args.paper,
            paper_position=paper_position
        )

        if current_position['side'] is not None:
            print()
            print("Closing open position...")
            latest_price = float(df['close'].iloc[-1])
            execute_trade(
                exchange,
                ccxt_symbol,
                'close',
                current_position['size'],
                latest_price,
                paper_mode=args.paper
            )

        print()
        print("Bot stopped successfully")
        print("=" * 70)


def main():
    """Main bot function."""
    args = parse_args()

    # Validate mode selection
    if not args.paper and not args.live:
        print()
        print("ERROR: Must specify either --paper or --live mode")
        print("For safety, paper mode is recommended for testing")
        print()
        print("Example: python scripts/bot.py --symbol BTCUSDT --interval 1h --paper")
        sys.exit(1)

    if args.paper and args.live:
        print()
        print("ERROR: Cannot specify both --paper and --live")
        print("Choose one mode")
        sys.exit(1)

    print()
    print("=" * 70)
    print("MAR TRADING BOT")
    print("=" * 70)

    if args.live:
        print()
        print("WARNING: LIVE TRADING MODE ENABLED")
        print("This bot will execute REAL trades with REAL money!")
        print()
        response = input("Type 'YES' to continue: ")
        if response != 'YES':
            print("Aborted")
            sys.exit(0)

    try:
        # Initialize exchange
        exchange = initialize_exchange(args)

        # Load model
        model, model_path = load_model(args)

        # Start trading loop
        trading_loop(args, exchange, model)

        return 0

    except FileNotFoundError as e:
        print()
        print("ERROR: Required files not found")
        print("=" * 70)
        print(str(e))
        print()
        print("Make sure you have trained a model:")
        print(f"  python scripts/train.py --interval {args.interval} --model xgboost")
        print("=" * 70)
        return 1

    except Exception as e:
        print()
        print("ERROR: Bot initialization failed")
        print("=" * 70)
        print(f"{type(e).__name__}: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
