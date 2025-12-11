"""
Configuration for the MAR Trading Bot
"""
from pathlib import Path

# === PATHS ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
LABELS_DIR = BASE_DIR / "labels"

# === DATA ===
SYMBOL = "BTCUSDT"
INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

# === MAR INDICATOR ===
MA_TYPES = ["SMA", "EMA", "WMA"]
MA_START = 5        # First MA period
MA_COUNT = 32       # Number of MAs
MA_STEP = 1         # Increment between MAs
# Results in periods: 5, 6, 7, ... 36

# === DERIVED FEATURES ===
SLOPE_LOOKBACK = 5          # Bars to calculate MA slope
COMPRESSION_THRESHOLD = 0.5  # % range considered "compressed"

# === ML ===
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# === BACKTESTING ===
INITIAL_CAPITAL = 10000
POSITION_SIZE = 0.1  # 10% of capital per trade
COMMISSION = 0.001   # 0.1% per trade (Binance futures)

# === LABELS ===
LABEL_CLASSES = {
    0: "ranging",
    1: "trending_up",
    2: "trending_down",
}
