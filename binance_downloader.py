"""
Binance Historical Data Downloader
Downloads OHLCV klines from data.binance.vision (futures UM monthly)
"""

import requests
import zipfile
import pandas as pd
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIGURATION ===
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

INTERVALS = [
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
    "1d",
]

START_YEAR = 2020
START_MONTH = 1
END_YEAR = 2025
END_MONTH = 11

OUTPUT_DIR = Path(r"C:\Users\jakers\Desktop\bot\data")
MARKET = "futures/um"  # futures/um, futures/cm, or spot
MAX_WORKERS = 8  # Parallel downloads
SAVE_CSV = False  # Set True if you also want CSV (slower for large files)

# === COLUMN NAMES ===
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]

BASE_URL = "https://data.binance.vision/data"


def generate_months(start_year, start_month, end_year, end_month):
    """Generate list of (year, month) tuples"""
    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def download_month(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame | None:
    """Download a single month of kline data"""
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{MARKET}/monthly/klines/{symbol}/{interval}/{filename}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)

        return df

    except requests.RequestException:
        return None


def download_symbol_interval(symbol: str, interval: str, months: list) -> pd.DataFrame:
    """Download all months for a symbol/interval combo"""
    all_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_month, symbol, interval, year, month): (year, month)
            for year, month in months
        }

        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    # Combine and clean
    combined = pd.concat(all_data, ignore_index=True)
    combined["open_time"] = pd.to_numeric(combined["open_time"], errors="coerce")
    combined["close_time"] = pd.to_numeric(combined["close_time"], errors="coerce")
    combined = combined.dropna(subset=["open_time"])
    combined = combined.sort_values("open_time").drop_duplicates(subset=["open_time"])

    # Convert timestamps
    combined["open_time"] = pd.to_datetime(combined["open_time"], unit="ms")
    combined["close_time"] = pd.to_datetime(combined["close_time"], unit="ms")

    # Convert all numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_base", "taker_buy_quote", "trades", "ignore"]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


def save_data(df: pd.DataFrame, symbol: str, interval: str):
    """Save to Parquet (and optionally CSV)"""
    subdir = OUTPUT_DIR / symbol
    subdir.mkdir(parents=True, exist_ok=True)

    # Parquet (fast, compact)
    parquet_path = subdir / f"{symbol}_{interval}.parquet"
    df.to_parquet(parquet_path, index=False)
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {parquet_path.name} ({size_mb:.1f} MB)")

    # CSV (optional - slow for large files)
    if SAVE_CSV:
        csv_path = subdir / f"{symbol}_{interval}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path.name}")


def main():
    print("=" * 60)
    print("Binance Historical Data Downloader")
    print("=" * 60)
    print(f"Market: {MARKET}")
    print(f"Period: {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Intervals: {', '.join(INTERVALS)}")
    print("=" * 60)

    months = generate_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    total_jobs = len(SYMBOLS) * len(INTERVALS)
    completed = 0

    for symbol in SYMBOLS:
        print(f"\n[{symbol}]")

        for interval in INTERVALS:
            completed += 1
            print(f"  ({completed}/{total_jobs}) {interval}...", end=" ", flush=True)

            df = download_symbol_interval(symbol, interval, months)

            if df.empty:
                print("no data")
                continue

            print(f"{len(df):,} rows", end=" ")
            save_data(df, symbol, interval)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
