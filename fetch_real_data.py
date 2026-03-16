"""
=============================================================
  SCALPING BOT — REAL DATA FETCHER
  Downloads BTC/USDT 5m OHLCV from Binance public REST API.
  No API key required.

  Usage:
    python fetch_real_data.py           # 90 days (default)
    python fetch_real_data.py --days 30 # custom window
=============================================================
"""
import time, json, os, argparse
import urllib.request
import pandas as pd
from datetime import datetime, timedelta, timezone

SYMBOL     = "BTCUSDT"
INTERVAL   = "5m"
BARS_LIMIT = 1000          # Binance max per request

def fetch_klines(symbol, interval, start_ms, end_ms):
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={symbol}&interval={interval}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={BARS_LIMIT}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def download_ohlcv(days: int = 90) -> pd.DataFrame:
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    all_rows = []
    cursor   = start_ms
    print(f"  Downloading {days}d of {SYMBOL} {INTERVAL} from Binance …")

    while cursor < end_ms:
        batch = fetch_klines(SYMBOL, INTERVAL, cursor, end_ms)
        if not batch:
            break
        all_rows.extend(batch)
        cursor = int(batch[-1][6]) + 1
        bars_per_day = 24 * 60 // 5
        days_done    = len(all_rows) / bars_per_day
        print(f"  {len(all_rows):>6} bars (~{days_done:.1f} days) …", end="\r")
        time.sleep(0.25)

    print(f"\n  ✓ {len(all_rows):,} bars downloaded")

    df = pd.DataFrame(all_rows, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","qav","num_trades","tbbav","tbqav","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")[["open","high","low","close","volume"]].astype(float)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days to download (default: 90)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    df = download_ohlcv(args.days)
    df.to_csv("logs/real_ohlcv.csv")
    print(f"  Saved → logs/real_ohlcv.csv")
    print(f"  Range: {df.index[0]}  →  {df.index[-1]}")
    print(f"  Price: ${df['close'].iloc[0]:,.0f}  →  ${df['close'].iloc[-1]:,.0f}")
    drift = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
    print(f"  Drift: {drift:+.1f}%")
