"""
=============================================================
  SCALPING BOT — INDICATORS  (v2)
=============================================================
  Additions vs v1:
    • ADX (14)        — trend-strength gate
    • Bar direction   — bullish/bearish candle flag
    • RSI direction   — rsi_rising (True/False)
    • StochRSI prev   — k_prev / d_prev for crossover detection
=============================================================
"""

import numpy as np
import pandas as pd
from config import (
    EMA_FAST, EMA_SLOW, EMA_TREND,
    RSI_PERIOD, STOCHRSI_PERIOD, STOCHRSI_K, STOCHRSI_D,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD,
    ATR_PERIOD, VOLUME_MA, ADX_PERIOD,
)


# ─── HELPERS ────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ─── TREND ──────────────────────────────────────────────────

def compute_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]  = _ema(df["close"], EMA_FAST)
    df["ema_slow"]  = _ema(df["close"], EMA_SLOW)
    df["ema_trend"] = _ema(df["close"], EMA_TREND)
    df["ema_bull"]  = (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_trend"])
    df["ema_bear"]  = (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["ema_trend"])
    return df


# ─── RSI ────────────────────────────────────────────────────

def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"]       = 100 - (100 / (1 + rs))
    # v2: direction flag used by the signal engine
    df["rsi_prev"]   = df["rsi"].shift(1)
    df["rsi_rising"] = df["rsi"] > df["rsi_prev"]
    return df


# ─── STOCHASTIC RSI ─────────────────────────────────────────

def compute_stoch_rsi(
    df: pd.DataFrame,
    period: int = STOCHRSI_PERIOD,
    k_period: int = STOCHRSI_K,
    d_period: int = STOCHRSI_D,
) -> pd.DataFrame:
    df = df.copy()
    if "rsi" not in df.columns:
        df = compute_rsi(df, period)
    rsi_min = df["rsi"].rolling(period).min()
    rsi_max = df["rsi"].rolling(period).max()
    stoch   = (df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    df["stochrsi_k"] = stoch.rolling(k_period).mean()
    df["stochrsi_d"] = df["stochrsi_k"].rolling(d_period).mean()
    # v2: prev-bar values for crossover-from-extreme detection
    df["stochrsi_k_prev"] = df["stochrsi_k"].shift(1)
    df["stochrsi_d_prev"] = df["stochrsi_d"].shift(1)
    return df


# ─── MACD ───────────────────────────────────────────────────

def compute_macd(
    df: pd.DataFrame,
    fast: int   = MACD_FAST,
    slow: int   = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    df = df.copy()
    ema_fast          = _ema(df["close"], fast)
    ema_slow          = _ema(df["close"], slow)
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], signal)
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_cross_up"]   = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    return df


# ─── BOLLINGER BANDS ────────────────────────────────────────

def compute_bollinger(
    df: pd.DataFrame,
    period: int = BB_PERIOD,
    std_dev: float = BB_STD,
) -> pd.DataFrame:
    df = df.copy()
    mid          = _sma(df["close"], period)
    std          = df["close"].rolling(period).std()
    df["bb_mid"]   = mid
    df["bb_upper"] = mid + std_dev * std
    df["bb_lower"] = mid - std_dev * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(20).min() * 1.1
    return df


# ─── ATR ────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    df = df.copy()
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(span=period, adjust=False).mean()
    return df


# ─── ADX  (v2 — new) ────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """
    Average Directional Index.
    adx > ADX_THRESHOLD  → trending market → allow entries
    plus_di > minus_di   → bullish trend
    minus_di > plus_di   → bearish trend
    """
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    up_move   = high.diff()
    down_move = (-low.diff())

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_s    = tr.ewm(span=period, adjust=False).mean()
    plus_di  = plus_dm.ewm(span=period, adjust=False).mean()  / atr_s * 100
    minus_di = minus_dm.ewm(span=period, adjust=False).mean() / atr_s * 100

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx    = (plus_di - minus_di).abs() / denom * 100
    df["adx"]      = dx.ewm(span=period, adjust=False).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    return df


# ─── VOLUME ─────────────────────────────────────────────────

def compute_volume(df: pd.DataFrame, ma_period: int = VOLUME_MA) -> pd.DataFrame:
    df = df.copy()
    df["vol_ma"]    = _sma(df["volume"], ma_period)
    df["vol_ratio"] = df["volume"] / df["vol_ma"]
    return df


# ─── CANDLE DIRECTION  (v2 — new) ───────────────────────────

def compute_candle_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    bar_bull: close > open (bullish candle)
    bar_bear: close < open (bearish candle)
    body_pct: relative body size vs ATR (filters doji bars)
    """
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    df["bar_bull"] = df["close"] > df["open"]
    df["bar_bear"] = df["close"] < df["open"]
    if "atr" in df.columns:
        df["body_pct"] = body / df["atr"].replace(0, np.nan)
    else:
        df["body_pct"] = 1.0          # neutral — ATR not computed yet
    return df


# ─── MASTER FUNCTION ────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator in one shot. Returns enriched DataFrame."""
    df = compute_emas(df)
    df = compute_rsi(df)
    df = compute_stoch_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger(df)
    df = compute_atr(df)
    df = compute_adx(df)
    df = compute_volume(df)
    df = compute_candle_direction(df)   # v2 — needs ATR already computed
    return df.dropna()
