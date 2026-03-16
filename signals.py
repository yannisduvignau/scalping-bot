"""
=============================================================
  SCALPING BOT — SIGNAL ENGINE  (v2)
=============================================================
  Scoring grid (6 votes):
    1. EMA alignment        — trend direction         (weight 1)
    2. RSI direction        — momentum, directional   (weight 1)
    3. StochRSI extreme     — crossover FROM zone     (weight 1)
    4. MACD crossover       — confirmation, cross only(weight 1)
    5. Bollinger + EMA      — trend-aligned BB touch  (weight 1)
    6. Candle body          — bar direction + size     (weight 1)

  Mandatory gates (not scored):
    • Volume  ≥ VOLUME_MULTIPLIER × rolling mean
    • ADX     ≥ ADX_THRESHOLD  (trending market only)

  v1 bug fixes:
    • RSI: had overlapping long/short zones (45–55 = both sides)
    • StochRSI: fired whenever k>d anywhere, not just from extremes
    • MACD: histogram used as permanent fallback → always voted
    • Bollinger: fired counter-trend (buying in a downtrend)
=============================================================
"""

from __future__ import annotations
import logging
import pandas as pd
from dataclasses import dataclass
from config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD,
    STOCHRSI_OB, STOCHRSI_OS,
    VOLUME_MULTIPLIER, MIN_SIGNAL_SCORE,
    ADX_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: str      # "LONG" | "SHORT" | "FLAT"
    score: int
    details: dict
    close: float
    atr: float
    symbol: str = ""


# ─── VOTE 1: EMA alignment ──────────────────────────────────

def _ema_vote(row: pd.Series) -> int:
    """
    +1 if EMA9 > EMA21 AND price > EMA50  (bull stack)
    -1 if EMA9 < EMA21 AND price < EMA50  (bear stack)
    """
    if row["ema_bull"]:
        return 1
    if row["ema_bear"]:
        return -1
    return 0


# ─── VOTE 2: RSI direction ──────────────────────────────────

def _rsi_vote(row: pd.Series) -> int:
    """
    v2 FIX: use direction (rising/falling) not value zones.
    Old v1 had overlapping zones: 30<rsi<55 = LONG AND 45<rsi<70 = SHORT.

    Rules:
      • rsi < RSI_OVERSOLD   → strong LONG (extreme oversold)
      • rsi > RSI_OVERBOUGHT → strong SHORT (extreme overbought)
      • 35 < rsi < 65 and rising  → LONG  (bullish momentum)
      • 35 < rsi < 65 and falling → SHORT (bearish momentum)
      • Middle ground or flat      → 0
    """
    rsi     = row["rsi"]
    rising  = bool(row.get("rsi_rising", False))

    if rsi <= RSI_OVERSOLD:
        return 1
    if rsi >= RSI_OVERBOUGHT:
        return -1
    if 35 < rsi < 65:
        return 1 if rising else -1
    return 0


# ─── VOTE 3: StochRSI extreme crossover ─────────────────────

def _stochrsi_vote(row: pd.Series) -> int:
    """
    v2 FIX: only fire when crossing FROM an extreme zone.
    v1 fired whenever k>d anywhere in 20–80 range → constant noise.

    Rules:
      +1 if k crossed above d AND the cross happened from the oversold zone
         (prev_k < STOCHRSI_OS or prev_k < 40 to catch recent exits from OS)
      -1 if k crossed below d AND the cross happened from the overbought zone
         (prev_k > STOCHRSI_OB or prev_k > 60)
    """
    k, d         = row["stochrsi_k"],      row["stochrsi_d"]
    k_prev, d_prev = row.get("stochrsi_k_prev", k), row.get("stochrsi_d_prev", d)

    crossed_up   = (k > d) and (k_prev <= d_prev)
    crossed_down = (k < d) and (k_prev >= d_prev)

    if crossed_up and k_prev < 40:         # crossed up from OS territory
        return 1
    if crossed_down and k_prev > 60:       # crossed down from OB territory
        return -1

    # Also reward being in low zone while above D (continuation)
    if k > d and k < STOCHRSI_OS + 10:
        return 1
    if k < d and k > STOCHRSI_OB - 10:
        return -1

    return 0


# ─── VOTE 4: MACD crossover ─────────────────────────────────

def _macd_vote(row: pd.Series) -> int:
    """
    v2 FIX: removed histogram fallback — it fired on every bar.
    Only actual signal-line crossovers score a vote.

    +1 on bullish crossover (MACD crosses above signal line)
    -1 on bearish crossover
     0 otherwise  ← key change vs v1
    """
    if row["macd_cross_up"]:
        return 1
    if row["macd_cross_down"]:
        return -1
    return 0


# ─── VOTE 5: Bollinger trend-aligned ────────────────────────

def _bollinger_vote(row: pd.Series) -> int:
    """
    v2 FIX: Bollinger vote now REQUIRES EMA alignment.
    v1 would buy the lower band in a downtrend → "catching falling knives".

    Rules:
      +1 if price near lower BB AND in an EMA bull trend
            (bounce off support in an uptrend)
      -1 if price near upper BB AND in an EMA bear trend
            (rejection at resistance in a downtrend)
      Squeeze → 0 (wait for breakout regardless of trend)
    """
    if row.get("bb_squeeze", False):
        return 0
    pct = row["bb_pct"]
    if pct < 0.15 and row["ema_bull"]:
        return 1
    if pct > 0.85 and row["ema_bear"]:
        return -1
    return 0


# ─── VOTE 6: Candle body direction  (v2 — new) ──────────────

def _candle_vote(row: pd.Series) -> int:
    """
    Confirms that the current bar is actually moving in the signal direction.
    Filter out doji bars (body too small vs ATR).

    +1 bullish candle body, body ≥ 0.15×ATR
    -1 bearish candle body, body ≥ 0.15×ATR
     0 doji / indecision
    """
    body_pct = float(row.get("body_pct", 0.0))
    if body_pct < 0.15:                  # doji — no conviction
        return 0
    if row.get("bar_bull", False):
        return 1
    if row.get("bar_bear", False):
        return -1
    return 0


# ─── MANDATORY GATES ────────────────────────────────────────

def _volume_gate(row: pd.Series) -> bool:
    """Volume must be above rolling mean × multiplier."""
    return float(row.get("vol_ratio", 0.0)) >= VOLUME_MULTIPLIER


def _adx_gate(row: pd.Series) -> bool:
    """
    v2 NEW: only trade when market is trending (ADX ≥ threshold).
    Flat, ranging markets produce whipsaws on momentum signals.
    """
    return float(row.get("adx", 0.0)) >= ADX_THRESHOLD


# ─── MASTER SIGNAL ──────────────────────────────────────────

def generate_signal(row: pd.Series, symbol: str = "") -> Signal:
    """Produce a Signal from the latest enriched OHLCV row."""

    # ── Mandatory gates ──────────────────────────────────────
    if not _volume_gate(row):
        return Signal("FLAT", 0, {"gate": "volume_low"}, row["close"], row["atr"], symbol)
    if not _adx_gate(row):
        return Signal("FLAT", 0, {"gate": "adx_low"}, row["close"], row["atr"], symbol)

    # ── Individual votes ─────────────────────────────────────
    votes = {
        "ema":        _ema_vote(row),
        "rsi":        _rsi_vote(row),
        "stoch_rsi":  _stochrsi_vote(row),
        "macd":       _macd_vote(row),
        "bollinger":  _bollinger_vote(row),
        "candle":     _candle_vote(row),
    }

    long_score  = sum(v for v in votes.values() if v > 0)
    short_score = abs(sum(v for v in votes.values() if v < 0))

    if long_score >= MIN_SIGNAL_SCORE and long_score > short_score:
        direction = "LONG"
        score     = long_score
    elif short_score >= MIN_SIGNAL_SCORE and short_score > long_score:
        direction = "SHORT"
        score     = short_score
    else:
        direction = "FLAT"
        score     = max(long_score, short_score)

    logger.debug("[%s] Signal=%s Score=%d | %s", symbol, direction, score, votes)

    return Signal(
        direction = direction,
        score     = score,
        details   = votes,
        close     = float(row["close"]),
        atr       = float(row["atr"]),
        symbol    = symbol,
    )


def generate_higher_tf_bias(df_htf: pd.DataFrame) -> str:
    """Quick bias from higher timeframe: LONG / SHORT / NEUTRAL."""
    if df_htf.empty:
        return "NEUTRAL"
    last = df_htf.iloc[-1]
    if last.get("ema_bull", False):
        return "LONG"
    if last.get("ema_bear", False):
        return "SHORT"
    return "NEUTRAL"
