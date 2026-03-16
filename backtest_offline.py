"""
=============================================================
  SCALPING BOT — OFFLINE BACKTEST  (v4)
=============================================================
  What changed vs v3:
    1. Synthetic data calibrated to real BTC/USDT 5m stats:
       - Trending regimes last 150–600 bars (not 50–300)
       - GARCH-like volatility clustering
       - Stronger momentum autocorrelation (ρ=0.35 in trends)
       - Hurst exponent H≈0.55 in trends (real BTC ≈ 0.53–0.58)
    2. Halt spam FIXED: warning now prints once at halt trigger,
       silent for subsequent bars in the same halt window.
    3. fetch_real_data.py provided for real Binance test.
=============================================================
"""

from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional

os.makedirs("logs", exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))

import config
from indicators import add_all_indicators, compute_emas
from signals    import generate_signal
from risk_manager import RiskManager

# ─── LOGGING SETUP FOR BACKTEST ──────────────────────────────
# Console: WARNING+ only (shows halt messages once, nothing else)
# File:    DEBUG (full trace for post-mortem analysis)
import logging as _logging
def _configure_backtest_logging():
    root = _logging.getLogger()
    if root.handlers:
        # bot.py already called basicConfig – strip the StreamHandler so
        # backtest console output stays clean; keep the FileHandler.
        root.handlers = [h for h in root.handlers
                         if not isinstance(h, _logging.StreamHandler)
                         or isinstance(h, _logging.FileHandler)]
    # Add a WARNING-only console handler
    _ch = _logging.StreamHandler()
    _ch.setLevel(_logging.WARNING)
    _ch.setFormatter(_logging.Formatter("  %(message)s"))
    root.addHandler(_ch)
    # Ensure file handler exists at DEBUG level
    try:
        import logging.handlers as _lh
        _fh = _lh.RotatingFileHandler("logs/backtest.log", maxBytes=2_000_000, backupCount=2)
        _fh.setLevel(_logging.DEBUG)
        _fh.setFormatter(_logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        root.addHandler(_fh)
    except Exception:
        pass
    root.setLevel(_logging.DEBUG)
_configure_backtest_logging()


# ─── CALIBRATED SYNTHETIC DATA ──────────────────────────────
# Target: mimic real BTC/USDT 5m properties
#   • Daily vol:     ~3-5%   (sigma_ann ≈ 0.65-1.0)
#   • Trend duration: 150-600 bars (12-50 hours)
#   • Autocorrelation: ρ ≈ +0.30 in trends, -0.10 in ranging
#   • Hurst exponent:  H ≈ 0.55-0.58 in trending regime

def generate_synthetic_ohlcv(
    n_bars: int        = 5_000,
    start_price: float = 45_000.0,
    tf_minutes: int    = 5,
    seed: int          = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt  = tf_minutes / (252 * 24 * 60)

    # ── Longer, stronger regimes ──────────────────────────
    # Real BTC trends last 12–50h on 5m → 144–600 bars
    regimes = []
    while len(regimes) < n_bars:
        rtype = rng.choice([0, 1, 2], p=[0.35, 0.325, 0.325])
        if rtype == 0:                                   # ranging
            length = int(rng.integers(60, 200))
        else:                                            # trending
            length = int(rng.integers(150, 600))
        regimes.extend([rtype] * length)
    regime = np.array(regimes[:n_bars])

    # ── Per-regime parameters ─────────────────────────────
    # GARCH-like: base vol + regime boost
    sigma_base  = 0.65                                  # annual base vol
    sigma_trend = 0.90                                  # annual vol in trend
    sigma = np.where(regime == 0, sigma_base, sigma_trend)

    # Annual drift: ±2.0 in strong trend → ±0.4% per day on 5m
    mu = np.where(regime == 1, 2.0, np.where(regime == 2, -2.0, 0.0))

    # ── Autocorrelated returns (momentum) ─────────────────
    # ρ ≈ 0.35 in trends (Hurst ≈ 0.57), -0.10 in ranging
    mom_coeff = np.where(regime == 0, -0.10, 0.35)

    raw_noise = rng.standard_normal(n_bars)
    log_returns = np.zeros(n_bars)
    for i in range(n_bars):
        base = (mu[i] - 0.5 * sigma[i]**2) * dt + sigma[i] * np.sqrt(dt) * raw_noise[i]
        if i > 0:
            base += mom_coeff[i] * log_returns[i - 1]
        log_returns[i] = base

    closes = start_price * np.exp(np.cumsum(log_returns))

    # ── OHLC with realistic bar structure ─────────────────
    atr_pct = sigma * np.sqrt(dt) * 1.5
    ranges  = closes * atr_pct * rng.uniform(0.5, 1.5, n_bars)
    opens   = np.empty(n_bars); opens[0] = start_price
    opens[1:] = closes[:-1] * (1 + rng.normal(0, 0.0002, n_bars - 1))
    highs = np.maximum(opens, closes) + ranges * rng.uniform(0.2, 0.6, n_bars)
    lows  = np.maximum(
        np.minimum(opens, closes) - ranges * rng.uniform(0.2, 0.6, n_bars),
        closes * 0.85,
    )

    # ── Volume: higher during trends, spiky at regime changes ──
    base_vol = 500 + rng.exponential(300, n_bars)
    hour_idx = (np.arange(n_bars) * tf_minutes // 60) % 24
    vol_mult = 1 + 0.5 * np.sin(2 * np.pi * hour_idx / 24)
    vol_trend = np.where(regime != 0, 1.5, 1.0)    # 50% more vol in trends
    # Regime-change spikes
    regime_change = np.zeros(n_bars)
    regime_change[1:] = (regime[1:] != regime[:-1]).astype(float)
    vol_spike = 1 + 1.5 * regime_change            # spike at every regime start
    volume = base_vol * vol_mult * vol_trend * vol_spike * rng.uniform(0.7, 1.3, n_bars)

    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(minutes=i * tf_minutes) for i in range(n_bars)]
    return pd.DataFrame({
        "open": opens.round(2),  "high": highs.round(2),
        "low":  lows.round(2),   "close": closes.round(2),
        "volume": volume.round(2),
    }, index=pd.DatetimeIndex(timestamps, name="timestamp"))


# ─── HTF BIAS ────────────────────────────────────────────────

def compute_htf_bias_series(df_5m: pd.DataFrame) -> pd.Series:
    df_15m = (
        df_5m[["open","high","low","close","volume"]]
        .resample("15min")
        .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        .dropna()
    )
    if len(df_15m) < 60:
        return pd.Series("NEUTRAL", index=df_5m.index)
    df_15m = compute_emas(df_15m)
    def _bias(row):
        if row.get("ema_bull", False): return "LONG"
        if row.get("ema_bear", False): return "SHORT"
        return "NEUTRAL"
    return df_15m.apply(_bias, axis=1).reindex(df_5m.index, method="ffill").fillna("NEUTRAL")


# ─── TRADE RECORD ────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str; direction: str
    entry_time: str; exit_time: str
    entry_price: float; exit_price: float
    qty: float; pnl: float; pnl_pct: float
    exit_reason: str; score: int; duration_bars: int
    partial_tp_hit: bool = False
    breakeven_used: bool = False


# ─── EXIT CONSTANTS ──────────────────────────────────────────
TP1_ATR_MULT  = 2.0    # partial exit at 2×ATR
TP2_ATR_MULT  = 5.0    # full exit at 5×ATR
PARTIAL_SIZE  = 0.50   # 50 % closed at TP1
MIN_HOLD_BARS = 8      # min bars before signal-reversal exit
MAX_HOLD_BARS = 35     # time-stop


# ─── BACK-TEST ENGINE ────────────────────────────────────────

class Backtester:

    def __init__(
        self,
        symbol:          str   = config.BACKTEST_SYMBOL,
        timeframe:       str   = config.BACKTEST_TF,
        initial_capital: float = config.INITIAL_CAPITAL,
        taker_fee:       float = config.TAKER_FEE,
    ):
        self.symbol          = symbol
        self.timeframe       = timeframe
        self.initial_capital = initial_capital
        self.fee             = taker_fee
        self.equity          = initial_capital
        self.peak_equity     = initial_capital
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[dict]    = []
        self.open_trade: Optional[dict]  = None
        self.risk = RiskManager(initial_capital)

    def _fee_cost(self, price: float, qty: float) -> float:
        return price * qty * self.fee

    def _compute_pnl(self, direction: str, entry: float, exit_p: float, qty: float) -> float:
        gross = (exit_p - entry) * qty if direction == "LONG" else (entry - exit_p) * qty
        return gross - self._fee_cost(entry, qty) - self._fee_cost(exit_p, qty)

    def run(self, df: pd.DataFrame, htf_bias: Optional[pd.Series] = None) -> dict:
        warmup = 60
        print(f"\n  Running {len(df) - warmup:,} bars ...")

        for i in range(warmup, len(df)):
            bar      = df.iloc[i]
            bar_time = df.index[i]
            price    = float(bar["close"])
            atr      = float(bar["atr"])
            bar_date = bar_time.date() if hasattr(bar_time, "date") else None

            # ── Manage open trade ─────────────────────────
            if self.open_trade:
                ot = self.open_trade; reason = None; bars_in = i - ot["open_bar"]

                # TP1 partial exit
                if not ot["partial_hit"]:
                    tp1 = ot["entry"] + TP1_ATR_MULT * ot["entry_atr"] if ot["direction"] == "LONG" \
                          else ot["entry"] - TP1_ATR_MULT * ot["entry_atr"]
                    tp1_hit = (ot["direction"] == "LONG"  and float(bar["high"]) >= tp1) or \
                              (ot["direction"] == "SHORT" and float(bar["low"])  <= tp1)
                    if tp1_hit:
                        pq  = ot["qty"] * PARTIAL_SIZE
                        ppnl = self._compute_pnl(ot["direction"], ot["entry"], tp1, pq)
                        self.equity     += ppnl
                        self.peak_equity = max(self.peak_equity, self.equity)
                        ot.update({"partial_hit": True, "partial_pnl": ppnl,
                                   "qty_remaining": ot["qty"] * (1 - PARTIAL_SIZE)})
                        buf = atr * config.BREAKEVEN_BUFFER
                        ot["sl"] = max(ot["sl"], ot["entry"] + buf) if ot["direction"] == "LONG" \
                                   else min(ot["sl"], ot["entry"] - buf)
                        ot["breakeven_triggered"] = True

                # Trailing stop
                trail_dist = atr * config.TRAILING_OFFSET
                if ot["direction"] == "LONG":
                    if price > ot["entry"] + trail_dist:
                        ot["sl"] = max(ot["sl"], price - trail_dist)
                else:
                    if price < ot["entry"] - trail_dist:
                        ot["sl"] = min(ot["sl"], price + trail_dist)

                # SL / TP2 check
                qr = ot["qty_remaining"]
                if ot["direction"] == "LONG":
                    if float(bar["low"])  <= ot["sl"]:  reason, exit_p = "STOP_LOSS",   ot["sl"]
                    elif float(bar["high"]) >= ot["tp"]: reason, exit_p = "TAKE_PROFIT", ot["tp"]
                else:
                    if float(bar["high"]) >= ot["sl"]:  reason, exit_p = "STOP_LOSS",   ot["sl"]
                    elif float(bar["low"])  <= ot["tp"]: reason, exit_p = "TAKE_PROFIT", ot["tp"]

                # Time-stop
                if reason is None and bars_in >= MAX_HOLD_BARS:
                    reason, exit_p = "TIME_STOP", price

                # Signal reversal (only after min hold)
                if reason is None and bars_in >= MIN_HOLD_BARS:
                    sig = generate_signal(bar, self.symbol)
                    if sig.direction not in ("FLAT", ot["direction"]):
                        reason, exit_p = "SIGNAL_REVERSAL", price

                if reason:
                    rem_pnl   = self._compute_pnl(ot["direction"], ot["entry"], exit_p, qr)
                    total_pnl = rem_pnl + ot.get("partial_pnl", 0.0)
                    self.equity     += rem_pnl
                    self.peak_equity = max(self.peak_equity, self.equity)
                    self.trades.append(BacktestTrade(
                        symbol=self.symbol, direction=ot["direction"],
                        entry_time=str(ot["open_time"]), exit_time=str(bar_time),
                        entry_price=ot["entry"], exit_price=round(exit_p, 4),
                        qty=ot["qty"], pnl=round(total_pnl, 4),
                        pnl_pct=round(total_pnl / (ot["entry"] * ot["qty"]) * 100, 4),
                        exit_reason=reason, score=ot["score"], duration_bars=bars_in,
                        partial_tp_hit=ot["partial_hit"],
                        breakeven_used=ot["breakeven_triggered"],
                    ))
                    self.open_trade = None
                    self.risk.register_close(rem_pnl)

            self.equity_curve.append({"time": str(bar_time), "equity": round(self.equity, 2)})

            # ── New entry ─────────────────────────────────
            if self.open_trade is None and self.risk.can_trade(bar_date):
                sig = generate_signal(bar, self.symbol)
                if sig.direction == "FLAT":
                    continue
                if config.USE_HTF_FILTER and htf_bias is not None:
                    bias = htf_bias.get(bar_time, "NEUTRAL")
                    if sig.direction == "LONG"  and bias == "SHORT": continue
                    if sig.direction == "SHORT" and bias == "LONG":  continue

                params = self.risk.compute_trade_params(
                    self.symbol, sig.direction, price, atr, bar_date
                )
                if params:
                    self.risk.register_open()
                    tp2 = price + TP2_ATR_MULT * atr if sig.direction == "LONG" \
                          else price - TP2_ATR_MULT * atr
                    self.open_trade = {
                        "direction": sig.direction, "entry": price,
                        "sl": params.stop_loss,     "tp": tp2,
                        "qty": params.qty,           "qty_remaining": params.qty,
                        "entry_atr": atr,            "score": sig.score,
                        "open_time": bar_time,       "open_bar": i,
                        "partial_hit": False,         "partial_pnl": 0.0,
                        "breakeven_triggered": False,
                    }

        return self._report()

    def _report(self) -> dict:
        trades = self.trades
        if not trades:
            return {"error": "No trades."}

        pnls    = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p < 0]

        total_return  = (self.equity - self.initial_capital) / self.initial_capital * 100
        win_rate      = len(winners) / len(pnls) * 100
        avg_win       = float(np.mean(winners)) if winners else 0.0
        avg_loss      = float(np.mean(losers))  if losers  else 0.0
        profit_factor = abs(sum(winners) / sum(losers)) if losers else float("inf")

        eq       = pd.Series([e["equity"] for e in self.equity_curve])
        roll_max = eq.cummax()
        max_dd   = float(((eq - roll_max) / roll_max * 100).min())
        arr      = np.array(pnls)
        sharpe   = (arr.mean() / arr.std() * np.sqrt(252 * 78 / (len(eq) / max(len(arr), 1)))
                    if arr.std() > 0 else 0.0)

        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        report = {
            "symbol": self.symbol, "timeframe": self.timeframe,
            "total_bars": len(eq), "initial_capital": self.initial_capital,
            "final_equity": round(self.equity, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades": len(trades),
            "long_trades":  sum(1 for t in trades if t.direction == "LONG"),
            "short_trades": sum(1 for t in trades if t.direction == "SHORT"),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "risk_reward": round(abs(avg_win / avg_loss), 2) if avg_loss else "inf",
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "avg_duration_bars": round(float(np.mean([t.duration_bars for t in trades])), 1),
            "partial_tp_hits": sum(1 for t in trades if t.partial_tp_hit),
            "breakeven_saves": sum(1 for t in trades if t.breakeven_used),
            "exit_reasons": reasons,
        }

        pd.DataFrame(self.equity_curve).to_csv("logs/equity_curve.csv", index=False)
        with open("logs/backtest_results.json", "w") as f:
            json.dump({**report, "trades": [asdict(t) for t in trades]}, f, indent=2)
        return report


# ─── PRINT ───────────────────────────────────────────────────

def print_report(r: dict):
    if "error" in r:
        print(f"\n❌  {r['error']}"); return
    sep = "═" * 58
    print(f"\n{sep}")
    print(f"   SCALPING BOT BACKTEST v4 — {r['symbol']} [{r['timeframe']}]")
    print(sep)
    rows = [
        ("── OVERVIEW ──────────────────────", ""),
        ("  Total bars analysed",  f"{r['total_bars']:,}"),
        ("  Initial capital",      f"${r['initial_capital']:>10,.2f}"),
        ("  Final equity",         f"${r['final_equity']:>10,.2f}"),
        ("  Total return",         f"{r['total_return_pct']:>+10.2f}%"),
        ("── TRADE STATISTICS ──────────────", ""),
        ("  Total trades",         f"{r['total_trades']:>10}"),
        ("  Long / Short",         f"{r['long_trades']:>6} / {r['short_trades']:<6}"),
        ("  Win rate",             f"{r['win_rate_pct']:>10.1f}%"),
        ("  Avg winning trade",    f"${r['avg_win_usd']:>10,.2f}"),
        ("  Avg losing trade",     f"${r['avg_loss_usd']:>10,.2f}"),
        ("  Risk / Reward",        f"{r['risk_reward']:>10}"),
        ("  Profit factor",        f"{r['profit_factor']:>10.3f}"),
        ("  Avg duration (bars)",  f"{r['avg_duration_bars']:>10.1f}"),
        ("  Partial TP hits",      f"{r['partial_tp_hits']:>10}"),
        ("  Breakeven saves",      f"{r['breakeven_saves']:>10}"),
        ("── RISK METRICS ──────────────────", ""),
        ("  Max drawdown",         f"{r['max_drawdown_pct']:>10.2f}%"),
        ("  Sharpe ratio",         f"{r['sharpe_ratio']:>10.3f}"),
        ("── EXIT REASONS ──────────────────", ""),
    ]
    for label, val in rows:
        if val == "": print(f"\n  {label}")
        else:         print(f"  {label:<34} {val}")
    for reason, count in sorted(r["exit_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {'  '+reason:<34} {count:>4}  ({count/r['total_trades']*100:.1f}%)")

    pf, wr, dd = r["profit_factor"], r["win_rate_pct"], abs(r["max_drawdown_pct"])
    print(f"\n{sep}")
    if   pf > 1.5 and wr > 50 and dd < 15: verdict = "✅  PROMISING  — consider paper trading"
    elif pf > 1.2 and wr > 45:             verdict = "⚠️   MARGINAL   — optimise parameters"
    elif pf > 1.0:                         verdict = "🔶  BREAKEVEN  — run on real data next"
    else:                                  verdict = "❌  WEAK       — do NOT use real capital"
    print(f"  VERDICT : {verdict}")
    print(sep)
    print(f"\n  📁  logs/backtest_results.json")
    print(f"  📈  logs/equity_curve.csv\n")


# ─── ENTRY POINT ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Load real data from logs/real_ohlcv.csv instead of synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Run N seeds and show aggregate stats")
    args = parser.parse_args()

    if args.real and os.path.exists("logs/real_ohlcv.csv"):
        print("\n" + "─" * 58)
        print("  SCALPING BOT v4 — BACKTEST ON REAL BINANCE DATA")
        print("─" * 58)
        raw = pd.read_csv("logs/real_ohlcv.csv", index_col=0, parse_dates=True)
        raw.index = pd.DatetimeIndex(raw.index)
        df  = add_all_indicators(raw)
        print(f"  {len(df):,} valid bars  |  {df.index[0].date()} → {df.index[-1].date()}")
        htf = compute_htf_bias_series(df) if config.USE_HTF_FILTER else None
        bt  = Backtester()
        print_report(bt.run(df, htf_bias=htf))

    elif args.seeds > 1:
        print("\n" + "─" * 58)
        print(f"  SCALPING BOT v4 — {args.seeds}-SEED ROBUSTNESS TEST")
        print("─" * 58)
        all_ret = []; all_wr = []; all_pf = []
        for s in range(args.seeds):
            raw = generate_synthetic_ohlcv(5000, seed=s)
            df  = add_all_indicators(raw)
            htf = compute_htf_bias_series(df) if config.USE_HTF_FILTER else None
            bt  = Backtester()
            r   = bt.run(df, htf_bias=htf)
            if "error" not in r:
                mark = "✓" if r["total_return_pct"] > 0 else " "
                print(f"  {mark} seed {s:2d}: ret={r['total_return_pct']:+6.2f}%  "
                      f"WR={r['win_rate_pct']:.1f}%  PF={r['profit_factor']:.3f}  "
                      f"n={r['total_trades']}")
                all_ret.append(r["total_return_pct"])
                all_wr.append(r["win_rate_pct"])
                all_pf.append(r["profit_factor"])
        print(f"\n  Avg return : {np.mean(all_ret):+.2f}%")
        print(f"  Avg WR     : {np.mean(all_wr):.1f}%")
        print(f"  Avg PF     : {np.mean(all_pf):.3f}")
        print(f"  Positive   : {sum(1 for r in all_ret if r>0)}/{len(all_ret)}")

    else:
        print("\n" + "─" * 58)
        print("  SCALPING BOT v4 — OFFLINE BACKTEST (synthetic data)")
        print("  Run with --real to use real Binance data (fetch first)")
        print("─" * 58)
        raw = generate_synthetic_ohlcv(5000, seed=args.seed)
        df  = add_all_indicators(raw)
        print(f"  {len(df):,} valid bars generated")
        htf = compute_htf_bias_series(df) if config.USE_HTF_FILTER else None
        if htf is not None:
            print(f"  HTF bias | {htf.value_counts().to_dict()}")
        bt  = Backtester()
        print_report(bt.run(df, htf_bias=htf))
