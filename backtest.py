"""
=============================================================
  SCALPING BOT — BACKTESTER
  Event-driven back-test engine.
  Downloads historical OHLCV, replays bar-by-bar,
  applies strategy + risk rules, then prints a full report.
=============================================================

  Usage:
    python backtest.py

  Output:
    • Console report with all performance metrics
    • logs/backtest_results.json   (machine-readable)
    • logs/equity_curve.csv        (for charting)
=============================================================
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Optional

import ccxt
import numpy as np
import pandas as pd

import config
from indicators import add_all_indicators
from signals    import generate_signal, Signal
from risk_manager import RiskManager, TradeParams

os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Backtest")


# ─── DATA DOWNLOAD ──────────────────────────────────────────

def download_ohlcv(
    symbol: str,
    timeframe: str,
    days: int,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Download up to `days` days of OHLCV from a public exchange endpoint.
    No API key required for public market data.
    """
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    since    = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    )

    all_bars = []
    current  = since
    print(f"Downloading {symbol} [{timeframe}] — {days} days ...")

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        except Exception as e:
            print(f"  Download error: {e}")
            break
        if not bars:
            break
        all_bars.extend(bars)
        current = bars[-1][0] + 1
        if bars[-1][0] >= exchange.milliseconds() - 60_000:
            break

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    df = df[df.index < datetime.utcnow()]
    print(f"  {len(df):,} bars downloaded ({df.index[0]} → {df.index[-1]})")
    return df


# ─── TRADE RECORD ───────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    score: int
    duration_bars: int


# ─── BACK-TEST ENGINE ───────────────────────────────────────

class Backtester:

    def __init__(
        self,
        symbol: str           = config.BACKTEST_SYMBOL,
        timeframe: str        = config.BACKTEST_TF,
        days: int             = config.BACKTEST_DAYS,
        initial_capital: float = config.INITIAL_CAPITAL,
        maker_fee: float      = config.MAKER_FEE,
        taker_fee: float      = config.TAKER_FEE,
    ):
        self.symbol          = symbol
        self.timeframe       = timeframe
        self.days            = days
        self.initial_capital = initial_capital
        self.fee             = taker_fee  # Assume market (taker) orders

        # State
        self.equity: float                   = initial_capital
        self.peak_equity: float              = initial_capital
        self.trades: List[BacktestTrade]     = []
        self.equity_curve: List[dict]        = []
        self.open_trade: Optional[dict]      = None
        self.risk                            = RiskManager(initial_capital)

    # ─── HELPERS ────────────────────────────────────────────

    def _fee_cost(self, price: float, qty: float) -> float:
        return price * qty * self.fee

    def _compute_pnl(self, direction: str, entry: float, exit_p: float, qty: float) -> float:
        if direction == "LONG":
            gross = (exit_p - entry) * qty
        else:
            gross = (entry - exit_p) * qty
        fees = self._fee_cost(entry, qty) + self._fee_cost(exit_p, qty)
        return gross - fees

    # ─── MAIN ENGINE ────────────────────────────────────────

    def run(self) -> dict:
        raw = download_ohlcv(self.symbol, self.timeframe, self.days)
        df  = add_all_indicators(raw)

        warmup = 60  # bars to skip (allow indicators to stabilise)
        print(f"Running backtest on {len(df) - warmup:,} bars ...\n")

        for i in range(warmup, len(df)):
            bar     = df.iloc[i]
            bar_time = df.index[i]
            price   = float(bar["close"])
            atr     = float(bar["atr"])

            # ── Manage open trade ──
            if self.open_trade:
                ot     = self.open_trade
                reason = None

                # Update trailing stop
                if ot["direction"] == "LONG":
                    trail_dist = atr * config.TRAILING_OFFSET
                    if price > ot["entry"] + trail_dist:
                        new_sl = price - trail_dist
                        if new_sl > ot["sl"]:
                            ot["sl"] = new_sl

                else:  # SHORT
                    trail_dist = atr * config.TRAILING_OFFSET
                    if price < ot["entry"] - trail_dist:
                        new_sl = price + trail_dist
                        if new_sl < ot["sl"]:
                            ot["sl"] = new_sl

                # Check SL / TP (use high/low for realistic fill)
                if ot["direction"] == "LONG":
                    if float(bar["low"]) <= ot["sl"]:
                        reason, exit_p = "STOP_LOSS",   ot["sl"]
                    elif float(bar["high"]) >= ot["tp"]:
                        reason, exit_p = "TAKE_PROFIT", ot["tp"]
                else:
                    if float(bar["high"]) >= ot["sl"]:
                        reason, exit_p = "STOP_LOSS",   ot["sl"]
                    elif float(bar["low"]) <= ot["tp"]:
                        reason, exit_p = "TAKE_PROFIT", ot["tp"]

                # Signal reversal
                if reason is None:
                    sig = generate_signal(bar, self.symbol)
                    if sig.direction not in ("FLAT", ot["direction"]):
                        reason, exit_p = "SIGNAL_REVERSAL", price

                if reason:
                    pnl     = self._compute_pnl(ot["direction"], ot["entry"], exit_p, ot["qty"])
                    pnl_pct = pnl / (ot["entry"] * ot["qty"]) * 100
                    self.equity += pnl
                    self.peak_equity = max(self.peak_equity, self.equity)

                    self.trades.append(BacktestTrade(
                        symbol        = self.symbol,
                        direction     = ot["direction"],
                        entry_time    = str(ot["open_time"]),
                        exit_time     = str(bar_time),
                        entry_price   = ot["entry"],
                        exit_price    = exit_p,
                        qty           = ot["qty"],
                        pnl           = round(pnl, 4),
                        pnl_pct       = round(pnl_pct, 4),
                        exit_reason   = reason,
                        score         = ot["score"],
                        duration_bars = i - ot["open_bar"],
                    ))
                    self.open_trade = None
                    self.risk.register_close(pnl)

            # ── Record equity snapshot ──
            self.equity_curve.append({"time": str(bar_time), "equity": round(self.equity, 2)})

            # ── Look for new entry ──
            if self.open_trade is None and self.risk.can_trade():
                sig = generate_signal(bar, self.symbol)
                if sig.direction != "FLAT":
                    params = self.risk.compute_trade_params(
                        self.symbol, sig.direction, price, atr
                    )
                    if params:
                        self.risk.register_open()
                        self.open_trade = {
                            "direction": sig.direction,
                            "entry":     price,
                            "sl":        params.stop_loss,
                            "tp":        params.take_profit,
                            "qty":       params.qty,
                            "score":     sig.score,
                            "open_time": bar_time,
                            "open_bar":  i,
                        }

        return self._report()

    # ─── PERFORMANCE REPORT ─────────────────────────────────

    def _report(self) -> dict:
        trades = self.trades
        if not trades:
            return {"error": "No trades generated — check your parameters."}

        pnls    = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p < 0]

        total_return  = (self.equity - self.initial_capital) / self.initial_capital * 100
        win_rate      = len(winners) / len(pnls) * 100 if pnls else 0
        avg_win       = np.mean(winners) if winners else 0
        avg_loss      = np.mean(losers)  if losers  else 0
        profit_factor = abs(sum(winners) / sum(losers)) if losers else float("inf")
        avg_duration  = np.mean([t.duration_bars for t in trades])

        # Max drawdown from equity curve
        eq = pd.Series([e["equity"] for e in self.equity_curve])
        roll_max  = eq.cummax()
        drawdown  = (eq - roll_max) / roll_max * 100
        max_dd    = float(drawdown.min())

        # Sharpe ratio (annualised, assume 5-min bars → ~105,120 bars/year)
        bars_per_year = 252 * 78  # 5-min bars in a trading year (24/7 crypto)
        returns_arr   = np.array(pnls)
        if returns_arr.std() > 0:
            sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(bars_per_year / (len(eq) / len(pnls)))
        else:
            sharpe = 0.0

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        report = {
            "symbol":           self.symbol,
            "timeframe":        self.timeframe,
            "period_days":      self.days,
            "initial_capital":  self.initial_capital,
            "final_equity":     round(self.equity, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades":     len(trades),
            "win_rate_pct":     round(win_rate, 2),
            "profit_factor":    round(profit_factor, 3),
            "avg_win_usd":      round(avg_win, 2),
            "avg_loss_usd":     round(avg_loss, 2),
            "risk_reward":      round(abs(avg_win / avg_loss), 2) if avg_loss else "∞",
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio":     round(sharpe, 3),
            "avg_duration_bars": round(avg_duration, 1),
            "exit_reasons":     reasons,
        }

        # Save equity curve
        pd.DataFrame(self.equity_curve).to_csv("logs/equity_curve.csv", index=False)

        # Save full report
        full = {**report, "trades": [asdict(t) for t in trades]}
        with open("logs/backtest_results.json", "w") as f:
            json.dump(full, f, indent=2)

        return report


# ─── PRETTY PRINT ───────────────────────────────────────────

def print_report(report: dict):
    if "error" in report:
        print(f"\n❌  {report['error']}")
        return

    stars = "═" * 56
    print(f"\n{stars}")
    print(f"  BACKTEST REPORT — {report['symbol']} [{report['timeframe']}]")
    print(stars)

    sections = [
        ("OVERVIEW",    [
            ("Symbol",            report["symbol"]),
            ("Timeframe",         report["timeframe"]),
            ("Period",            f"{report['period_days']} days"),
            ("Initial capital",   f"${report['initial_capital']:,.2f}"),
            ("Final equity",      f"${report['final_equity']:,.2f}"),
            ("Total return",      f"{report['total_return_pct']:+.2f}%"),
        ]),
        ("TRADE STATS", [
            ("Total trades",      report["total_trades"]),
            ("Win rate",          f"{report['win_rate_pct']:.1f}%"),
            ("Avg win",           f"${report['avg_win_usd']:,.2f}"),
            ("Avg loss",          f"${report['avg_loss_usd']:,.2f}"),
            ("Risk / Reward",     report["risk_reward"]),
            ("Profit factor",     report["profit_factor"]),
            ("Avg duration",      f"{report['avg_duration_bars']} bars"),
        ]),
        ("RISK",        [
            ("Max drawdown",      f"{report['max_drawdown_pct']:.2f}%"),
            ("Sharpe ratio",      report["sharpe_ratio"]),
        ]),
        ("EXIT REASONS", [(k, v) for k, v in report["exit_reasons"].items()]),
    ]

    for title, rows in sections:
        print(f"\n  ▸ {title}")
        for label, val in rows:
            print(f"    {label:<22} {val}")

    # Verdict
    print(f"\n{stars}")
    pf = report["profit_factor"]
    wr = report["win_rate_pct"]
    dd = abs(report["max_drawdown_pct"])
    if pf > 1.5 and wr > 50 and dd < 15:
        verdict = "✅  PROMISING — consider paper trading next"
    elif pf > 1.2 and wr > 45:
        verdict = "⚠️  MARGINAL — needs parameter optimisation"
    else:
        verdict = "❌  WEAK — do NOT deploy with real capital"
    print(f"  VERDICT:  {verdict}")
    print(f"{stars}\n")
    print("  Full results → logs/backtest_results.json")
    print("  Equity curve → logs/equity_curve.csv\n")


# ─── ENTRY POINT ────────────────────────────────────────────

if __name__ == "__main__":
    bt     = Backtester()
    result = bt.run()
    print_report(result)