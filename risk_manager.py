"""
=============================================================
  SCALPING BOT — RISK MANAGER  (v2)
=============================================================
  v2 change: _refresh_daily() accepts an optional current_date.
  In live trading, date.today() is used as before.
  In backtesting, the bar's timestamp date is passed so that
  the daily P&L resets correctly at every simulated midnight
  — fixing the critical bug where the backtest treated the
  entire simulation as a single trading day and permanently
  halted after the first losing streak.
=============================================================
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from config import (
    RISK_PER_TRADE, MAX_OPEN_TRADES, MAX_DAILY_LOSS,
    MAX_DRAWDOWN, STOP_LOSS_ATR, TAKE_PROFIT_ATR,
    TRAILING_STOP, TRAILING_OFFSET,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeParams:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: float
    risk_usd: float
    trailing_active: bool  = False
    trailing_price: float  = 0.0
    breakeven_triggered: bool = False   # v2


@dataclass
class RiskState:
    equity: float
    peak_equity: float          = 0.0
    daily_start_equity: float   = 0.0
    last_reset_date: date       = field(default_factory=date.today)
    open_trades: int            = 0
    halted: bool                = False
    halt_reason: str            = ""
    daily_pnl: float            = 0.0
    total_pnl: float            = 0.0

    def __post_init__(self):
        self.peak_equity        = self.equity
        self.daily_start_equity = self.equity


class RiskManager:

    def __init__(self, initial_equity: float):
        self.state = RiskState(equity=initial_equity)
        logger.info("RiskManager initialized | equity=%.2f", initial_equity)

    # ─── DAILY RESET ────────────────────────────────────────

    def _refresh_daily(self, current_date: Optional[date] = None):
        """
        v2: accepts current_date so backtests can pass the bar date
        instead of relying on wall-clock date.today().
        """
        today = current_date or date.today()
        if self.state.last_reset_date != today:
            logger.info(
                "Daily reset | prev_equity=%.2f daily_pnl=%.2f",
                self.state.equity, self.state.daily_pnl,
            )
            self.state.daily_start_equity = self.state.equity
            self.state.daily_pnl          = 0.0
            self.state.last_reset_date    = today
            if self.state.halt_reason == "DAILY_LOSS":
                self.state.halted      = False
                self.state.halt_reason = ""

    # ─── GUARD CHECKS ───────────────────────────────────────

    def can_trade(self, current_date: Optional[date] = None) -> bool:
        self._refresh_daily(current_date)

        if self.state.halted:
            # ← no repeated log here; _halt() already logged it once
            return False

        daily_loss_pct = (
            (self.state.equity - self.state.daily_start_equity)
            / self.state.daily_start_equity
        )
        if daily_loss_pct <= -MAX_DAILY_LOSS:
            self._halt(f"DAILY_LOSS ({daily_loss_pct:.2%})")
            return False

        drawdown = (self.state.equity - self.state.peak_equity) / self.state.peak_equity
        if drawdown <= -MAX_DRAWDOWN:
            self._halt(f"MAX_DRAWDOWN ({drawdown:.2%})")
            return False

        if self.state.open_trades >= MAX_OPEN_TRADES:
            logger.debug("Max open trades reached (%d)", MAX_OPEN_TRADES)
            return False

        return True

    def _halt(self, reason: str):
        self.state.halted      = True
        self.state.halt_reason = reason
        # Single message — no duplicate.  bot.py uses StreamHandler for live mode;
        # backtest_offline configures WARNING-only console so this appears once.
        logger.warning("⛔  HALTED: %s — resumes next session", reason)

    # ─── POSITION SIZING ────────────────────────────────────

    def compute_trade_params(
        self,
        symbol: str,
        direction: str,
        entry: float,
        atr: float,
        current_date: Optional[date] = None,
    ) -> Optional[TradeParams]:
        if not self.can_trade(current_date):
            return None

        risk_usd = self.state.equity * RISK_PER_TRADE
        sl_dist  = atr * STOP_LOSS_ATR
        tp_dist  = atr * TAKE_PROFIT_ATR

        if sl_dist <= 0 or entry <= 0:
            logger.warning("Invalid SL distance or entry price, skipping.")
            return None

        qty = round(risk_usd / sl_dist, 6)

        if direction == "LONG":
            stop_loss   = round(entry - sl_dist, 6)
            take_profit = round(entry + tp_dist, 6)
        else:
            stop_loss   = round(entry + sl_dist, 6)
            take_profit = round(entry - tp_dist, 6)

        params = TradeParams(
            symbol=symbol, direction=direction,
            entry_price=entry, stop_loss=stop_loss,
            take_profit=take_profit, qty=qty, risk_usd=risk_usd,
        )
        logger.info(
            "[%s] %s | entry=%.4f SL=%.4f TP=%.4f qty=%.6f risk=$%.2f",
            symbol, direction, entry, stop_loss, take_profit, qty, risk_usd,
        )
        return params

    # ─── TRAILING STOP ──────────────────────────────────────

    def update_trailing_stop(
        self,
        params: TradeParams,
        current_price: float,
        atr: float,
    ) -> TradeParams:
        if not TRAILING_STOP:
            return params

        trail_dist = atr * TRAILING_OFFSET

        if params.direction == "LONG":
            if current_price > params.entry_price + trail_dist:
                new_trail = current_price - trail_dist
                if new_trail > params.trailing_price:
                    params.trailing_price  = new_trail
                    params.trailing_active = True
                    params.stop_loss       = max(params.stop_loss, new_trail)
        else:
            if current_price < params.entry_price - trail_dist:
                new_trail = current_price + trail_dist
                if new_trail < params.trailing_price or params.trailing_price == 0:
                    params.trailing_price  = new_trail
                    params.trailing_active = True
                    params.stop_loss       = min(params.stop_loss, new_trail)

        return params

    # ─── P&L ACCOUNTING ─────────────────────────────────────

    def register_open(self):
        self.state.open_trades += 1

    def register_close(self, pnl: float):
        self.state.open_trades = max(0, self.state.open_trades - 1)
        self.state.equity     += pnl
        self.state.daily_pnl  += pnl
        self.state.total_pnl  += pnl
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity
        logger.info(
            "Trade closed | pnl=%.2f equity=%.2f daily_pnl=%.2f",
            pnl, self.state.equity, self.state.daily_pnl,
        )

    def report(self) -> dict:
        return {
            "equity":       round(self.state.equity, 2),
            "peak_equity":  round(self.state.peak_equity, 2),
            "total_pnl":    round(self.state.total_pnl, 2),
            "daily_pnl":    round(self.state.daily_pnl, 2),
            "open_trades":  self.state.open_trades,
            "drawdown_pct": round(
                (self.state.equity - self.state.peak_equity) / self.state.peak_equity * 100, 2
            ),
            "halted":       self.state.halted,
            "halt_reason":  self.state.halt_reason,
        }
