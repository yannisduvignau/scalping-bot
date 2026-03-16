"""
=============================================================
  SCALPING BOT — MAIN BOT
  Live trading loop: fetches OHLCV, computes signals,
  manages positions, enforces risk rules.
=============================================================

  Quick start (sandbox):
    python bot.py

  Dependencies:
    pip install ccxt pandas numpy requests
=============================================================
"""

import time
import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Optional

import ccxt
import pandas as pd
import requests

import config
from indicators import add_all_indicators
from signals   import generate_signal, generate_higher_tf_bias, Signal
from risk_manager import RiskManager, TradeParams


# ─── LOGGING ────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level    = getattr(logging, config.LOG_LEVEL),
    format   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            config.LOG_FILE, maxBytes=5_000_000, backupCount=3
        ),
    ],
)
logger = logging.getLogger("ScalpBot")


# ─── TELEGRAM NOTIFICATION ──────────────────────────────────

def notify(message: str):
    """Send a Telegram message if credentials are configured."""
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        logger.warning("Telegram notification failed: %s", e)


# ─── EXCHANGE LAYER ─────────────────────────────────────────

class ExchangeClient:

    def __init__(self):
        exchange_class = getattr(ccxt, config.EXCHANGE_ID)
        params = {
            "apiKey": config.API_KEY,
            "secret": config.API_SECRET,
            "enableRateLimit": True,
        }
        if config.SANDBOX_MODE:
            params["options"] = {"defaultType": "future"}
        self.exchange = exchange_class(params)
        if config.SANDBOX_MODE:
            self.exchange.set_sandbox_mode(True)
        logger.info("Exchange: %s | Sandbox=%s", config.EXCHANGE_ID, config.SANDBOX_MODE)

    def fetch_ohlcv(self, symbol: str, tf: str, limit: int = 200) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_balance(self) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0))

    def place_market_order(self, symbol: str, side: str, qty: float) -> dict:
        logger.info("ORDER → %s %s %.6f", side.upper(), symbol, qty)
        if config.SANDBOX_MODE:
            logger.info("[SANDBOX] Order simulated — not sent to exchange.")
            return {"id": "SIMULATED", "symbol": symbol, "side": side, "amount": qty}
        order = self.exchange.create_market_order(symbol, side, qty)
        return order

    def place_stop_order(self, symbol: str, side: str, qty: float, stop_price: float) -> dict:
        if config.SANDBOX_MODE:
            return {"id": "SIM_STOP", "stop_price": stop_price}
        try:
            params = {"stopPrice": stop_price, "type": "stop_market"}
            order  = self.exchange.create_order(symbol, "stop", side, qty, stop_price, params)
            return order
        except Exception as e:
            logger.error("Stop order failed: %s", e)
            return {}

    def cancel_order(self, order_id: str, symbol: str):
        if config.SANDBOX_MODE or order_id in ("SIMULATED", "SIM_STOP"):
            return
        try:
            self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.warning("Cancel order failed: %s", e)


# ─── POSITION TRACKER ───────────────────────────────────────

class Position:
    """Represents a live open position."""

    def __init__(self, params: TradeParams, entry_order_id: str):
        self.params         = params
        self.entry_order_id = entry_order_id
        self.sl_order_id    = ""
        self.tp_order_id    = ""
        self.opened_at      = datetime.utcnow()
        self.pnl            = 0.0

    def compute_pnl(self, exit_price: float) -> float:
        p = self.params
        if p.direction == "LONG":
            return (exit_price - p.entry_price) * p.qty
        else:
            return (p.entry_price - exit_price) * p.qty

    def is_stopped(self, current_price: float) -> bool:
        p = self.params
        if p.direction == "LONG"  and current_price <= p.stop_loss:
            return True
        if p.direction == "SHORT" and current_price >= p.stop_loss:
            return True
        return False

    def is_target_hit(self, current_price: float) -> bool:
        p = self.params
        if p.direction == "LONG"  and current_price >= p.take_profit:
            return True
        if p.direction == "SHORT" and current_price <= p.take_profit:
            return True
        return False


# ─── MAIN BOT ───────────────────────────────────────────────

class ScalpBot:

    def __init__(self):
        self.exchange  = ExchangeClient()
        balance        = self.exchange.get_balance()
        self.risk      = RiskManager(initial_equity=balance or config.INITIAL_CAPITAL)
        self.positions: Dict[str, Position] = {}
        logger.info("Bot started | balance=%.2f USDT", self.risk.state.equity)
        notify(f"🤖 ScalpBot started | equity={self.risk.state.equity:.2f} USDT")

    # ─── DATA ───────────────────────────────────────────────

    def _get_enriched(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        try:
            df = self.exchange.fetch_ohlcv(symbol, tf, limit=300)
            return add_all_indicators(df)
        except Exception as e:
            logger.error("[%s] OHLCV fetch failed: %s", symbol, e)
            return None

    # ─── ENTRY LOGIC ────────────────────────────────────────

    def _try_open(self, symbol: str):
        if symbol in self.positions:
            return  # already in a trade for this symbol

        df     = self._get_enriched(symbol, config.TIMEFRAME)
        df_htf = self._get_enriched(symbol, config.HIGHER_TF)
        if df is None or df_htf is None:
            return

        signal  = generate_signal(df.iloc[-1], symbol)
        htf_bias = generate_higher_tf_bias(df_htf)

        # Higher-TF filter: don't trade against the trend
        if signal.direction == "LONG"  and htf_bias == "SHORT":
            logger.debug("[%s] Long signal rejected — HTF bias is SHORT", symbol)
            return
        if signal.direction == "SHORT" and htf_bias == "LONG":
            logger.debug("[%s] Short signal rejected — HTF bias is LONG", symbol)
            return

        if signal.direction == "FLAT":
            return

        params = self.risk.compute_trade_params(
            symbol, signal.direction, signal.close, signal.atr
        )
        if params is None:
            return

        # Submit market entry
        side  = "buy" if signal.direction == "LONG" else "sell"
        order = self.exchange.place_market_order(symbol, side, params.qty)
        self.risk.register_open()

        pos = Position(params, order.get("id", ""))
        self.positions[symbol] = pos

        msg = (
            f"✅ {signal.direction} {symbol} | "
            f"entry={params.entry_price:.4f} "
            f"SL={params.stop_loss:.4f} TP={params.take_profit:.4f} "
            f"score={signal.score}/5"
        )
        logger.info(msg)
        notify(msg)

    # ─── EXIT LOGIC ─────────────────────────────────────────

    def _manage_positions(self):
        for symbol in list(self.positions.keys()):
            df = self._get_enriched(symbol, config.TIMEFRAME)
            if df is None:
                continue

            pos   = self.positions[symbol]
            price = float(df.iloc[-1]["close"])
            atr   = float(df.iloc[-1]["atr"])

            # Update trailing stop
            pos.params = self.risk.update_trailing_stop(pos.params, price, atr)

            # Check exit conditions
            reason = None
            if pos.is_stopped(price):
                reason = "STOP_LOSS"
            elif pos.is_target_hit(price):
                reason = "TAKE_PROFIT"

            # Opposite signal exit
            new_sig = generate_signal(df.iloc[-1], symbol)
            if new_sig.direction not in ("FLAT", pos.params.direction):
                reason = "SIGNAL_REVERSAL"

            if reason:
                self._close_position(symbol, price, reason)

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        pos  = self.positions.pop(symbol)
        pnl  = pos.compute_pnl(exit_price)
        self.risk.register_close(pnl)

        side = "sell" if pos.params.direction == "LONG" else "buy"
        self.exchange.place_market_order(symbol, side, pos.params.qty)

        emoji = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} CLOSE {symbol} | {reason} | "
            f"exit={exit_price:.4f} pnl={pnl:+.2f} "
            f"equity={self.risk.state.equity:.2f}"
        )
        logger.info(msg)
        notify(msg)

    # ─── MAIN LOOP ──────────────────────────────────────────

    def run(self, poll_interval: int = 60):
        """
        Main trading loop. Polls every `poll_interval` seconds.
        Default 60s fits a 5-minute candle strategy (checks 1× per minute).
        """
        logger.info("Entering main loop | symbols=%s interval=%ds", config.SYMBOLS, poll_interval)
        while True:
            try:
                report = self.risk.report()
                logger.info("Status | %s", json.dumps(report))

                if self.risk.state.halted:
                    logger.warning("Bot halted. Sleeping 5 minutes...")
                    time.sleep(300)
                    continue

                # Manage existing positions first
                self._manage_positions()

                # Look for new entries
                for symbol in config.SYMBOLS:
                    self._try_open(symbol)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                notify("🛑 ScalpBot stopped.")
                break
            except Exception as e:
                logger.error("Unexpected error: %s", e, exc_info=True)

            time.sleep(poll_interval)


# ─── ENTRY POINT ────────────────────────────────────────────

if __name__ == "__main__":
    bot = ScalpBot()
    bot.run()