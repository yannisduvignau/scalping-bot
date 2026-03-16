"""
=============================================================
  SCALPING BOT — CONFIG  (v2 — improved parameters)
=============================================================
"""

# ─── EXCHANGE ────────────────────────────────────────────────
EXCHANGE_ID       = "binance"
API_KEY           = "YOUR_API_KEY"
API_SECRET        = "YOUR_API_SECRET"
SANDBOX_MODE      = True

# ─── TRADING UNIVERSE ────────────────────────────────────────
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

TIMEFRAME         = "5m"
HIGHER_TF         = "15m"
LEVERAGE          = 1

# ─── RISK MANAGEMENT ────────────────────────────────────────
RISK_PER_TRADE    = 0.01
MAX_OPEN_TRADES   = 3
MAX_DAILY_LOSS    = 0.05
MAX_DRAWDOWN      = 0.15

# v2: wider SL gives price room to breathe (was 1.5)
STOP_LOSS_ATR     = 2.5
# v2: TP extended for R:R ≥ 2.0 (was 2.5)
TAKE_PROFIT_ATR   = 5.0

TRAILING_STOP     = True
# v2: trailing too tight was clipping winners (was 0.5)
TRAILING_OFFSET   = 2.0

# v2: move SL to breakeven after price travels N×ATR in our favour
BREAKEVEN_ATR     = 2.0
BREAKEVEN_BUFFER  = 0.1            # lock in entry + 0.1×ATR (not exactly zero)

# ─── INDICATORS ──────────────────────────────────────────────
EMA_FAST          = 9
EMA_SLOW          = 21
EMA_TREND         = 50

RSI_PERIOD        = 14
RSI_OVERBOUGHT    = 70
RSI_OVERSOLD      = 30

STOCHRSI_PERIOD   = 14
STOCHRSI_K        = 3
STOCHRSI_D        = 3
STOCHRSI_OB       = 80
STOCHRSI_OS       = 20

MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9

BB_PERIOD         = 20
BB_STD            = 2.0

ATR_PERIOD        = 14

VOLUME_MA         = 20
# v2: raised to filter low-quality bars (was 1.2)
VOLUME_MULTIPLIER = 1.2

# v2: ADX trend-strength gate
ADX_PERIOD        = 14
ADX_THRESHOLD     = 18

# v2: HTF bias filter
USE_HTF_FILTER    = True

# ─── SIGNAL SCORING ─────────────────────────────────────────
# 6 votes total (EMA, RSI_dir, StochRSI_extreme, MACD_cross, BB_trend, Candle)
MIN_SIGNAL_SCORE  = 3

# ─── BACK-TEST ───────────────────────────────────────────────
BACKTEST_SYMBOL   = "BTC/USDT"
BACKTEST_TF       = "5m"
BACKTEST_DAYS     = 60
INITIAL_CAPITAL   = 10_000
MAKER_FEE         = 0.001
TAKER_FEE         = 0.001

# ─── LOGGING ─────────────────────────────────────────────────
LOG_LEVEL         = "INFO"
LOG_FILE          = "logs/bot.log"
TELEGRAM_TOKEN    = ""
TELEGRAM_CHAT_ID  = ""
