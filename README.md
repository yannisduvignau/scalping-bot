# ⚡ CryptoScalp Bot

A professional Python scalping bot for cryptocurrency markets.  
Multi-indicator signal engine · ATR-based risk management · Built-in backtester.

---

## Architecture

```
scalping-bot/
├── config.py           ← All parameters in one place
├── indicators.py       ← EMA, RSI, StochRSI, MACD, BB, ATR, Volume
├── signals.py          ← Scoring engine (5 votes → LONG / SHORT / FLAT)
├── risk_manager.py     ← Position sizing, SL/TP, drawdown guard
├── bot.py              ← Live trading loop (ccxt)
├── backtest.py         ← Historical backtest (downloads real OHLCV)
├── backtest_offline.py ← Offline backtest with synthetic data
├── requirements.txt
└── logs/
    ├── bot.log
    ├── backtest_results.json
    └── equity_curve.csv
```

---

## Indicators & Strategy

### Signal Scoring System (5 votes)

| # | Indicator | LONG condition | SHORT condition |
|---|-----------|---------------|----------------|
| 1 | **EMA (9/21/50)** | EMA9 > EMA21 & price > EMA50 | EMA9 < EMA21 & price < EMA50 |
| 2 | **RSI (14)** | 30 < RSI < 55 (coming up from OS) | 45 < RSI < 70 (coming down from OB) |
| 3 | **Stochastic RSI** | %K crosses above %D from OS | %K crosses below %D from OB |
| 4 | **MACD (12/26/9)** | Bullish crossover or positive histogram | Bearish crossover or negative histogram |
| 5 | **Bollinger Bands** | Price touches lower band (%B < 0.15) | Price touches upper band (%B > 0.85) |

**Volume gate (mandatory):** Bar volume must exceed 1.2× rolling average — no vote, just blocks entry.

**Higher-TF filter:** Signals are cross-checked against the 15-min EMA trend.  
A LONG signal on 5m will be rejected if the 15m trend is bearish, and vice versa.

**Minimum score to trade:** 3 / 5 (configurable via `MIN_SIGNAL_SCORE`)

---

## Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RISK_PER_TRADE` | 1% | % of equity risked per trade |
| `MAX_OPEN_TRADES` | 3 | Maximum concurrent positions |
| `MAX_DAILY_LOSS` | 5% | Daily drawdown halt |
| `MAX_DRAWDOWN` | 15% | Total drawdown halt |
| `STOP_LOSS_ATR` | 1.5× | SL distance in ATR units |
| `TAKE_PROFIT_ATR` | 2.5× | TP distance in ATR units (R:R = 1.67) |
| `TRAILING_STOP` | True | Activates after entry + 0.5 ATR |

**Position sizing formula:**
```
risk_usd  = equity × RISK_PER_TRADE
sl_dist   = ATR(14) × STOP_LOSS_ATR
qty       = risk_usd / sl_dist
```

This ensures every trade risks exactly the same dollar amount regardless of volatility.

---

## Quick Start

### 1. Setup venv and install dependencies
```bash
python3 -m venv env
source env/bin/activate          # macOS/Linux
# .\env\Scripts\activate         # Windows

pip install --upgrade pip --quiet
pip install -r requirements.txt
```

### 2. Run the offline backtest (no API key needed)
```bash
python backtest_offline.py
```

### 3. Run the real historical backtest (downloads data from Binance public API)
```bash
python backtest.py
```

### 4. Paper trade (sandbox mode)
```python
# config.py
SANDBOX_MODE = True
API_KEY      = "your_testnet_key"
API_SECRET   = "your_testnet_secret"
```
```bash
python bot.py
```

### 5. Go live (only after thorough testing)
```python
# config.py
SANDBOX_MODE = False
```

> ⚠️ **Always** run in sandbox mode first. Only switch `SANDBOX_MODE = False`
> after consistent profitability in paper trading.

---

## Interpreting Backtest Results

| Metric | Minimum to proceed | Target |
|--------|-------------------|--------|
| Win rate | > 45% | > 55% |
| Profit factor | > 1.2 | > 1.5 |
| Max drawdown | < 20% | < 10% |
| Sharpe ratio | > 0.5 | > 1.0 |

The backtest verdict categories:

- ✅ **PROMISING** — PF > 1.5, WR > 50%, DD < 15% → paper trade
- ⚠️ **MARGINAL** — PF > 1.2, WR > 45% → optimise config first
- ❌ **WEAK** — below targets → do NOT deploy real capital

---

## Optimisation Tips

1. **Adjust `MIN_SIGNAL_SCORE`** — lower = more trades, higher = better quality
2. **Tune ATR multipliers** — tighter SL improves R:R but increases stop-outs
3. **Test multiple symbols** — add/remove from `SYMBOLS` list
4. **Try different timeframes** — 1m for aggressive scalping, 15m for swing
5. **Season the strategy** — backtest bull, bear, and sideways markets separately

---

## Supported Exchanges

Any exchange supported by [ccxt](https://github.com/ccxt/ccxt):
Binance, Bybit, OKX, Kraken, Coinbase Advanced, and 100+ others.

Change `EXCHANGE_ID` in `config.py` to switch.

---

## Disclaimer

> This software is for **educational purposes only**.  
> Cryptocurrency trading involves substantial risk of loss.  
> Past performance (including backtest results) does not guarantee future results.  
> Never trade with money you cannot afford to lose.