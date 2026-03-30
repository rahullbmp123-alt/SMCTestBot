# SMC Trading Bot — Production-Grade ICT/Smart Money System

A complete, modular Python trading system built on **Smart Money Concepts (SMC)** and **ICT methodology**. Features a 3-layer Multi-Timeframe (MTF) model, ML-based probability filter, adaptive self-learning, and a full backtesting engine.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Architecture Overview

```
smc-bot/
├── config/            # Central settings loader (.env)
├── core/              # Pure market analysis primitives
│   ├── market_structure.py  # BOS, CHoCH, HH/HL/LH/LL, Bias
│   ├── liquidity.py         # BSL/SSL pools, sweep detection
│   ├── order_blocks.py      # OB detection & mitigation
│   ├── fvg.py               # Fair Value Gap detection & fills
│   └── atr.py               # ATR, volatility, position sizing
├── strategy/          # 3-layer MTF trading strategy
│   ├── htf_bias.py          # Layer 1: 4H+1H bias engine
│   ├── mtf_refinement.py    # Layer 2: 15m zone refinement
│   ├── ltf_execution.py     # Layer 3: 5m signal generation
│   └── scoring.py           # 0–100 setup scoring engine
├── ai/                # ML probability model
│   ├── feature_engineering.py
│   └── probability_model.py  # Random Forest + calibration
├── journal/           # Trade lifecycle management
│   ├── trade_logger.py      # Full trade journaling (CSV + JSON)
│   └── risk_manager.py      # Risk rules, BE, partial TP, trailing SL
├── backtesting/       # Historical simulation
│   ├── engine.py            # Walk-forward backtester
│   └── plot.py              # Equity curve + drawdown charts
├── optimizer/         # Self-learning system
│   └── self_learning.py     # Adaptive parameter tuning + RL logic
├── data/              # Data loading & generation
│   └── loader.py            # CSV loader, resampler, synthetic generator
├── mt5/               # Future MT5 integration (stubs)
│   └── connector.py         # Abstract ExecutionAdapter interface
├── tests/             # Pytest test suite
├── utils/             # Shared utilities
├── run_backtest.py    # Main entry point
├── .env.example       # All configuration parameters
└── requirements.txt
```

---

## Strategy Logic

### 3-Layer MTF Model

```
HTF (4H + 1H)  →  Bias Engine
    ↓
MTF (15m)      →  Zone Refinement
    ↓
LTF (5m)       →  Entry Execution
```

**Layer 1 — HTF Bias:**
- Detects HH/HL (bullish) or LH/LL (bearish) swing structure
- Identifies BOS (Break of Structure) and CHoCH (Change of Character)
- Maps buy-side / sell-side liquidity pools
- Determines premium/discount zones

**Layer 2 — MTF Refinement:**
- Narrows to 15m Order Blocks (OBs) and Fair Value Gaps (FVGs)
- Only keeps zones aligned with HTF bias
- Validates with internal 15m BOS/CHoCH
- Scores zone quality (OB+FVG confluence = highest quality)

**Layer 3 — LTF Execution:**
Entry fires ONLY when ALL conditions are met:
1. Price enters a refined zone
2. Liquidity sweep confirmed (stop-hunt)
3. CHoCH on 5m confirms directional shift
4. Optional: candle confirmation (engulfing / pin bar)

### Scoring Engine (0–100)

| Factor | Max Points |
|--------|-----------|
| HTF bias alignment | 20 |
| Liquidity sweep strength | 20 |
| CHoCH clarity | 15 |
| OB/FVG quality (confluence bonus) | 15 |
| Session timing | 10 |
| ATR/volatility | 10 |
| Clean structure + RR | 10 |

- **≥70** → Execute
- **50–69** → Optional
- **<50** → Skip

### AI Filter

Random Forest classifier (calibrated) predicts trade success probability.
Falls back to score-normalised heuristic until `AI_MIN_SAMPLES_TO_TRAIN` trades are logged.

### Risk Management

- **Risk per trade**: configurable % of account
- **Minimum RR**: 1:2 (configurable)
- **SL**: structure-based (below/above zone) + ATR buffer
- **TP1**: 1:1 → break-even + partial close (50%)
- **TP2**: minimum RR target
- **Trailing SL**: 1.5× ATR distance
- **Daily loss limit**: halts trading for the day
- **Max trades/day**: hard cap
- **Max drawdown protection**: halts system

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run Backtest

```bash
# Generate synthetic data + run backtest
python run_backtest.py --generate --bars 10000

# Use your own CSV
python run_backtest.py --csv data/your_xauusd_5m.csv

# Fetch live data then backtest (last 30 days, 5m bars)
python run_backtest.py --live --days 30

# Run backtest + trigger self-learning
python run_backtest.py --generate --learn
```

### 4. Run Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

---

## Fetching Live Data

Live data is fetched from **Yahoo Finance** via `yfinance`. Primary ticker for XAUUSD is `GC=F` (Gold Futures) with automatic fallback to `XAUUSD=X` and `GLD`.

### Via `run_backtest.py` (fetch + backtest in one command)

```bash
# Fetch last 60 days of 5m bars and backtest (60d is Yahoo's max for 5m)
python run_backtest.py --live

# Fetch last 30 days of 5m bars and backtest
python run_backtest.py --live --days 30

# Fetch 1h bars (up to 730 days) and backtest
python run_backtest.py --live --interval 1h

# Fetch 1 year of 1h bars and backtest
python run_backtest.py --live --interval 1h --days 365

# Fetch 1h bars for 2 years (Yahoo Finance max)
python run_backtest.py --live --interval 1h --days 730

# Fetch daily bars (unlimited history) and backtest
python run_backtest.py --live --interval 1d

# Fetch + backtest + run self-learning
python run_backtest.py --live --days 30 --learn

# Fetch live data, override starting balance, skip chart
python run_backtest.py --live --days 30 --balance 5000 --no-plot
```

### Via `data/fetch_live.py` (download only, saves CSV)

```bash
# Download 5m bars — last 60 days (Yahoo Finance max for 5m)
python data/fetch_live.py

# Download 5m bars — last 30 days
python data/fetch_live.py --days 30

# Download 1h bars — last 2 years (Yahoo Finance max for 1h)
python data/fetch_live.py --interval 1h

# Download 1h bars — last 365 days
python data/fetch_live.py --interval 1h --days 365

# Download daily bars
python data/fetch_live.py --interval 1d

# Specify date range (any interval)
python data/fetch_live.py --interval 1h --start 2024-01-01 --end 2024-12-31

# Save to a custom path
python data/fetch_live.py --interval 5m --days 30 --out data/gold_jan2025_5m.csv

# Use a specific ticker explicitly
python data/fetch_live.py --ticker GC=F --interval 5m --days 60
python data/fetch_live.py --ticker XAUUSD=X --interval 1h
```

### Yahoo Finance Limits

| Interval | Max Lookback | Notes |
|----------|-------------|-------|
| `1m`     | 7 days      | Very short window |
| `5m`     | 60 days     | Good for backtesting |
| `15m`    | 60 days     | Good for backtesting |
| `1h`     | 730 days    | ~2 years; best for longer tests |
| `1d`     | Unlimited   | Use for multi-year analysis |

### Supported Tickers (auto-selected by `SYMBOL` in `.env`)

| Symbol  | Primary    | Fallbacks          |
|---------|------------|--------------------|
| XAUUSD  | `GC=F`     | `XAUUSD=X`, `GLD`  |
| XAGUSD  | `XAGUSD=X` | `SI=F`             |
| EURUSD  | `EURUSD=X` | —                  |
| GBPUSD  | `GBPUSD=X` | —                  |
| USDJPY  | `JPY=X`    | —                  |

---

## CSV Data Format

Your CSV must have these columns (header names are case-insensitive):

```csv
datetime,open,high,low,close,volume
2024-01-02 08:00:00+00:00,1.10523,1.10601,1.10498,1.10577,1234
2024-01-02 08:05:00+00:00,1.10577,1.10622,1.10551,1.10609,987
```

- `datetime`: ISO format, UTC timezone preferred
- All OHLC values in the quote currency (e.g., 1.10523 for EURUSD)
- `volume` column is optional

---

## Configuration (.env)

Key parameters (see `.env.example` for the full list):

```env
ACCOUNT_BALANCE=1000.0
RISK_PER_TRADE=0.01          # 1% per trade
MAX_TRADES_PER_DAY=3
DAILY_LOSS_LIMIT=0.03        # 3% max daily loss
MIN_RR=2.0                   # minimum 1:2 risk/reward

# Sessions (UTC hours) — Asian gold session included
SESSION_ASIAN_START=0
SESSION_ASIAN_END=5
SESSION_LONDON_START=7
SESSION_LONDON_END=16
SESSION_NEWYORK_START=12
SESSION_NEWYORK_END=21

MIN_SCORE_EXECUTE=50         # score threshold to execute
AI_PROBABILITY_THRESHOLD=0.65

# Key filters (proven by backtesting)
FILTER_PREMIUM_DISCOUNT=true # only enter in HTF discount (buy) or premium (sell) zones
FILTER_HTF_CONFLUENCE=false  # require 4H+1H bias agreement (currently hurts performance)
BREAKEVEN_BUFFER_PIPS=100.0  # move SL to entry+100 pips after TP1 (converts BEs to wins)
```

---

## Trade Journal

Every trade is logged to:
- `logs/trades.csv` — flat table, easy to analyse in Excel/pandas
- `logs/trades.json` — full nested record with all actions
- `logs/smc_bot.log` — timestamped system log

Each trade record includes:
- Entry/exit prices, times, reasons
- SL, TP1, TP2, RR planned vs achieved
- All management actions (BE, partial TP, trailing SL updates)
- Score, AI probability, session, zone type
- P&L in pips and currency

---

## Self-Learning System

After accumulating `MIN_TRADES_BEFORE_ADAPT` trades (default: 20):

1. **Session analysis** — win rate per session → adjust session weights
2. **Setup analysis** — win rate per setup type → adjust setup weights
3. **Quality thresholds** — if overall WR < 40%, raise score/AI thresholds
4. **AI retraining** — retrain ML model with new trade data
5. **Reinforcement** — recent losing streak → reduce risk; winning streak → restore

All learned parameters saved to `optimizer/learned_params.json`.

---

## MT5 Integration (Future)

`mt5/connector.py` defines an abstract `ExecutionAdapter` interface.
`BacktestAdapter` is the paper-trading implementation.
`MT5Adapter` has stubs with implementation guidance.

To add live trading:
1. `pip install MetaTrader5`
2. Implement `MT5Adapter` methods following the inline TODO comments
3. Replace `BacktestAdapter` with `MT5Adapter` in the main loop

---

## Added Beyond Requirements

The following were added based on best practices for production trading systems:

- **Premium/Discount zones** — enforces entries only in discount (buy) or premium (sell) areas of the HTF range
- **Candle confirmation patterns** — optional engulfing/pin bar filter on LTF
- **Confluence detection** — OB+FVG overlap scoring bonus
- **Calibrated ML** — `CalibratedClassifierCV` for accurate probabilities
- **Walk-forward backtesting** — strict look-ahead prevention
- **Synthetic data generator** — test without real data
- **Abstract execution layer** — future-proof for any broker API
- **Per-session weight adaptation** — learns which sessions to avoid
- **Reinforcement-style risk scaling** — reduces lot size during losing streaks
