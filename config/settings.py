"""
Central configuration loader — reads from .env and validates all settings.

Symbol-specific defaults (POINT_VALUE, CONTRACT_SIZE, ATR_THRESHOLD,
SPREAD_LIMIT) are loaded from the symbol profile for the active SYMBOL.
Any value can still be overridden in .env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from config.symbol_profiles import get_profile

# Load .env from project root
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)
load_dotenv(_ROOT / ".env.example", override=False)  # fallback defaults


def _get(key: str, default=None, cast=str):
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Missing required env var: {key}")
    try:
        return cast(val)
    except (ValueError, TypeError) as e:
        raise EnvironmentError(f"Invalid value for {key}={val!r}: {e}") from e


def _get_list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


# ── Account ──────────────────────────────────────────────────────────────────
ACCOUNT_BALANCE: float    = _get("ACCOUNT_BALANCE",    "10000.0", float)
RISK_PER_TRADE: float     = _get("RISK_PER_TRADE",     "0.01",    float)
MAX_TRADES_PER_DAY: int   = _get("MAX_TRADES_PER_DAY",   "3",    int)
DAILY_LOSS_LIMIT: float   = _get("DAILY_LOSS_LIMIT",    "0.03",  float)
# After a loss, skip N × 5m bars before looking for new entries.
# Prevents repeatedly re-entering the same failing zone.
LOSS_COOLDOWN_BARS: int   = _get("LOSS_COOLDOWN_BARS",  "12",    int)  # 12 × 5m = 1 hour
# Zone cooldown — hours to block re-entry into the same price zone after any trade closes.
# Prevents cascading losses from re-entering a mitigated zone on the same day.
# 0.0 = disabled. Recommended: 3.0 (3 hours between trades at same zone level).
ZONE_COOLDOWN_HOURS: float = _get("ZONE_COOLDOWN_HOURS", "3.0",  float)
# Lot tier: one extra 0.01 lot allowed per LOT_TIER_CAPITAL dollars of balance.
# e.g. $1000→0.01, $2000-$2999→0.02, $3000-$3999→0.03 …
LOT_TIER_CAPITAL: float          = _get("LOT_TIER_CAPITAL",          "1000.0", float)
MIN_LOT_SIZE: float              = _get("MIN_LOT_SIZE",               "0.01",   float)
# Reject if actual $ risk at final lot > balance × risk_pct × this factor.
# 1.2 = allow up to 20% above intended risk to accommodate min-lot rounding.
# e.g. $1000 balance, 1% risk → max allowed loss = $12.
MAX_RISK_TOLERANCE_MULT: float   = _get("MAX_RISK_TOLERANCE_MULT",    "1.2",    float)
# Fixed lot override — set > 0 to always trade this lot size regardless of
# balance or SL distance. 0.0 = use dynamic risk-based sizing (default).
FIXED_LOT_SIZE: float            = _get("FIXED_LOT_SIZE",             "0.0",    float)

# ── Instrument ────────────────────────────────────────────────────────────────
SYMBOL: str = _get("SYMBOL", "XAUUSD")

# Load symbol profile — provides defaults for all instrument-specific values.
# Any of these can be overridden by setting the key explicitly in .env.
_profile = get_profile(SYMBOL)

POINT_VALUE: float = _get("POINT_VALUE", str(_profile.point_value), float)
CONTRACT_SIZE: float = _get("CONTRACT_SIZE", str(_profile.contract_size), float)

# ── Timeframes ────────────────────────────────────────────────────────────────
HTF_TIMEFRAMES: list[str] = _get_list("HTF_TIMEFRAMES", "4H,1H")
MTF_TIMEFRAME: str = _get("MTF_TIMEFRAME", "15m")
LTF_TIMEFRAME: str = _get("LTF_TIMEFRAME", "5m")

# ── Filters ───────────────────────────────────────────────────────────────────
ATR_THRESHOLD: float = _get("ATR_THRESHOLD", str(_profile.atr_threshold), float)
SPREAD_LIMIT: float = _get("SPREAD_LIMIT", str(_profile.spread_limit), float)
MIN_RR: float = _get("MIN_RR", "2.0", float)

# ── Sessions (UTC hours) ──────────────────────────────────────────────────────
SESSION_ASIAN_START: int   = _get("SESSION_ASIAN_START",   "0",  int)
SESSION_ASIAN_END: int     = _get("SESSION_ASIAN_END",     "5",  int)
SESSION_LONDON_START: int  = _get("SESSION_LONDON_START",  "7",  int)
SESSION_LONDON_END: int    = _get("SESSION_LONDON_END",    "16", int)
SESSION_NEWYORK_START: int = _get("SESSION_NEWYORK_START", "12", int)
SESSION_NEWYORK_END: int   = _get("SESSION_NEWYORK_END",   "21", int)

# ── Scoring ───────────────────────────────────────────────────────────────────
MIN_SCORE_EXECUTE: int = _get("MIN_SCORE_EXECUTE", "70", int)
MIN_SCORE_OPTIONAL: int = _get("MIN_SCORE_OPTIONAL", "50", int)

# ── AI ────────────────────────────────────────────────────────────────────────
AI_PROBABILITY_THRESHOLD: float = _get("AI_PROBABILITY_THRESHOLD", "0.65", float)
AI_MIN_SAMPLES_TO_TRAIN: int = _get("AI_MIN_SAMPLES_TO_TRAIN", "50", int)

# ── Self-learning ─────────────────────────────────────────────────────────────
MIN_TRADES_BEFORE_ADAPT: int = _get("MIN_TRADES_BEFORE_ADAPT", "20", int)
MAX_DRAWDOWN_PROTECTION: float = _get("MAX_DRAWDOWN_PROTECTION", "0.15", float)
LEARNING_ENABLED: bool = _get("LEARNING_ENABLED", "true").lower() == "true"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: Path = _ROOT / _get("DATA_DIR", "data")
LOGS_DIR: Path = _ROOT / _get("LOGS_DIR", "logs")
MODELS_DIR: Path = _ROOT / _get("MODELS_DIR", "ai/models")

for _d in (DATA_DIR, LOGS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Order Block detection ─────────────────────────────────────────────────────
OB_LOOKBACK: int              = _get("OB_LOOKBACK",               "50",  int)
OB_IMPULSE_THRESHOLD_HTF: float = _get("OB_IMPULSE_THRESHOLD_HTF", "1.5", float)
OB_IMPULSE_THRESHOLD_MTF: float = _get("OB_IMPULSE_THRESHOLD_MTF", "0.7", float)
OB_IMPULSE_LOOKAHEAD: int     = _get("OB_IMPULSE_LOOKAHEAD",       "10",  int)
OB_ZONE_MAX_AGE_BARS: int     = _get("OB_ZONE_MAX_AGE_BARS",       "50",  int)   # max 15m bars since zone formed (~12.5 h)

# ── Market structure / swing detection ────────────────────────────────────────
SWING_LOOKBACK_HTF: int = _get("SWING_LOOKBACK_HTF", "5", int)
SWING_LOOKBACK_MTF: int = _get("SWING_LOOKBACK_MTF", "3", int)
SWING_LOOKBACK_LTF: int = _get("SWING_LOOKBACK_LTF", "2", int)

# ── Liquidity detection ───────────────────────────────────────────────────────
LIQUIDITY_EQ_TOLERANCE: float = _get("LIQUIDITY_EQ_TOLERANCE", "0.20",  float)
LIQUIDITY_LOOKBACK: int       = _get("LIQUIDITY_LOOKBACK",      "20",    int)

# ── ATR ───────────────────────────────────────────────────────────────────────
ATR_PERIOD: int = _get("ATR_PERIOD", "14", int)

# ── FVG ───────────────────────────────────────────────────────────────────────
FVG_MIN_SIZE: float = _get("FVG_MIN_SIZE", "0.0", float)

# ── Backtest realism ─────────────────────────────────────────────────────────
# These are zero by default so they don't affect live trading logic,
# but setting realistic values here makes backtests far more accurate.
BACKTEST_SPREAD: float      = _get("BACKTEST_SPREAD",      "0.30", float)  # price units per trade (gold: ~$0.30)
BACKTEST_COMMISSION: float  = _get("BACKTEST_COMMISSION",  "0.0",  float)  # $ per lot per side (ECN e.g. 3.5)
BACKTEST_SLIPPAGE: float    = _get("BACKTEST_SLIPPAGE",    "0.10", float)  # price units on entry (gold: ~$0.10)

# ── Trade management ─────────────────────────────────────────────────────────
SL_ATR_BUFFER: float        = _get("SL_ATR_BUFFER",        "0.3", float)
TRAILING_SL_ATR_MULT: float = _get("TRAILING_SL_ATR_MULT", "3.0", float)
PARTIAL_TP_PCT: float       = _get("PARTIAL_TP_PCT",        "0.5", float)  # fraction of position closed at TP1
# USE_BREAKEVEN=true  → move SL to entry+buffer after price hits TP1 (protects profit, can get stopped at BE on pullbacks)
# USE_BREAKEVEN=false → keep original structure SL for the full trade (higher risk but lets winners run to TP2)
USE_BREAKEVEN: bool          = _get("USE_BREAKEVEN",         "true").lower() == "true"
# How far price must move (as multiple of initial risk) before SL moves to break-even.
# 1.0 = trigger at TP1 (1:1). 1.5 = trigger halfway between TP1 and TP2. 2.0 = trigger at TP2.
BREAKEVEN_TRIGGER_RR: float  = _get("BREAKEVEN_TRIGGER_RR",  "1.0",  float)
BREAKEVEN_BUFFER_PIPS: float = _get("BREAKEVEN_BUFFER_PIPS", "100.0", float) # pips above/below entry when moving SL to BE (× POINT_VALUE = price buffer)
ATR_TRAILING_BARS: int      = _get("ATR_TRAILING_BARS",     "50",  int)    # lookback bars for trailing SL ATR

# ── Signal confirmation ───────────────────────────────────────────────────────
CANDLE_WICK_RATIO: float     = _get("CANDLE_WICK_RATIO",     "2.0", float)  # wick:body ratio for hammer/shooting star
CHOCH_BREAKS_TO_CHECK: int   = _get("CHOCH_BREAKS_TO_CHECK", "5",   int)    # last N structure breaks to scan
MAX_SL_ATR_MULT: float       = _get("MAX_SL_ATR_MULT",       "5.0", float)  # reject if SL > N× ATR (prevents zero-lot trades)
# Minimum zone quality score (0–1). OB+internal_bos=0.5, OB+FVG=0.7, OB+FVG+confluence=0.9.
# Set to 0.5 to block standalone FVG-only zones (0.3) and bare OBs with no structure (0.4).
MIN_ZONE_QUALITY: float      = _get("MIN_ZONE_QUALITY",      "0.0", float)

# ── Backtest metrics ──────────────────────────────────────────────────────────
SHARPE_BARS_PER_DAY: int = _get("SHARPE_BARS_PER_DAY", "78",  int)    # 5m bars in a trading day (6.5h × 12)
RISK_FREE_RATE: float    = _get("RISK_FREE_RATE",       "0.0", float)  # annualised, for Sharpe calculation

# ── Backtest warmup ──────────────────────────────────────────────────────────
HTF_WARMUP_BARS: int = _get("HTF_WARMUP_BARS", "200", int)
MTF_WARMUP_BARS: int = _get("MTF_WARMUP_BARS", "100", int)
LTF_WARMUP_BARS: int = _get("LTF_WARMUP_BARS", "60",  int)

# ── Sweep scoring thresholds (in price units / pips) ─────────────────────────
SWEEP_HIGH_PIPS: float = _get("SWEEP_HIGH_PIPS", "10.0", float)
SWEEP_MED_PIPS: float  = _get("SWEEP_MED_PIPS",  "5.0",  float)
SWEEP_LOW_PIPS: float  = _get("SWEEP_LOW_PIPS",  "2.0",  float)
SWEEP_MIN_PIPS: float  = _get("SWEEP_MIN_PIPS",  "0.5",  float)

# ── Condition Filters (set false to bypass for testing) ───────────────────────
# HTF filters
FILTER_HTF_CONFLUENCE: bool  = _get("FILTER_HTF_CONFLUENCE",   "true").lower() == "true"
# MTF filters
FILTER_PREMIUM_DISCOUNT: bool = _get("FILTER_PREMIUM_DISCOUNT", "true").lower() == "true"
# LTF filters
FILTER_SESSION: bool         = _get("FILTER_SESSION",          "true").lower() == "true"
FILTER_SPREAD: bool          = _get("FILTER_SPREAD",           "true").lower() == "true"
FILTER_PRICE_IN_ZONE: bool   = _get("FILTER_PRICE_IN_ZONE",    "true").lower() == "true"
FILTER_ZONE_ENTRY_HALF: bool = _get("FILTER_ZONE_ENTRY_HALF",  "true").lower() == "true"
FILTER_LIQUIDITY_SWEEP: bool = _get("FILTER_LIQUIDITY_SWEEP",  "true").lower() == "true"
FILTER_CHOCH: bool           = _get("FILTER_CHOCH",            "true").lower() == "true"
FILTER_CANDLE_CONFIRMATION: bool = _get("FILTER_CANDLE_CONFIRMATION", "false").lower() == "true"
FILTER_ENTRY_BAR_CLOSE: bool = _get("FILTER_ENTRY_BAR_CLOSE",   "false").lower() == "true"
ENTRY_BAR_CLOSE_PCT: float   = _get("ENTRY_BAR_CLOSE_PCT",      "0.35",  float)
FILTER_MIN_RR: bool          = _get("FILTER_MIN_RR",           "true").lower() == "true"
# Engine filters
FILTER_SCORE: bool           = _get("FILTER_SCORE",            "true").lower() == "true"
FILTER_AI: bool              = _get("FILTER_AI",               "true").lower() == "true"
FILTER_RISK_MANAGEMENT: bool = _get("FILTER_RISK_MANAGEMENT",  "true").lower() == "true"

# ── MT5 Live Trading ──────────────────────────────────────────────────────────
MT5_LOGIN: str = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str = os.getenv("MT5_SERVER", "")
MT5_TERMINAL_PATH: str = os.getenv("MT5_TERMINAL_PATH", "")
# Symbol name as it appears in MT5 (Exness uses "XAUUSD" on most account types)
MT5_SYMBOL: str = os.getenv("MT5_SYMBOL", SYMBOL)
# Unique magic number to identify this bot's orders in MT5 (avoid conflicts with manual trades)
MT5_MAGIC: int = int(os.getenv("MT5_MAGIC", "20250101"))
# How many historical 5m bars to fetch for strategy warmup on startup (~35 days)
MT5_WARMUP_BARS: int = int(os.getenv("MT5_WARMUP_BARS", "10000"))
