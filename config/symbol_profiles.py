"""
Symbol profiles — instrument-specific parameters.

Each profile defines the contract specs and filter defaults for a symbol.
When a new symbol is set in .env, its profile is loaded automatically.
Individual values can still be overridden in .env.

To add a new symbol: add an entry to PROFILES below.

Fields:
    point_value     : smallest price movement value (in account currency per lot)
    contract_size   : units per standard lot
    atr_threshold   : minimum ATR to consider the market tradeable
    spread_limit    : maximum allowed spread (same units as price)
    price_baseline  : approximate current price (used for synthetic data generation)
    atr_5m_range    : (min, max) realistic ATR for a 5m bar (for synthetic data)
    yahoo_ticker    : Yahoo Finance ticker for live data fetching
    pip_decimals    : decimal places to round prices to
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolProfile:
    point_value: float
    contract_size: float
    atr_threshold: float
    spread_limit: float
    price_baseline: float
    atr_5m_range: tuple[float, float]
    yahoo_ticker: str
    pip_decimals: int


# ── Profile registry ──────────────────────────────────────────────────────────
PROFILES: dict[str, SymbolProfile] = {
    # Gold (spot) — primary ticker is GC=F (Gold Futures, reliable on Yahoo Finance)
    # XAUUSD=X is kept as fallback; it returns 404 intermittently since mid-2025
    "XAUUSD": SymbolProfile(
        point_value=0.01,
        contract_size=100,
        atr_threshold=3.0,      # 5m ATR at $4600+ gold is ~$3–8; was 1.0 (calibrated for $2000 gold)
        spread_limit=0.5,
        price_baseline=4600.0,  # updated from 2000.0 — used for synthetic data generation
        atr_5m_range=(3.0, 12.0),
        yahoo_ticker="GC=F",
        pip_decimals=2,
    ),
    # Silver (spot)
    "XAGUSD": SymbolProfile(
        point_value=0.001,
        contract_size=5000,
        atr_threshold=0.05,
        spread_limit=0.03,
        price_baseline=25.0,
        atr_5m_range=(0.05, 0.20),
        yahoo_ticker="XAGUSD=X",
        pip_decimals=3,
    ),
    # Euro / US Dollar
    "EURUSD": SymbolProfile(
        point_value=0.00001,
        contract_size=100000,
        atr_threshold=0.0003,
        spread_limit=0.0002,
        price_baseline=1.10,
        atr_5m_range=(0.0003, 0.0012),
        yahoo_ticker="EURUSD=X",
        pip_decimals=5,
    ),
    # British Pound / US Dollar
    "GBPUSD": SymbolProfile(
        point_value=0.00001,
        contract_size=100000,
        atr_threshold=0.0004,
        spread_limit=0.0003,
        price_baseline=1.27,
        atr_5m_range=(0.0004, 0.0015),
        yahoo_ticker="GBPUSD=X",
        pip_decimals=5,
    ),
    # US Dollar / Japanese Yen
    "USDJPY": SymbolProfile(
        point_value=0.001,
        contract_size=100000,
        atr_threshold=0.03,
        spread_limit=0.02,
        price_baseline=150.0,
        atr_5m_range=(0.03, 0.12),
        yahoo_ticker="JPY=X",
        pip_decimals=3,
    ),
    # US Dollar / Swiss Franc
    "USDCHF": SymbolProfile(
        point_value=0.00001,
        contract_size=100000,
        atr_threshold=0.0003,
        spread_limit=0.0002,
        price_baseline=0.90,
        atr_5m_range=(0.0003, 0.0010),
        yahoo_ticker="CHF=X",
        pip_decimals=5,
    ),
    # US Dollar / Canadian Dollar
    "USDCAD": SymbolProfile(
        point_value=0.00001,
        contract_size=100000,
        atr_threshold=0.0003,
        spread_limit=0.0002,
        price_baseline=1.36,
        atr_5m_range=(0.0003, 0.0010),
        yahoo_ticker="CAD=X",
        pip_decimals=5,
    ),
}

# Default fallback for unknown symbols
_DEFAULT_PROFILE = SymbolProfile(
    point_value=0.00001,
    contract_size=100000,
    atr_threshold=0.0003,
    spread_limit=0.0003,
    price_baseline=1.0,
    atr_5m_range=(0.0003, 0.0012),
    yahoo_ticker="",
    pip_decimals=5,
)


def get_profile(symbol: str) -> SymbolProfile:
    """Return the profile for *symbol*, falling back to the default."""
    return PROFILES.get(symbol.upper(), _DEFAULT_PROFILE)
