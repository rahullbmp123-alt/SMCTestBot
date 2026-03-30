"""
ATR calculation and volatility utilities.
"""
import pandas as pd

from config import settings


def calculate_atr(df: pd.DataFrame, period: int = None) -> pd.Series:
    """True Range → ATR (Wilder smoothing)."""
    if period is None:
        period = settings.ATR_PERIOD
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - close).abs(), (low - close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def current_atr(df: pd.DataFrame, period: int = None) -> float:
    return float(calculate_atr(df, period).iloc[-1])


def lot_size_from_risk(
    account_balance: float,
    risk_pct: float,
    sl_pips: float,
    pip_value: float,
    contract_size: float,
    lot_tier_capital: float = 1000.0,
    min_lot: float = 0.01,
) -> float:
    """
    Calculate position size in lots, capped to balance-based tier.

    Tier rule: $1000 → max 0.01 lot, $2000 → max 0.02, $3000 → max 0.03, etc.
    This prevents over-sizing on small accounts and keeps risk per trade stable.

    risk_amount = account_balance * risk_pct
    lots = risk_amount / (sl_pips * pip_value * contract_size)
    """
    if sl_pips <= 0 or pip_value <= 0 or contract_size <= 0:
        return 0.0
    risk_amount = account_balance * risk_pct
    lot = risk_amount / (sl_pips * pip_value * contract_size)

    # Cap to balance tier: floor($balance / $lot_tier_capital) × min_lot
    tier_cap = max(min_lot, (int(account_balance // lot_tier_capital)) * min_lot)
    lot = min(lot, tier_cap)

    return round(lot, 2)
