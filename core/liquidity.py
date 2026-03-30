"""
Liquidity detection:
  - Equal highs / equal lows (inducement levels)
  - Previous session highs/lows
  - Liquidity sweep detection
  - Buy-side / sell-side liquidity pools
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LiquidityLevel:
    price: float
    kind: str              # "BSL" (buy-side) or "SSL" (sell-side)
    timestamp: pd.Timestamp
    swept: bool = False
    sweep_timestamp: Optional[pd.Timestamp] = None
    sweep_size: float = 0.0  # pips swept beyond the level


@dataclass
class LiquidityMap:
    bsl: list[LiquidityLevel] = field(default_factory=list)   # buy-side (equal/prev highs)
    ssl: list[LiquidityLevel] = field(default_factory=list)   # sell-side (equal/prev lows)


def detect_equal_levels(
    df: pd.DataFrame, tolerance: float = None
) -> tuple[list[LiquidityLevel], list[LiquidityLevel]]:
    """
    Find equal highs (BSL) and equal lows (SSL).
    Two candles are "equal" if their high/low differ by < tolerance.
    """
    if tolerance is None:
        tolerance = settings.LIQUIDITY_EQ_TOLERANCE
    bsl, ssl = [], []
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index

    for i in range(1, len(df)):
        for j in range(max(0, i - settings.LIQUIDITY_LOOKBACK), i):
            if abs(highs[i] - highs[j]) <= tolerance:
                bsl.append(
                    LiquidityLevel(
                        price=max(highs[i], highs[j]),
                        kind="BSL",
                        timestamp=timestamps[i],
                    )
                )
            if abs(lows[i] - lows[j]) <= tolerance:
                ssl.append(
                    LiquidityLevel(
                        price=min(lows[i], lows[j]),
                        kind="SSL",
                        timestamp=timestamps[i],
                    )
                )
    return bsl, ssl


def check_sweeps(
    df: pd.DataFrame,
    liquidity_map: LiquidityMap,
    point_value: float = 0.00001,
) -> LiquidityMap:
    """
    Mark levels as swept if price wicked beyond them on a candle.
    A sweep requires: wick past level but close inside (stop-hunt pattern).
    """
    for lvl in liquidity_map.bsl:
        if lvl.swept:
            continue
        mask = (df["high"] > lvl.price) & (df["close"] < lvl.price)
        swept_candles = df[mask & (df.index > lvl.timestamp)]
        if not swept_candles.empty:
            row = swept_candles.iloc[0]
            lvl.swept = True
            lvl.sweep_timestamp = swept_candles.index[0]
            lvl.sweep_size = (row["high"] - lvl.price) / point_value

    for lvl in liquidity_map.ssl:
        if lvl.swept:
            continue
        mask = (df["low"] < lvl.price) & (df["close"] > lvl.price)
        swept_candles = df[mask & (df.index > lvl.timestamp)]
        if not swept_candles.empty:
            row = swept_candles.iloc[0]
            lvl.swept = True
            lvl.sweep_timestamp = swept_candles.index[0]
            lvl.sweep_size = (lvl.price - row["low"]) / point_value

    return liquidity_map


def build_liquidity_map(
    df: pd.DataFrame,
    tolerance: float = None,
    point_value: float = None,
) -> LiquidityMap:
    if tolerance is None:
        tolerance = settings.LIQUIDITY_EQ_TOLERANCE
    if point_value is None:
        point_value = settings.POINT_VALUE
    eq_bsl, eq_ssl = detect_equal_levels(df, tolerance)
    lm = LiquidityMap(bsl=eq_bsl, ssl=eq_ssl)
    lm = check_sweeps(df, lm, point_value)
    return lm


def get_recent_sweep(lm: LiquidityMap, kind: str = "SSL") -> Optional[LiquidityLevel]:
    """Return the most recently swept level of given kind."""
    pool = lm.ssl if kind == "SSL" else lm.bsl
    swept = [l for l in pool if l.swept]
    if not swept:
        return None
    return max(swept, key=lambda l: l.sweep_timestamp)
