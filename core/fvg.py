"""
Fair Value Gap (FVG) / Imbalance detection.

A bullish FVG:  low[i+1] > high[i-1]  (gap up — unfilled space)
A bearish FVG:  high[i+1] < low[i-1]  (gap down — unfilled space)

The candle at i is the "impulse" candle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FairValueGap:
    kind: str             # "bullish" or "bearish"
    top: float            # upper boundary
    bottom: float         # lower boundary
    timestamp: pd.Timestamp
    index: int
    filled: bool = False
    fill_timestamp: Optional[pd.Timestamp] = None

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def detect_fvgs(
    df: pd.DataFrame,
    min_size: float = None,   # filter tiny FVGs (in price units)
) -> list[FairValueGap]:
    if min_size is None:
        min_size = settings.FVG_MIN_SIZE
    fvgs: list[FairValueGap] = []

    for i in range(1, len(df) - 1):
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        next_high = df["high"].iloc[i + 1]
        next_low = df["low"].iloc[i + 1]

        # Bullish FVG
        if next_low > prev_high:
            size = next_low - prev_high
            if size >= min_size:
                fvgs.append(
                    FairValueGap(
                        kind="bullish",
                        top=next_low,
                        bottom=prev_high,
                        timestamp=df.index[i],
                        index=i,
                    )
                )

        # Bearish FVG
        if next_high < prev_low:
            size = prev_low - next_high
            if size >= min_size:
                fvgs.append(
                    FairValueGap(
                        kind="bearish",
                        top=prev_low,
                        bottom=next_high,
                        timestamp=df.index[i],
                        index=i,
                    )
                )

    return fvgs


def check_fills(df: pd.DataFrame, fvgs: list[FairValueGap]) -> list[FairValueGap]:
    """Mark FVGs as filled when price trades back through them."""
    for fvg in fvgs:
        if fvg.filled:
            continue
        future = df[df.index > fvg.timestamp]
        if fvg.kind == "bullish":
            filled_candles = future[future["low"] <= fvg.bottom]
        else:
            filled_candles = future[future["high"] >= fvg.top]

        if not filled_candles.empty:
            fvg.filled = True
            fvg.fill_timestamp = filled_candles.index[0]

    return fvgs


def get_valid_fvgs(fvgs: list[FairValueGap], bias: str) -> list[FairValueGap]:
    """Return unfilled FVGs aligned with the given bias."""
    return [f for f in fvgs if not f.filled and f.kind == bias]


def get_confluence_zone(
    ob: "OrderBlock", fvg: "FairValueGap"  # noqa: F821
) -> Optional[tuple[float, float]]:
    """
    Return the overlapping price range if an OB and FVG overlap.
    Overlapping = OB+FVG confluence → highest quality zone.
    """
    low = max(ob.bottom, fvg.bottom)
    high = min(ob.top, fvg.top)
    if high > low:
        return low, high
    return None
