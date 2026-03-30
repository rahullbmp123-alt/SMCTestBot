"""
Market structure analysis:
  - Swing highs / swing lows detection
  - HH, HL, LH, LL labelling
  - BOS (Break of Structure)
  - CHoCH (Change of Character)
  - Trend / bias determination
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SwingType(str, Enum):
    HH = "HH"   # Higher High
    HL = "HL"   # Higher Low
    LH = "LH"   # Lower High
    LL = "LL"   # Lower Low


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    swing_type: str  # "high" or "low"
    label: Optional[SwingType] = None


@dataclass
class StructureBreak:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: str          # "BOS" or "CHoCH"
    direction: str     # "bullish" or "bearish"
    broken_level: float


@dataclass
class MarketStructure:
    bias: Bias = Bias.NEUTRAL
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    breaks: list[StructureBreak] = field(default_factory=list)
    last_bos: Optional[StructureBreak] = None
    last_choch: Optional[StructureBreak] = None


def _find_swings(df: pd.DataFrame, lookback: int = 3) -> tuple[list[int], list[int]]:
    """
    Detect local swing highs and lows.
    A swing high at bar i: high[i] > high[i-1..i-lookback] AND high[i] > high[i+1..i+lookback]
    """
    highs, lows = [], []
    n = len(df)
    for i in range(lookback, n - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_h.max():
            highs.append(i)
        if df["low"].iloc[i] == window_l.min():
            lows.append(i)
    return highs, lows


def _label_swings(
    swing_points: list[SwingPoint], kind: str
) -> list[SwingPoint]:
    """Label each swing as HH/HL or LH/LL relative to the previous one."""
    labelled = []
    for i, sp in enumerate(swing_points):
        if i == 0:
            labelled.append(sp)
            continue
        prev = labelled[-1]
        if kind == "high":
            sp.label = SwingType.HH if sp.price > prev.price else SwingType.LH
        else:
            sp.label = SwingType.HL if sp.price > prev.price else SwingType.LL
        labelled.append(sp)
    return labelled


def _detect_breaks(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    current_bias: Bias,
) -> list[StructureBreak]:
    """
    BOS  = continuation break (break in direction of current bias)
    CHoCH = counter-trend break (potential bias flip)
    """
    breaks = []
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return breaks

    # last meaningful swing levels
    last_sh = swing_highs[-1]
    last_sl = swing_lows[-1]
    prev_sh = swing_highs[-2]
    prev_sl = swing_lows[-2]

    # check most recent candles for breaks
    recent = df.iloc[max(last_sh.index, last_sl.index):]
    for i, row in recent.iterrows():
        idx = df.index.get_loc(i)

        # Bullish break of previous swing high
        if row["close"] > prev_sh.price:
            kind = "BOS" if current_bias == Bias.BULLISH else "CHoCH"
            breaks.append(
                StructureBreak(
                    index=idx,
                    timestamp=i,
                    price=row["close"],
                    kind=kind,
                    direction="bullish",
                    broken_level=prev_sh.price,
                )
            )

        # Bearish break of previous swing low
        if row["close"] < prev_sl.price:
            kind = "BOS" if current_bias == Bias.BEARISH else "CHoCH"
            breaks.append(
                StructureBreak(
                    index=idx,
                    timestamp=i,
                    price=row["close"],
                    kind=kind,
                    direction="bearish",
                    broken_level=prev_sl.price,
                )
            )

    return breaks


def _determine_bias(
    swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]
) -> Bias:
    """
    Bullish  → majority of recent swings show HH + HL pattern
    Bearish  → majority of recent swings show LH + LL pattern

    Uses up to the last 4 swing points for a majority vote, making it
    significantly more robust than a single last-vs-prev comparison.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return Bias.NEUTRAL

    # Use last min(4, N) swings for majority vote
    n = min(4, len(swing_highs), len(swing_lows))
    recent_sh = swing_highs[-n:]
    recent_sl = swing_lows[-n:]
    possible = n - 1  # number of pairwise comparisons

    hh_count = sum(1 for i in range(1, n) if recent_sh[i].price > recent_sh[i - 1].price)
    hl_count = sum(1 for i in range(1, n) if recent_sl[i].price > recent_sl[i - 1].price)

    # Require majority (> 50%) on both highs and lows
    if hh_count > possible / 2 and hl_count > possible / 2:
        return Bias.BULLISH
    if hh_count < possible / 2 and hl_count < possible / 2:
        return Bias.BEARISH

    # Partial alignment — swing high direction is the tiebreaker
    if swing_highs[-1].price > swing_highs[-2].price:
        return Bias.BULLISH
    if swing_highs[-1].price < swing_highs[-2].price:
        return Bias.BEARISH
    return Bias.NEUTRAL


def analyse(df: pd.DataFrame, lookback: int = 3) -> MarketStructure:
    """
    Full market structure analysis on a single timeframe DataFrame.

    Expects columns: open, high, low, close (lowercase), DatetimeIndex.
    """
    ms = MarketStructure()

    high_idxs, low_idxs = _find_swings(df, lookback)

    # Build SwingPoint objects
    raw_sh = [
        SwingPoint(
            index=i,
            timestamp=df.index[i],
            price=float(df["high"].iloc[i]),
            swing_type="high",
        )
        for i in high_idxs
    ]
    raw_sl = [
        SwingPoint(
            index=i,
            timestamp=df.index[i],
            price=float(df["low"].iloc[i]),
            swing_type="low",
        )
        for i in low_idxs
    ]

    ms.swing_highs = _label_swings(raw_sh, "high")
    ms.swing_lows = _label_swings(raw_sl, "low")
    ms.bias = _determine_bias(ms.swing_highs, ms.swing_lows)

    ms.breaks = _detect_breaks(df, ms.swing_highs, ms.swing_lows, ms.bias)

    # Update bias from CHoCH signals
    for brk in reversed(ms.breaks):
        if brk.kind == "CHoCH":
            ms.last_choch = brk
            ms.bias = Bias.BULLISH if brk.direction == "bullish" else Bias.BEARISH
            break

    for brk in reversed(ms.breaks):
        if brk.kind == "BOS":
            ms.last_bos = brk
            break

    return ms
