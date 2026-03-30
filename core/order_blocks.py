"""
Order Block (OB) detection.

An order block is the last opposing candle before an impulsive move that
causes a BOS or a strong displacement.

Rules:
  Bullish OB  → last red (bearish) candle before a bullish impulse
  Bearish OB  → last green (bullish) candle before a bearish impulse

Mitigation: OB is "mitigated" when price trades back into the 50% level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class OrderBlock:
    kind: str                  # "bullish" or "bearish"
    top: float
    bottom: float
    timestamp: pd.Timestamp
    index: int
    mitigated: bool = False
    mitigation_ts: Optional[pd.Timestamp] = None
    strength: float = 1.0      # 1.0 normal, >1 strong (backed by FVG etc.)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def is_in_premium_discount(self, price: float, bias: str) -> bool:
        """Check if price entering OB is at discount (buy) or premium (sell)."""
        if bias == "bullish":
            return price <= self.mid   # discount zone
        return price >= self.mid       # premium zone


def _is_impulse(df: pd.DataFrame, start: int, end: int, threshold_atr_mult: float = None) -> bool:
    """Return True if the move from start→end is impulsive (large & fast)."""
    if threshold_atr_mult is None:
        threshold_atr_mult = settings.OB_IMPULSE_THRESHOLD_HTF
    if end <= start:
        return False
    move = abs(df["close"].iloc[end] - df["close"].iloc[start])
    # simple ATR approximation over the window
    tr = (df["high"].iloc[start:end] - df["low"].iloc[start:end]).mean()
    if tr == 0:
        return False
    return (move / tr) >= threshold_atr_mult


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = None,
    impulse_threshold: float = None,
) -> list[OrderBlock]:
    """
    Detect bullish and bearish OBs in the last `lookback` candles.
    """
    if lookback is None:
        lookback = settings.OB_LOOKBACK
    if impulse_threshold is None:
        impulse_threshold = settings.OB_IMPULSE_THRESHOLD_HTF
    obs: list[OrderBlock] = []
    n = len(df)
    start = max(0, n - lookback)

    for i in range(start + 1, n - 1):
        candle = df.iloc[i]
        is_bearish_candle = candle["close"] < candle["open"]
        is_bullish_candle = candle["close"] > candle["open"]

        # Look ahead for impulse
        for j in range(i + 1, min(i + settings.OB_IMPULSE_LOOKAHEAD, n)):
            next_move = df["close"].iloc[j] - df["close"].iloc[i]

            # Bullish OB: bearish candle → bullish impulse
            if is_bearish_candle and next_move > 0:
                if _is_impulse(df, i, j, impulse_threshold):
                    obs.append(
                        OrderBlock(
                            kind="bullish",
                            top=float(candle["high"]),
                            bottom=float(candle["low"]),
                            timestamp=df.index[i],
                            index=i,
                        )
                    )
                    break

            # Bearish OB: bullish candle → bearish impulse
            if is_bullish_candle and next_move < 0:
                if _is_impulse(df, i, j, impulse_threshold):
                    obs.append(
                        OrderBlock(
                            kind="bearish",
                            top=float(candle["high"]),
                            bottom=float(candle["low"]),
                            timestamp=df.index[i],
                            index=i,
                        )
                    )
                    break

    return obs


def check_mitigation(df: pd.DataFrame, obs: list[OrderBlock]) -> list[OrderBlock]:
    """
    Mark OBs as mitigated only when price closes through the opposite boundary.

    A bullish OB is still valid when price wicks into it or tests it — that is
    exactly the re-entry event we trade. It is only dead when a candle CLOSES
    below the OB bottom (full structural break). Likewise a bearish OB is dead
    only when a candle closes above the OB top.
    """
    for ob in obs:
        if ob.mitigated:
            continue
        future = df[df.index > ob.timestamp]
        if ob.kind == "bullish":
            broken = future[future["close"] < ob.bottom]
        else:
            broken = future[future["close"] > ob.top]
        if not broken.empty:
            ob.mitigated = True
            ob.mitigation_ts = broken.index[0]
    return obs


def get_valid_obs(obs: list[OrderBlock], bias: str) -> list[OrderBlock]:
    """Return unmitigated OBs aligned with the given bias."""
    return [
        ob for ob in obs
        if not ob.mitigated and ob.kind == bias
    ]
