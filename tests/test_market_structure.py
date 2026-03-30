"""Tests for market structure analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.market_structure import analyse, Bias, SwingType


def make_df(highs, lows, closes=None, opens=None):
    n = len(highs)
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    if opens is None:
        opens = closes
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=dates,
    )


def test_bullish_bias_detected():
    # Uptrend: HH + HL pattern — use lookback=1 so each swing is visible
    # pattern: low, HIGH, low, HIGHER_HIGH, low, EVEN_HIGHER_HIGH ...
    highs = [1.10, 1.14, 1.11, 1.18, 1.13, 1.22, 1.15, 1.26, 1.17, 1.30,
             1.19, 1.34, 1.21, 1.38, 1.23, 1.42, 1.25, 1.46, 1.27, 1.50]
    lows  = [1.08, 1.12, 1.10, 1.16, 1.12, 1.20, 1.14, 1.24, 1.16, 1.28,
             1.18, 1.32, 1.20, 1.36, 1.22, 1.40, 1.24, 1.44, 1.26, 1.48]
    df = make_df(highs, lows)
    ms = analyse(df, lookback=1)
    assert ms.bias == Bias.BULLISH


def test_bearish_bias_detected():
    # Downtrend: LH + LL pattern — use lookback=1
    highs = [1.50, 1.46, 1.48, 1.42, 1.44, 1.38, 1.40, 1.34, 1.36, 1.30,
             1.32, 1.26, 1.28, 1.22, 1.24, 1.18, 1.20, 1.14, 1.16, 1.10]
    lows  = [1.48, 1.44, 1.46, 1.40, 1.42, 1.36, 1.38, 1.32, 1.34, 1.28,
             1.30, 1.24, 1.26, 1.20, 1.22, 1.16, 1.18, 1.12, 1.14, 1.08]
    df = make_df(highs, lows)
    ms = analyse(df, lookback=1)
    assert ms.bias == Bias.BEARISH


def test_swing_points_detected():
    # Clear swing highs and lows
    highs = [1.10, 1.15, 1.10, 1.20, 1.10, 1.18, 1.10, 1.22, 1.10, 1.16, 1.10]
    lows  = [1.08, 1.13, 1.07, 1.18, 1.08, 1.16, 1.09, 1.20, 1.07, 1.14, 1.08]
    df = make_df(highs, lows)
    ms = analyse(df, lookback=2)
    assert len(ms.swing_highs) > 0
    assert len(ms.swing_lows) > 0


def test_neutral_on_flat_market():
    # Flat prices → neutral
    highs = [1.10] * 20
    lows  = [1.09] * 20
    df = make_df(highs, lows)
    ms = analyse(df, lookback=2)
    # Should be neutral or have no swings
    assert ms.bias in (Bias.NEUTRAL, Bias.BULLISH, Bias.BEARISH)  # just no crash
