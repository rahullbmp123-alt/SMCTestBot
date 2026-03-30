"""Tests for Fair Value Gap detection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from core.fvg import detect_fvgs, check_fills, get_valid_fvgs


def make_fvg_df():
    """Three candles creating a bullish FVG: low[2] > high[0]."""
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    data = [
        {"open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005},  # candle 0
        {"open": 1.1010, "high": 1.1050, "low": 1.1008, "close": 1.1045},  # candle 1 (impulse)
        {"open": 1.1040, "high": 1.1060, "low": 1.1020, "close": 1.1055},  # candle 2 — low[2]=1.1020 > high[0]=1.1010
        {"open": 1.1055, "high": 1.1070, "low": 1.1040, "close": 1.1065},
        {"open": 1.1065, "high": 1.1080, "low": 1.1050, "close": 1.1075},
    ]
    return pd.DataFrame(data, index=dates)


def test_bullish_fvg_detected():
    df = make_fvg_df()
    fvgs = detect_fvgs(df)
    bullish = [f for f in fvgs if f.kind == "bullish"]
    assert len(bullish) >= 1


def test_fvg_boundaries_correct():
    df = make_fvg_df()
    fvgs = detect_fvgs(df)
    for fvg in fvgs:
        assert fvg.top > fvg.bottom
        assert fvg.size > 0


def test_fvg_fill_detection():
    df = make_fvg_df()
    fvgs = detect_fvgs(df)
    fvgs = check_fills(df, fvgs)
    for fvg in fvgs:
        assert isinstance(fvg.filled, bool)


def test_get_valid_fvgs():
    df = make_fvg_df()
    fvgs = detect_fvgs(df)
    valid = get_valid_fvgs(fvgs, "bullish")
    for f in valid:
        assert f.kind == "bullish"
        assert not f.filled
