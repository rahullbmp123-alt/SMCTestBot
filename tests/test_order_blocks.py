"""Tests for Order Block detection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from core.order_blocks import detect_order_blocks, check_mitigation, get_valid_obs


def make_impulse_df():
    """Create a DataFrame with a clear bullish impulse → should produce bullish OB."""
    # Bearish candles, then strong bullish impulse
    data = []
    dates = pd.date_range("2024-01-01", periods=15, freq="1h", tz="UTC")
    for i in range(15):
        if i < 5:
            # Ranging / bearish
            o, c = 1.1020 - i*0.001, 1.1015 - i*0.001
            h, l = o + 0.0005, c - 0.0005
        elif i == 5:
            # Last bearish candle before impulse (potential OB)
            o, c = 1.0975, 1.0970
            h, l = 1.0980, 1.0965
        else:
            # Bullish impulse
            o = 1.0975 + (i - 5) * 0.002
            c = o + 0.002
            h, l = c + 0.0005, o - 0.0005
        data.append({"open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(data, index=dates)


def test_detect_ob_returns_list():
    df = make_impulse_df()
    obs = detect_order_blocks(df)
    assert isinstance(obs, list)


def test_ob_kind_is_valid():
    df = make_impulse_df()
    obs = detect_order_blocks(df)
    for ob in obs:
        assert ob.kind in ("bullish", "bearish")
        assert ob.top >= ob.bottom


def test_mitigation_check():
    df = make_impulse_df()
    obs = detect_order_blocks(df)
    obs = check_mitigation(df, obs)
    # All should have a mitigated bool
    for ob in obs:
        assert isinstance(ob.mitigated, bool)


def test_get_valid_obs_filters_by_bias():
    df = make_impulse_df()
    obs = detect_order_blocks(df)
    valid_bull = get_valid_obs(obs, "bullish")
    valid_bear = get_valid_obs(obs, "bearish")
    for ob in valid_bull:
        assert ob.kind == "bullish"
        assert not ob.mitigated
    for ob in valid_bear:
        assert ob.kind == "bearish"
        assert not ob.mitigated
