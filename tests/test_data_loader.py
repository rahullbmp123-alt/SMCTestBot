"""Tests for data loading and resampling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
import tempfile

from data.loader import load_csv, generate_sample_data, load_multi_timeframe


def test_generate_sample_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.csv"
        df = generate_sample_data(n_bars=100, out_path=out)
        assert len(df) == 100
        assert "open" in df.columns
        assert "high" in df.columns


def test_load_csv_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.csv"
        generate_sample_data(n_bars=200, out_path=out)
        df = load_csv(out)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 200
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns


def test_load_csv_validates_ohlc():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "bad.csv"
        # Create CSV with inverted high/low
        pd.DataFrame({
            "datetime": ["2024-01-01 00:00:00"],
            "open": [1.0],
            "high": [0.9],   # high < low — invalid!
            "low": [1.1],
            "close": [1.0],
            "volume": [100],
        }).to_csv(out, index=False)
        df = load_csv(out)
        # Should drop the bad row and return empty DataFrame
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)


def test_load_multi_timeframe():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test5m.csv"
        generate_sample_data(n_bars=1000, out_path=out)
        tfs = load_multi_timeframe(out)
        assert "5m" in tfs
        assert "15m" in tfs
        assert "1H" in tfs
        assert "4H" in tfs
        # Higher TFs should have fewer bars
        assert len(tfs["15m"]) < len(tfs["5m"])
        assert len(tfs["1H"]) < len(tfs["15m"])
