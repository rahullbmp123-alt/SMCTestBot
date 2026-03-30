"""
Data loader — reads OHLCV CSV files and validates format.

Expected CSV format (see data/sample_eurusd_5m.csv):
  datetime,open,high,low,close,volume
  2024-01-02 08:00:00,1.10523,1.10601,1.10498,1.10577,1234
  ...

Datetime must be UTC or timezone-naive (treated as UTC).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import get_logger
from utils.timeframe import resample_ohlcv
from config.symbol_profiles import get_profile

log = get_logger(__name__)

_REQUIRED_COLS = {"open", "high", "low", "close"}


def load_csv(
    filepath: str | Path,
    datetime_col: str = "datetime",
    parse_tz: bool = True,
) -> pd.DataFrame:
    """
    Load and validate an OHLCV CSV.

    Returns a DataFrame with DatetimeIndex and lowercase OHLCV columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Date parsing
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found. Available: {list(df.columns)}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=parse_tz)
    df = df.set_index(datetime_col).sort_index()
    df.index.name = "datetime"

    # Validate columns
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add volume if absent
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Type cast
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop bad rows
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < before:
        log.warning(f"Dropped {before - len(df)} rows with NaN OHLC values")

    # Sanity check
    if not df.empty:
        bad = (df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["open"])
        if bad.any():
            log.warning(f"Found {bad.sum()} candles with invalid OHLC ordering — removing")
            df = df[~bad]

    if df.empty:
        log.warning(f"No valid bars in {path.name} after validation")
        return df

    log.info(f"Loaded {len(df)} bars from {path.name} ({df.index[0]} → {df.index[-1]})")
    return df


def load_multi_timeframe(
    base_csv: str | Path,
    base_tf: str = "5m",
) -> dict[str, pd.DataFrame]:
    """
    Load a 5m CSV and resample to all required timeframes.

    Returns:
        {"5m": df_5m, "15m": df_15m, "1H": df_1h, "4H": df_4h}
    """
    df_base = load_csv(base_csv)

    result = {base_tf: df_base}
    for tf in ["15m", "1H", "4H"]:
        try:
            result[tf] = resample_ohlcv(df_base, tf)
            log.info(f"Resampled to {tf}: {len(result[tf])} bars")
        except Exception as e:
            log.warning(f"Could not resample to {tf}: {e}")

    return result


def generate_sample_data(
    n_bars: int = 5000,
    symbol: str | None = None,
    out_path: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 5m OHLCV data for *symbol* using realistic
    price and volatility values from the symbol profile.

    If *symbol* is not given, the active SYMBOL from settings is used.
    """
    import numpy as np
    from config import settings as _settings

    sym = (symbol or _settings.SYMBOL).upper()
    profile = get_profile(sym)

    if out_path is None:
        out_path = f"data/sample_{sym.lower()}_5m.csv"

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 08:00", periods=n_bars, freq="5min", tz="UTC")

    # Volatility scaled to symbol — 0.05% per 5m bar is a reasonable starting point
    vol_per_bar = profile.atr_5m_range[0] / profile.price_baseline

    # Trending regimes: alternating bull/bear phases so HTF bias is detectable.
    # Each regime has a drift of ~0.2× vol_per_bar in the regime direction.
    n_regimes = 10
    regime_len = n_bars // n_regimes
    drift = np.zeros(n_bars)
    dir_sign = 1
    for r in range(n_regimes):
        start = r * regime_len
        end = min(start + regime_len, n_bars)
        drift[start:end] = dir_sign * vol_per_bar * 0.2
        dir_sign *= -1

    returns = rng.normal(drift, vol_per_bar, n_bars)
    close = profile.price_baseline * np.exp(returns.cumsum())

    atr_lo, atr_hi = profile.atr_5m_range
    atr_sim = rng.uniform(atr_lo, atr_hi, n_bars)
    high = close + atr_sim * rng.uniform(0.3, 0.7, n_bars)
    low = close - atr_sim * rng.uniform(0.3, 0.7, n_bars)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.integers(100, 5000, n_bars).astype(float)

    df = pd.DataFrame({
        "datetime": dates,
        "open": np.round(open_, profile.pip_decimals),
        "high": np.round(high, profile.pip_decimals),
        "low": np.round(low, profile.pip_decimals),
        "close": np.round(close, profile.pip_decimals),
        "volume": volume,
    })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"Sample data generated → {out} ({n_bars} bars, {sym})")
    return df
