"""
Timeframe helpers — convert string TF to pandas freq and minutes.
"""

_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1H": 60, "2H": 120, "4H": 240, "1D": 1440,
}

_TF_FREQ = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1H": "1h", "2H": "2h", "4H": "4h", "1D": "1D",
}


def tf_to_minutes(tf: str) -> int:
    if tf not in _TF_MINUTES:
        raise ValueError(f"Unknown timeframe: {tf}. Choose from {list(_TF_MINUTES)}")
    return _TF_MINUTES[tf]


def tf_to_freq(tf: str) -> str:
    if tf not in _TF_FREQ:
        raise ValueError(f"Unknown timeframe: {tf}")
    return _TF_FREQ[tf]


def resample_ohlcv(df, target_tf: str):
    """Resample a 1-min OHLCV DataFrame to a higher timeframe."""
    import pandas as pd

    freq = tf_to_freq(target_tf)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(freq).agg(agg).dropna()
