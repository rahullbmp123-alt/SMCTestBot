"""
Fetch historical OHLCV data directly from MetaTrader 5 terminal.

Supports 5m, 15m, 1h, 4h bars. Higher timeframes are fetched directly
from MT5 (not resampled), giving broker-accurate candle boundaries.

Usage (standalone):
    python data/fetch_mt5.py                          # 5000 x 5m bars, auto symbol
    python data/fetch_mt5.py --bars 20000             # 20 000 x 5m bars
    python data/fetch_mt5.py --timeframe 1h           # 1h bars
    python data/fetch_mt5.py --symbol XAUUSD --bars 10000
    python data/fetch_mt5.py --out data/gold_5m.csv

Usage from code:
    from data.fetch_mt5 import fetch_ohlcv_mt5, save_csv
    df = fetch_ohlcv_mt5(n_bars=10000)
    save_csv(df, "data/xauusd_5m.csv")

Requirements:
    pip install MetaTrader5   (Windows only — MT5 terminal must be running)
    Linux: run inside Wine or a Windows VM.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings
from utils.logger import get_logger

log = get_logger("fetch_mt5")

# MT5 timeframe integer constants (matches MetaTrader5 package values)
_MT5_TIMEFRAMES: dict[str, int] = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "30m": 30,
    "1h":  16385,
    "4h":  16388,
    "1d":  16408,
}

# Approximate bars per calendar day per timeframe (trading days only, 5d/wk)
_BARS_PER_DAY: dict[str, float] = {
    "1m":  1440,
    "5m":  288,
    "15m": 96,
    "30m": 48,
    "1h":  24,
    "4h":  6,
    "1d":  1,
}


def days_to_bars(days: int, timeframe: str = "5m") -> int:
    """Convert a lookback period in calendar days to an approximate bar count."""
    trading_days = days * 5 / 7  # rough 5-day week approximation
    return max(1, int(trading_days * _BARS_PER_DAY.get(timeframe, 288)))


def fetch_ohlcv_mt5(
    symbol: Optional[str] = None,
    timeframe: str = "5m",
    n_bars: int = 5000,
    days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Connect to a running MT5 terminal and fetch historical OHLCV bars.

    Parameters
    ----------
    symbol    : MT5 symbol (default: settings.MT5_SYMBOL or settings.SYMBOL)
    timeframe : "1m", "5m", "15m", "30m", "1h", "4h", "1d"
    n_bars    : number of bars to fetch (ignored when *days* is set)
    days      : lookback window in calendar days; converted to bar count automatically

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close, volume
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 package not installed.\n"
            "Install with: pip install MetaTrader5\n"
            "Note: this package only works on Windows."
        )

    sym = symbol or getattr(settings, "MT5_SYMBOL", None) or settings.SYMBOL

    if timeframe not in _MT5_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Choose from: {list(_MT5_TIMEFRAMES)}"
        )
    tf_const = _MT5_TIMEFRAMES[timeframe]

    if days is not None:
        n_bars = days_to_bars(days, timeframe)
        log.info(f"Converted {days} days → {n_bars} bars for {timeframe}")

    # ── Connect ────────────────────────────────────────────────────────────
    path = getattr(settings, "MT5_TERMINAL_PATH", None) or None
    if not mt5.initialize(path=path):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    # Optional login (skipped when credentials not configured)
    login = getattr(settings, "MT5_LOGIN", None)
    password = getattr(settings, "MT5_PASSWORD", None)
    server = getattr(settings, "MT5_SERVER", None)

    if login and password:
        ok = mt5.login(int(login), password=password, server=server or None)
        if not ok:
            mt5.shutdown()
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(sym, True):
        log.warning(f"symbol_select({sym}) failed — will try fetching anyway")

    # ── Fetch ──────────────────────────────────────────────────────────────
    log.info(f"Fetching {n_bars} x {timeframe} bars for {sym} from MT5…")
    rates = mt5.copy_rates_from_pos(sym, tf_const, 0, n_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 returned no data for {sym} {timeframe}. "
            "Check that the symbol exists and history is downloaded in MT5."
        )

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df.index.name = "datetime"
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Drop the last (still-forming) bar
    df = df.iloc[:-1]

    log.info(
        f"Fetched {len(df)} bars | {df.index[0]} → {df.index[-1]} | "
        f"price range {df['low'].min():.2f}–{df['high'].max():.2f}"
    )
    return df


def save_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save DataFrame to CSV with 'datetime' index column."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out, index=False)
    log.info(f"Saved → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV data from MetaTrader 5"
    )
    parser.add_argument(
        "--symbol", default=None,
        help="MT5 symbol (default: MT5_SYMBOL from .env or SYMBOL)",
    )
    parser.add_argument(
        "--timeframe", default="5m",
        choices=list(_MT5_TIMEFRAMES),
        help="Bar timeframe (default: 5m)",
    )
    parser.add_argument(
        "--bars", type=int, default=5000,
        help="Number of bars to fetch (default: 5000, ~17 days of 5m)",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Lookback in calendar days (overrides --bars)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path (default: data/<symbol>_<timeframe>.csv)",
    )
    args = parser.parse_args()

    sym = args.symbol or getattr(settings, "MT5_SYMBOL", None) or settings.SYMBOL
    out_path = args.out or f"data/{sym.lower()}_{args.timeframe}_mt5.csv"

    df = fetch_ohlcv_mt5(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_bars=args.bars,
        days=args.days,
    )
    save_csv(df, out_path)

    print(f"\nFetched {len(df)} bars from MT5")
    print(f"Symbol    : {sym}")
    print(f"Timeframe : {args.timeframe}")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    print(f"Price     : {df['low'].min():.2f} – {df['high'].max():.2f}")
    print(f"Saved to  : {out_path}\n")


if __name__ == "__main__":
    main()
