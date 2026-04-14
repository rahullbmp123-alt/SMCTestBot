"""
Fetch historical OHLCV data directly from MetaTrader 5 terminal.

Works on both Windows (native MetaTrader5 package) and Ubuntu/Linux
(mt5linux bridge — see setup instructions below).

──────────────────────────────────────────────────────────────
UBUNTU / LINUX SETUP  (one-time)
──────────────────────────────────────────────────────────────
The MetaTrader5 Python package only runs on Windows.  On Ubuntu
you need mt5linux, which is a thin TCP bridge:

  Step 1 — on the Windows machine that runs MT5:
    pip install mt5linux
    python -m mt5linux              # starts server on 0.0.0.0:18812

  Step 2 — on Ubuntu (this machine):
    pip install mt5linux

  Step 3 — add to .env:
    MT5_HOST=<windows-ip>           # e.g. 192.168.1.50
    MT5_PORT=18812                  # default port

  That's it.  run_backtest.py --mt5 will auto-detect Linux and use
  the bridge instead of the native package.
──────────────────────────────────────────────────────────────

Usage (standalone):
    python data/fetch_mt5.py                          # 5000 x 5m bars
    python data/fetch_mt5.py --days 90                # last 90 calendar days
    python data/fetch_mt5.py --timeframe 1h --days 365
    python data/fetch_mt5.py --symbol XAUUSD --bars 20000 --out data/gold_5m.csv

Usage from code:
    from data.fetch_mt5 import fetch_ohlcv_mt5, save_csv
    df = fetch_ohlcv_mt5(n_bars=10000)
    save_csv(df, "data/xauusd_5m.csv")
"""
from __future__ import annotations

import argparse
import os
import platform
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

# Approximate bars per calendar day per timeframe (5-day trading week)
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
    trading_days = days * 5 / 7   # rough 5-day trading week
    return max(1, int(trading_days * _BARS_PER_DAY.get(timeframe, 288)))


def _get_mt5_module():
    """
    Return an MT5 module/instance that exposes the standard MT5 API.

    - Windows: uses the native ``MetaTrader5`` package (module-level calls).
    - Linux/Mac: uses ``mt5linux`` bridge (instance-level calls, same API).

    Raises RuntimeError with clear install instructions if neither is found.
    """
    is_linux = platform.system() != "Windows"

    if is_linux:
        # ── Ubuntu / Linux — use mt5linux bridge ──────────────────────────
        try:
            from mt5linux import MetaTrader5  # type: ignore
        except ImportError:
            raise RuntimeError(
                "mt5linux is not installed.\n\n"
                "Ubuntu setup:\n"
                "  pip install mt5linux\n\n"
                "Then on your Windows MT5 machine run:\n"
                "  pip install mt5linux\n"
                "  python -m mt5linux\n\n"
                "Add to .env:\n"
                "  MT5_HOST=<windows-ip>   # IP of the Windows machine\n"
                "  MT5_PORT=18812          # default port\n"
            )
        host = getattr(settings, "MT5_HOST", "localhost") or "localhost"
        port = int(getattr(settings, "MT5_PORT", 18812) or 18812)
        log.info(f"Linux detected — connecting to mt5linux bridge at {host}:{port}")
        return MetaTrader5(host=host, port=port)
    else:
        # ── Windows — use native MetaTrader5 package ───────────────────────
        try:
            import MetaTrader5 as mt5  # type: ignore
            return mt5
        except ImportError:
            raise RuntimeError(
                "MetaTrader5 package not installed.\n"
                "Install with: pip install MetaTrader5\n"
                "The MT5 terminal must also be running and logged in."
            )


def fetch_ohlcv_mt5(
    symbol: Optional[str] = None,
    timeframe: str = "5m",
    n_bars: int = 5000,
    days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Connect to MT5 and fetch historical OHLCV bars.

    Parameters
    ----------
    symbol    : MT5 symbol (default: MT5_SYMBOL / SYMBOL from .env)
    timeframe : "1m", "5m", "15m", "30m", "1h", "4h", "1d"
    n_bars    : number of bars to fetch (ignored when *days* is set)
    days      : lookback in calendar days (auto-converted to bar count)

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns open/high/low/close/volume
    """
    sym = symbol or getattr(settings, "MT5_SYMBOL", None) or settings.SYMBOL

    if timeframe not in _MT5_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Choose from: {list(_MT5_TIMEFRAMES)}"
        )
    tf_const = _MT5_TIMEFRAMES[timeframe]

    if days is not None:
        n_bars = days_to_bars(days, timeframe)
        log.info(f"Converted {days} days → {n_bars} bars ({timeframe})")

    mt5 = _get_mt5_module()

    # ── Initialize ─────────────────────────────────────────────────────────
    path = getattr(settings, "MT5_TERMINAL_PATH", None) or None
    init_kwargs: dict = {}
    if path:
        init_kwargs["path"] = path

    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    try:
        # Optional login
        login = getattr(settings, "MT5_LOGIN", None)
        password = getattr(settings, "MT5_PASSWORD", None)
        server = getattr(settings, "MT5_SERVER", None)

        if login and password:
            ok = mt5.login(int(login), password=password, server=server or None)
            if not ok:
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

        # Make sure symbol is visible in Market Watch
        if not mt5.symbol_select(sym, True):
            log.warning(f"symbol_select({sym}) returned False — will try anyway")

        # ── Fetch bars ──────────────────────────────────────────────────────
        log.info(f"Fetching {n_bars} x {timeframe} bars for {sym} from MT5…")
        rates = mt5.copy_rates_from_pos(sym, tf_const, 0, n_bars)

    finally:
        mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 returned no data for {sym} {timeframe}.\n"
            "Check that:\n"
            "  1. The symbol exists and is shown in Market Watch\n"
            "  2. MT5 has downloaded enough history (Tools → History Center)\n"
            "  3. You are connected to the broker server"
        )

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df.index.name = "datetime"
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.iloc[:-1]   # drop last (still-forming) bar

    log.info(
        f"Fetched {len(df)} bars | {df.index[0]} → {df.index[-1]} | "
        f"price {df['low'].min():.2f}–{df['high'].max():.2f}"
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
        description="Fetch historical OHLCV data from MetaTrader 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol", default=None, help="MT5 symbol (default: from .env)")
    parser.add_argument(
        "--timeframe", default="5m",
        choices=list(_MT5_TIMEFRAMES),
        help="Bar timeframe (default: 5m)",
    )
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars (default: 5000)")
    parser.add_argument("--days", type=int, default=None, help="Lookback in calendar days (overrides --bars)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/<symbol>_<tf>_mt5.csv)")
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
