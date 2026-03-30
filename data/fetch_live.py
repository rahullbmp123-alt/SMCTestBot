"""
Fetch live historical XAUUSD data from Yahoo Finance and save as CSV.

Usage:
    python data/fetch_live.py                        # 5m data, last 60 days
    python data/fetch_live.py --interval 1h          # 1h data, last 2 years
    python data/fetch_live.py --interval 5m --days 30
    python data/fetch_live.py --out data/xauusd_1h.csv --interval 1h

Yahoo Finance limits:
    5m  → max 60 days history
    1h  → max 730 days history
    1d  → unlimited history
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.symbol_profiles import get_profile
from config import settings
from utils.logger import get_logger

log = get_logger("fetch_live")

# Map our internal timeframe labels to yfinance interval strings
_TF_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "4h": "1h",   # yfinance has no 4h; fetch 1h and resample externally
    "1d": "1d",
}

# Maximum lookback per interval (Yahoo Finance limits)
_MAX_DAYS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "1h": 730,
    "60m": 730,
    "1d": 3650,
}


def fetch_ohlcv(
    ticker: str | None = None,
    interval: str = "5m",
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Returns a DataFrame with DatetimeIndex (UTC) and columns:
        open, high, low, close, volume
    """
    # Resolve ticker from active symbol profile if not explicitly provided
    if ticker is None:
        profile = get_profile(settings.SYMBOL)
        ticker = profile.yahoo_ticker or settings.SYMBOL
        log.info(f"Using ticker '{ticker}' for symbol {settings.SYMBOL}")

    yf_interval = _TF_MAP.get(interval, interval)
    max_days = _MAX_DAYS.get(yf_interval, 60)

    if days is None:
        days = max_days
    elif days > max_days:
        log.warning(
            f"Requested {days} days but Yahoo Finance only supports {max_days} days "
            f"for {yf_interval} interval. Capping at {max_days}."
        )
        days = max_days

    period = f"{days}d" if start is None else None

    # Fallback tickers tried in order if the primary returns no data
    _FALLBACKS: dict[str, list[str]] = {
        "GC=F":     ["XAUUSD=X", "GLD"],   # gold futures → spot → ETF
        "XAUUSD=X": ["GC=F",     "GLD"],
        "XAGUSD=X": ["SI=F"],
        "SI=F":     ["XAGUSD=X"],
    }
    tickers_to_try = [ticker] + _FALLBACKS.get(ticker, [])

    df = pd.DataFrame()
    for t_sym in tickers_to_try:
        log.info(f"Fetching {t_sym} @ {yf_interval} | period={period or f'{start}→{end}'}")
        try:
            raw = yf.Ticker(t_sym).history(
                period=period,
                interval=yf_interval,
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
            )
            if not raw.empty:
                df = raw
                ticker = t_sym
                break
            log.warning(f"No data from {t_sym}, trying next...")
        except Exception as e:
            log.warning(f"Error fetching {t_sym}: {e}, trying next...")

    if df.empty:
        raise RuntimeError(
            f"No data returned for {tickers_to_try} interval={yf_interval}. "
            "Check ticker symbol and internet connection."
        )

    # Normalise columns
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "datetime"

    # Ensure UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Keep only OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()

    # Drop rows with NaN OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])

    log.info(f"Downloaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")
    return df


def save_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.reset_index()
    df_out.to_csv(out, index=False)
    log.info(f"Saved → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live XAUUSD data from Yahoo Finance")
    parser.add_argument("--ticker", default=None, help="Yahoo ticker (default: auto from SYMBOL in .env)")
    parser.add_argument(
        "--interval", default="5m",
        choices=["1m", "5m", "15m", "1h", "1d"],
        help="Bar interval (default: 5m)",
    )
    parser.add_argument("--days", type=int, default=None, help="Lookback days (default: max allowed)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/xauusd_<interval>.csv)")
    args = parser.parse_args()

    out_path = args.out or f"data/xauusd_{args.interval}.csv"

    df = fetch_ohlcv(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        start=args.start,
        end=args.end,
    )
    save_csv(df, out_path)
    print(f"\nSaved {len(df)} bars to {out_path}")
    print(f"Date range : {df.index[0]} → {df.index[-1]}")
    print(f"Price range: {df['low'].min():.2f} – {df['high'].max():.2f} USD\n")


if __name__ == "__main__":
    main()
