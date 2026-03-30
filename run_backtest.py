"""
run_backtest.py — Main entry point for backtesting.

Usage:
    python run_backtest.py                          # uses sample generated data
    python run_backtest.py --csv data/my_data.csv   # use your own CSV
    python run_backtest.py --generate               # generate + run on synthetic data
    python run_backtest.py --live                   # fetch live XAUUSD data then backtest
    python run_backtest.py --live --interval 1h     # fetch 1h bars (up to 2 years)
    python run_backtest.py --learn                  # run backtest then trigger self-learning
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import load_csv, generate_sample_data
from backtesting.engine import BacktestEngine
from backtesting.plot import plot_equity_curve
from ai.probability_model import ProbabilityModel
from optimizer.self_learning import SelfLearningSystem
from config import settings
from utils.logger import get_logger

log = get_logger("run_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="SMC Bot Backtester")
    parser.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic XAUUSD data")
    parser.add_argument("--bars", type=int, default=5000, help="Bars to generate (default: 5000)")
    parser.add_argument("--live", action="store_true", help="Fetch live XAUUSD data from Yahoo Finance")
    parser.add_argument(
        "--interval", default="5m",
        choices=["1m", "5m", "15m", "1h", "1d"],
        help="Bar interval for --live fetch (default: 5m)",
    )
    parser.add_argument("--days", type=int, default=None, help="Lookback days for --live fetch")
    parser.add_argument("--balance", type=float, default=None, help="Initial balance override")
    parser.add_argument("--learn", action="store_true", help="Run self-learning after backtest")
    parser.add_argument("--no-plot", action="store_true", help="Skip equity curve plot")
    args = parser.parse_args()

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.csv:
        csv_path = args.csv
    elif args.live:
        from data.fetch_live import fetch_ohlcv, save_csv
        log.info(f"Fetching live {settings.SYMBOL} data ({args.interval})...")
        df_live = fetch_ohlcv(interval=args.interval, days=args.days)
        csv_path = f"data/{settings.SYMBOL.lower()}_{args.interval}.csv"
        save_csv(df_live, csv_path)
        log.info(f"Live data saved to {csv_path}")
    else:
        sample_path = f"data/sample_{settings.SYMBOL.lower()}_5m.csv"
        if args.generate or not Path(sample_path).exists():
            log.info(f"Generating {args.bars} synthetic {settings.SYMBOL} bars...")
            generate_sample_data(n_bars=args.bars)
        csv_path = sample_path

    df_5m = load_csv(csv_path)
    log.info(f"Data loaded: {len(df_5m)} bars | {df_5m.index[0]} → {df_5m.index[-1]}")

    # ── AI Model ──────────────────────────────────────────────────────────────
    ai = ProbabilityModel()

    # ── Run Backtest ──────────────────────────────────────────────────────────
    engine = BacktestEngine(
        df_5m=df_5m,
        ai_model=ai,
        initial_balance=args.balance,
    )
    result = engine.run()

    # ── Print Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SMC BOT BACKTEST RESULTS")
    print("=" * 60)
    for k, v in result.summary().items():
        print(f"  {k.replace('_', ' ').title():<25} {v}")
    print("=" * 60)
    if engine._rejections:
        total_rej = sum(engine._rejections.values())
        print(f"\n  Filter Rejections ({total_rej} total):")
        for reason, count in sorted(engine._rejections.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {count:>6}")
        print()
    print(f"  Trade log → {settings.LOGS_DIR / 'trades.csv'}")
    print(f"  Full log  → {settings.LOGS_DIR / 'smc_bot.log'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_equity_curve(result, save=True)
        print(f"  Chart     → {settings.LOGS_DIR / 'equity_curve.png'}")

    print("=" * 60 + "\n")

    # ── Self-learning ─────────────────────────────────────────────────────────
    if args.learn and settings.LEARNING_ENABLED:
        log.info("Running self-learning system...")
        from journal.trade_logger import TradeJournal
        journal = TradeJournal()
        df_trades = journal.load_all()

        if len(df_trades) >= settings.MIN_TRADES_BEFORE_ADAPT:
            sls = SelfLearningSystem()
            updated_params = sls.adapt(df_trades)
            sls.apply_reinforcement(df_trades)
            log.info(f"Self-learning complete. Params v{updated_params.version} saved.")
        else:
            log.warning(
                f"Only {len(df_trades)} trades — need {settings.MIN_TRADES_BEFORE_ADAPT} "
                "for self-learning"
            )


if __name__ == "__main__":
    main()
