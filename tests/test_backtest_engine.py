"""Integration test for the backtesting engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from data.loader import generate_sample_data
from backtesting.engine import BacktestEngine, BacktestResult


def test_backtest_runs_without_error(tmp_path):
    """Full end-to-end backtest on synthetic data."""
    csv = tmp_path / "data.csv"
    generate_sample_data(n_bars=2000, out_path=csv, seed=1)

    from data.loader import load_csv
    df = load_csv(csv)

    engine = BacktestEngine(df_5m=df, initial_balance=10000)
    result = engine.run()

    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 0
    assert len(result.equity_curve) > 0
    assert 0.0 <= result.max_drawdown <= 1.0


def test_backtest_result_summary_keys(tmp_path):
    csv = tmp_path / "data.csv"
    generate_sample_data(n_bars=1000, out_path=csv, seed=2)

    from data.loader import load_csv
    df = load_csv(csv)

    engine = BacktestEngine(df_5m=df, initial_balance=5000)
    result = engine.run()
    summary = result.summary()

    expected_keys = {"total_trades", "wins", "losses", "win_rate", "profit_factor", "net_pnl"}
    assert expected_keys.issubset(set(summary.keys()))
