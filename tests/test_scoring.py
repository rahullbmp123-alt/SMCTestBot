"""Tests for the scoring engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from strategy.scoring import score_signal, should_execute, ScoreBreakdown
from strategy.ltf_execution import TradeSignal
from strategy.mtf_refinement import RefinedZone


def make_signal(
    session="london",
    sweep_size=8.0,
    rr=2.5,
    choch=True,
    candle=True,
    has_ob=True,
    has_fvg=True,
    has_confluence=True,
    internal_bos=True,
    atr=0.001,
):
    zone = RefinedZone(
        kind="bullish",
        entry_top=1.1050,
        entry_bottom=1.1020,
        has_confluence=has_confluence,
        internal_bos=internal_bos,
    )
    from core.order_blocks import OrderBlock
    from core.fvg import FairValueGap
    import pandas as pd
    if has_ob:
        zone.ob = OrderBlock(kind="bullish", top=1.1050, bottom=1.1020, timestamp=pd.Timestamp("2024-01-01"), index=0)
    if has_fvg:
        zone.fvg = FairValueGap(kind="bullish", top=1.1048, bottom=1.1022, timestamp=pd.Timestamp("2024-01-01"), index=0)
    zone.quality_score = 0.9 if has_confluence else (0.6 if has_ob else 0.4)

    sig = TradeSignal(
        direction="buy",
        entry_price=1.1030,
        sl=1.1000,
        tp1=1.1060,
        tp2=1.1090,
        rr=rr,
        session=session,
        atr=atr,
        sweep_size=sweep_size,
        choch_confirmed=choch,
        candle_confirmation=candle,
        zone=zone,
    )
    return sig


def test_high_quality_signal_scores_above_70():
    sig = make_signal()
    total, bd = score_signal(sig)
    assert total >= 70, f"Expected ≥70, got {total}"
    assert should_execute(total)


def test_low_quality_signal_scores_below_50():
    sig = make_signal(
        session="off",
        sweep_size=0.1,
        rr=2.0,
        choch=False,
        candle=False,
        has_ob=False,
        has_fvg=False,
        has_confluence=False,
        internal_bos=False,
        atr=0.0001,
    )
    total, bd = score_signal(sig)
    assert total < 50, f"Expected <50, got {total}"
    assert not should_execute(total)


def test_score_breakdown_sums_to_total():
    sig = make_signal()
    total, bd = score_signal(sig)
    assert abs(bd.total - total) < 0.01


def test_optional_range():
    sig = make_signal(
        session="london",
        sweep_size=2.0,
        choch=True,
        candle=False,
        has_ob=True,
        has_fvg=False,
        has_confluence=False,
        internal_bos=False,
        atr=0.0006,
    )
    total, bd = score_signal(sig)
    # We don't assert exact range — just that it doesn't crash
    assert 0 <= total <= 100
