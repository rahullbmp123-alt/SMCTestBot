"""Tests for the risk manager."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from journal.risk_manager import RiskManager, RiskState


def make_rm(balance=10000, daily_pnl=0, trades_today=0):
    state = RiskState(
        account_balance=balance,
        peak_balance=balance,
        daily_pnl=daily_pnl,
        trades_today=trades_today,
    )
    return RiskManager(state)


def test_can_trade_default():
    rm = make_rm()
    allowed, reason = rm.can_trade()
    assert allowed


def test_blocks_at_max_trades():
    from config import settings
    rm = make_rm(trades_today=settings.MAX_TRADES_PER_DAY)
    allowed, reason = rm.can_trade()
    assert not allowed
    assert "max trades" in reason.lower()


def test_blocks_at_daily_loss():
    from config import settings
    balance = 10000
    daily_loss = balance * settings.DAILY_LOSS_LIMIT + 1
    rm = make_rm(balance=balance - daily_loss, daily_pnl=-daily_loss)
    allowed, reason = rm.can_trade()
    assert not allowed


def test_drawdown_halts_trading():
    from config import settings
    balance = 10000 * (1 - settings.MAX_DRAWDOWN_PROTECTION - 0.01)
    state = RiskState(account_balance=balance, peak_balance=10000)
    rm = RiskManager(state)
    allowed, reason = rm.can_trade()
    assert not allowed
    assert rm.state.halted


def test_breakeven_trigger():
    # Buy: entry=1.1000, tp1=1.1030, current=1.1031 → should trigger
    assert RiskManager.check_breakeven(1.1000, 1.1031, 1.1030, "buy")
    assert not RiskManager.check_breakeven(1.1000, 1.1025, 1.1030, "buy")


def test_trailing_sl_buy():
    current_sl = 1.0990
    new_sl = RiskManager.trailing_sl(current_sl, 1.1050, 0.0010, "buy", atr_mult=1.5)
    # Should trail up: 1.1050 - 0.0015 = 1.1035 > 1.0990
    assert new_sl > current_sl


def test_trailing_sl_never_moves_against():
    current_sl = 1.1040
    # Price moved back — should NOT lower SL
    new_sl = RiskManager.trailing_sl(current_sl, 1.1020, 0.0010, "buy", atr_mult=1.5)
    assert new_sl == current_sl  # stays at 1.1040
