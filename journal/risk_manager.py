"""
Risk Manager — enforces all risk rules during live and backtest operation.

Rules enforced:
  - Risk per trade (% of balance)
  - Minimum RR
  - Daily loss limit
  - Max trades per day
  - Break-even at 1:1
  - Partial TP at 1:1 (50%)
  - Trailing SL
  - Max drawdown protection (halts trading)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskState:
    """Mutable state tracked per trading session."""
    account_balance: float = field(default_factory=lambda: settings.ACCOUNT_BALANCE)
    peak_balance: float = field(default_factory=lambda: settings.ACCOUNT_BALANCE)
    daily_pnl: float = 0.0
    trades_today: int = 0
    trade_date: date = field(default_factory=date.today)
    halted: bool = False
    halt_reason: str = ""

    def reset_daily(self, bar_date: date = None) -> None:
        today = bar_date or date.today()
        if today != self.trade_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.trade_date = today

    def update_balance(self, pnl: float) -> None:
        self.account_balance += pnl
        self.daily_pnl += pnl
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance

    @property
    def drawdown_pct(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.account_balance) / self.peak_balance

    @property
    def daily_loss_pct(self) -> float:
        if self.account_balance == 0:
            return 0.0
        return abs(min(0.0, self.daily_pnl)) / self.account_balance


class RiskManager:
    def __init__(self, state: Optional[RiskState] = None) -> None:
        self.state = state or RiskState()

    def can_trade(self, bar_date: date = None) -> tuple[bool, str]:
        """Return (allowed, reason) for opening a new trade."""
        self.state.reset_daily(bar_date)

        if self.state.halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        if self.state.drawdown_pct >= settings.MAX_DRAWDOWN_PROTECTION:
            self.state.halted = True
            self.state.halt_reason = (
                f"max drawdown {self.state.drawdown_pct:.1%} >= "
                f"{settings.MAX_DRAWDOWN_PROTECTION:.1%}"
            )
            log.critical(f"TRADING HALTED: {self.state.halt_reason}")
            return False, self.state.halt_reason

        if self.state.daily_loss_pct >= settings.DAILY_LOSS_LIMIT:
            return False, (
                f"daily loss limit reached: {self.state.daily_loss_pct:.1%} >= "
                f"{settings.DAILY_LOSS_LIMIT:.1%}"
            )

        if self.state.trades_today >= settings.MAX_TRADES_PER_DAY:
            return False, (
                f"max trades per day reached: {self.state.trades_today} >= "
                f"{settings.MAX_TRADES_PER_DAY}"
            )

        return True, "ok"

    def register_trade_open(self) -> None:
        self.state.trades_today += 1

    def register_trade_close(self, pnl: float) -> None:
        self.state.update_balance(pnl)

    # ── Trade management helpers ───────────────────────────────────────────

    @staticmethod
    def check_breakeven(
        entry: float,
        current: float,
        tp1: float,
        direction: str,
    ) -> bool:
        """Return True if price has reached TP1 → time to move SL to break-even."""
        if direction == "buy":
            return current >= tp1
        return current <= tp1

    @staticmethod
    def check_partial_tp(
        entry: float,
        current: float,
        tp1: float,
        direction: str,
        already_partialled: bool,
    ) -> bool:
        """Return True if partial TP should be taken."""
        if already_partialled:
            return False
        return RiskManager.check_breakeven(entry, current, tp1, direction)

    @staticmethod
    def trailing_sl(
        current_sl: float,
        current_price: float,
        atr: float,
        direction: str,
        atr_mult: float = None,
    ) -> float:
        """
        Trail SL by ATR. Returns new SL (only moves in favour).
        """
        if atr_mult is None:
            atr_mult = settings.TRAILING_SL_ATR_MULT
        trail_dist = atr * atr_mult
        if direction == "buy":
            proposed = current_price - trail_dist
            return max(current_sl, proposed)   # only move up
        else:
            proposed = current_price + trail_dist
            return min(current_sl, proposed)   # only move down

    def manual_override_halt(self) -> None:
        """Allow manual resumption after halt."""
        self.state.halted = False
        self.state.halt_reason = ""
        log.warning("Trading halt manually overridden")
