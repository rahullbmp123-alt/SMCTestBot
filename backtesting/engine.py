"""
Backtesting Engine — simulates the full SMC strategy on historical OHLCV data.

Process per bar (LTF 5m):
  1. Build HTF/MTF/LTF DataFrames up to current bar
  2. Run HTF bias → MTF refinement → LTF signal generation
  3. Score signal → AI filter
  4. If approved: open trade, manage it bar-by-bar
  5. Log all trades

Outputs:
  - Equity curve
  - Win rate, profit factor, max drawdown, Sharpe
  - Per-trade log saved to journal
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from strategy.htf_bias import run_htf_bias, HTFAnalysis
from strategy.mtf_refinement import run_mtf_refinement, MTFAnalysis
from strategy.ltf_execution import generate_signal, TradeSignal
from strategy.scoring import score_signal, should_execute
from ai.probability_model import ProbabilityModel
from core.atr import calculate_atr
from journal.trade_logger import TradeJournal, TradeRecord
from journal.risk_manager import RiskManager, RiskState
from utils.timeframe import resample_ohlcv
from utils.logger import get_logger
from config import settings

log = get_logger(__name__)


@dataclass
class BacktestResult:
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[pd.Timestamp] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float("inf")
        return self.gross_profit / abs(self.gross_loss)

    @property
    def net_pnl(self) -> float:
        return self.gross_profit + self.gross_loss

    def summary(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "breakevens": self.breakevens,
            "win_rate": f"{self.win_rate:.1%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "net_pnl": f"${self.net_pnl:+.2f}",
            "gross_profit": f"${self.gross_profit:.2f}",
            "gross_loss": f"${self.gross_loss:.2f}",
            "max_drawdown": f"{self.max_drawdown:.1%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
        }


class BacktestEngine:
    """
    Walks forward through 5m data bar by bar.
    Higher TFs are derived by resampling.
    """

    # Minimum bars needed before we start analysing (from .env)
    HTF_WARMUP = settings.HTF_WARMUP_BARS
    MTF_WARMUP = settings.MTF_WARMUP_BARS
    LTF_WARMUP = settings.LTF_WARMUP_BARS

    def __init__(
        self,
        df_5m: pd.DataFrame,
        ai_model: Optional[ProbabilityModel] = None,
        initial_balance: float = None,
    ) -> None:
        self.df_5m = df_5m.copy()
        self.df_15m = resample_ohlcv(df_5m, "15m")
        self.df_1h = resample_ohlcv(df_5m, "1H")
        self.df_4h = resample_ohlcv(df_5m, "4H")

        self.ai = ai_model or ProbabilityModel()
        self.balance = initial_balance or settings.ACCOUNT_BALANCE
        self.journal = TradeJournal()
        self.risk = RiskManager(RiskState(account_balance=self.balance))
        self.result = BacktestResult()

        # Active trade state
        self._active_signal: Optional[TradeSignal] = None
        self._active_record: Optional[TradeRecord] = None
        self._partialled: bool = False
        self._be_moved: bool = False
        self._current_sl: float = 0.0
        self._remaining_lot_ratio: float = 1.0  # fraction of lot still open after partial TP
        self._partial_pnl: float = 0.0           # accumulated partial TP P&L for current trade
        self._last_loss_ts: Optional[pd.Timestamp] = None  # cooldown after a loss
        self._mfe_best_price: float = 0.0  # best price reached in trade's favour (for MFE)
        self._last_zone_mid: float = 0.0              # zone midpoint of most recently closed trade
        self._last_trade_close_ts: Optional[pd.Timestamp] = None  # when last trade closed

        # HTF/MTF caches — recompute only when a new higher-TF bar forms
        self._htf_cache: Optional[HTFAnalysis] = None
        self._mtf_cache: Optional[MTFAnalysis] = None
        self._last_1h_bar: Optional[pd.Timestamp] = None
        self._last_15m_bar: Optional[pd.Timestamp] = None

        # Rejection counters — keyed by filter name
        self._rejections: dict[str, int] = {}

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_slice(self, df: pd.DataFrame, up_to: pd.Timestamp, n: int) -> pd.DataFrame:
        sub = df[df.index <= up_to]
        return sub.iloc[-n:] if len(sub) >= n else sub

    def _is_sl_hit(self, low: float, high: float) -> bool:
        sig = self._active_signal
        if sig.direction == "buy":
            return low <= self._current_sl
        return high >= self._current_sl

    def _is_tp_hit(self, low: float, high: float, tp: float, direction: str) -> bool:
        if direction == "buy":
            return high >= tp
        return low <= tp

    def _close_active(
        self, exit_price: float, exit_reason: str, bar_ts: pd.Timestamp
    ) -> None:
        sig = self._active_signal
        rec = self._active_record

        # Compute MFE in pips (best price reached in trade's favour)
        if sig.direction == "buy":
            mfe_pips = (self._mfe_best_price - sig.entry_price) / settings.POINT_VALUE
        else:
            mfe_pips = (sig.entry_price - self._mfe_best_price) / settings.POINT_VALUE
        mfe_pips = max(0.0, mfe_pips)  # never negative

        closed = self.journal.close_trade(
            trade_id=sig.trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            account_balance=self.risk.state.account_balance,
            exit_time=str(bar_ts),
            max_favorable_excursion=mfe_pips,
        )
        if closed:
            # Scale P&L to remaining lot (after any partial TP already booked)
            remaining_lot = closed.lot_size * self._remaining_lot_ratio
            if self._remaining_lot_ratio != 1.0:
                closed.pnl_currency = round(
                    closed.pnl_pips * settings.POINT_VALUE * settings.CONTRACT_SIZE * remaining_lot, 2
                )
            # Deduct commission on the remaining close
            commission = settings.BACKTEST_COMMISSION * remaining_lot
            pnl = closed.pnl_currency - commission
            self.risk.register_trade_close(pnl)
            # Record balance AFTER P&L is applied
            closed.account_balance_after = round(self.risk.state.account_balance, 2)
            self.result.total_trades += 1
            # Use total P&L (remaining lot + any partial TP already booked) to determine
            # the true outcome — avoids counting partial-TP + BE-close as a breakeven/loss
            total_pnl = pnl + self._partial_pnl
            if total_pnl > 0:
                self.result.wins += 1
                self.result.gross_profit += pnl
            elif total_pnl < 0:
                self.result.losses += 1
                self.result.gross_loss += pnl
                self._last_loss_ts = bar_ts  # start cooldown
            else:
                self.result.breakevens += 1
            self.result.trades.append(closed.to_flat_dict())
        if sig.zone:
            self._last_zone_mid = sig.zone.mid
        self._last_trade_close_ts = bar_ts
        self._active_signal = None
        self._active_record = None
        self._partialled = False
        self._be_moved = False
        self._remaining_lot_ratio = 1.0
        self._partial_pnl = 0.0

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        log.info(f"Backtest starting: {len(self.df_5m)} 5m bars")
        t0 = time.time()

        equity = [self.balance]
        timestamps = [self.df_5m.index[0]]

        bar_iter = enumerate(self.df_5m.iterrows())
        if _HAS_TQDM:
            bar_iter = enumerate(
                _tqdm(self.df_5m.iterrows(), total=len(self.df_5m),
                      desc="Backtesting", unit="bar", dynamic_ncols=True)
            )

        for i, (ts, row) in bar_iter:
            if i < self.LTF_WARMUP:
                continue

            # ── Manage active trade ────────────────────────────────────────
            if self._active_signal is not None:
                sig = self._active_signal
                lo, hi = row["low"], row["high"]
                close = row["close"]

                # Track best price reached in trade's favour (MFE)
                if sig.direction == "buy":
                    self._mfe_best_price = max(self._mfe_best_price, hi)
                else:
                    self._mfe_best_price = min(self._mfe_best_price, lo)

                # Trailing SL — only after break-even is confirmed (protects profit,
                # avoids getting stopped out on normal pullbacks before TP1)
                if self._be_moved:
                    atr_series = calculate_atr(
                        self._get_slice(self.df_5m, ts, settings.ATR_TRAILING_BARS)
                    )
                    atr_val = float(atr_series.iloc[-1]) if len(atr_series) else sig.atr
                    new_sl = RiskManager.trailing_sl(self._current_sl, close, atr_val, sig.direction)
                    if new_sl != self._current_sl:
                        self._current_sl = new_sl
                        self.journal.add_action(sig.trade_id, "trailing_sl", new_sl, str(ts))

                # Partial TP — book PARTIAL_TP_PCT of position at TP1 immediately
                # Skip if lot is too small to split (e.g. 0.01 × 50% = 0.00 lots)
                if not self._partialled and RiskManager.check_partial_tp(
                    sig.entry_price, close, sig.tp1, sig.direction, self._partialled
                ):
                    self._partialled = True
                    pct = settings.PARTIAL_TP_PCT
                    partial_lot = round(sig.lot_size * pct, 2)
                    if partial_lot > 0.0:
                        if sig.direction == "buy":
                            partial_pips = (sig.tp1 - sig.entry_price) / settings.POINT_VALUE
                        else:
                            partial_pips = (sig.entry_price - sig.tp1) / settings.POINT_VALUE
                        partial_pnl = (
                            partial_pips * settings.POINT_VALUE * settings.CONTRACT_SIZE * partial_lot
                            - settings.BACKTEST_COMMISSION * partial_lot
                        )
                        self.risk.state.update_balance(partial_pnl)
                        self.result.gross_profit += partial_pnl   # partial TP is real profit — count it
                        self._partial_pnl += partial_pnl           # track for outcome determination
                        self._remaining_lot_ratio = 1.0 - pct
                        self.journal.add_action(
                            sig.trade_id, "partial_tp", sig.tp1,
                            f"{pct:.0%} closed @ {sig.tp1:.5f} P&L=${partial_pnl:+.2f}"
                        )
                    else:
                        # Lot too small to split — hold full position to TP2
                        self._remaining_lot_ratio = 1.0
                        self.journal.add_action(sig.trade_id, "partial_tp_skipped", sig.tp1, "lot too small to split")

                # Break-even — move SL to entry + optional buffer
                # Anchor to sig.tp1/tp2 (pre-slippage levels) so BE fires on the
                # exact same bar as partial TP when BREAKEVEN_TRIGGER_RR=1.0.
                # Using sig.entry_price (post-slippage) created a small gap that
                # caused partial TP to fire but BE to never trigger.
                if settings.BREAKEVEN_TRIGGER_RR <= 1.0:
                    _be_trigger = sig.tp1
                else:
                    _t = (settings.BREAKEVEN_TRIGGER_RR - 1.0) / max(settings.MIN_RR - 1.0, 1e-9)
                    _be_trigger = sig.tp1 + _t * (sig.tp2 - sig.tp1)
                if settings.USE_BREAKEVEN and not self._be_moved and RiskManager.check_breakeven(sig.entry_price, close, _be_trigger, sig.direction):
                    be_buffer = settings.BREAKEVEN_BUFFER_PIPS * settings.POINT_VALUE
                    if sig.direction == "buy":
                        new_be = sig.entry_price + be_buffer
                        self._current_sl = max(self._current_sl, new_be)
                    else:
                        new_be = sig.entry_price - be_buffer
                        self._current_sl = min(self._current_sl, new_be)
                    self._be_moved = True
                    self.journal.add_action(sig.trade_id, "break_even", self._current_sl, str(ts))

                # SL / TP2 exit — if both hit on the same bar, use distance from bar open
                # to decide which fired first (avoids always preferring SL, which
                # systematically understates win rate on volatile bars).
                sl_hit  = self._is_sl_hit(lo, hi)
                tp2_hit = self._is_tp_hit(lo, hi, sig.tp2, sig.direction)

                if sl_hit and tp2_hit:
                    bar_open = float(row["open"])
                    if sig.direction == "buy":
                        dist_sl  = bar_open - self._current_sl
                        dist_tp2 = sig.tp2 - bar_open
                    else:
                        dist_sl  = self._current_sl - bar_open
                        dist_tp2 = bar_open - sig.tp2
                    if dist_tp2 <= dist_sl:
                        self._close_active(sig.tp2, "tp2", ts)
                    else:
                        self._close_active(self._current_sl, "sl", ts)
                elif sl_hit:
                    self._close_active(self._current_sl, "sl", ts)
                elif tp2_hit:
                    self._close_active(sig.tp2, "tp2", ts)

            # ── Look for new signal ────────────────────────────────────────
            if self._active_signal is None:
                # Loss cooldown — prevent immediate re-entry into the same zone
                # after a losing trade (e.g., 4 consecutive SL hits on same zone)
                if settings.LOSS_COOLDOWN_BARS > 0 and self._last_loss_ts is not None:
                    bars_since_loss = (ts - self._last_loss_ts) / pd.Timedelta("5min")
                    if bars_since_loss < settings.LOSS_COOLDOWN_BARS:
                        self._rejections["loss_cooldown"] = self._rejections.get("loss_cooldown", 0) + 1
                        continue

                if settings.FILTER_RISK_MANAGEMENT:
                    allowed, reason = self.risk.can_trade(bar_date=ts.date())
                    if not allowed:
                        log.debug(f"Risk block: {reason}")
                        continue

                df_4h = self._get_slice(self.df_4h, ts, self.HTF_WARMUP)
                df_1h = self._get_slice(self.df_1h, ts, self.HTF_WARMUP)
                df_15m = self._get_slice(self.df_15m, ts, self.MTF_WARMUP)
                df_5m_w = self._get_slice(self.df_5m, ts, self.LTF_WARMUP)

                if len(df_4h) < 20 or len(df_1h) < 20 or len(df_15m) < 20 or len(df_5m_w) < 20:
                    continue

                try:
                    # Recompute HTF only when a new 1H bar has formed
                    current_1h_bar = df_1h.index[-1] if len(df_1h) > 0 else None
                    if current_1h_bar != self._last_1h_bar or self._htf_cache is None:
                        self._htf_cache = run_htf_bias(
                            df_4h, df_1h,
                            point_value=settings.POINT_VALUE,
                        )
                        self._last_1h_bar = current_1h_bar
                        self._mtf_cache = None  # invalidate MTF when HTF changes
                    htf = self._htf_cache

                    # Recompute MTF only when a new 15m bar has formed
                    current_15m_bar = df_15m.index[-1] if len(df_15m) > 0 else None
                    if current_15m_bar != self._last_15m_bar or self._mtf_cache is None:
                        self._mtf_cache = run_mtf_refinement(df_15m, htf)
                        self._last_15m_bar = current_15m_bar
                    mtf = self._mtf_cache

                    # Zone cooldown — block re-entry into recently traded zone.
                    # Prevents cascading losses when re-entering a mitigated zone
                    # (e.g. Feb 17: same zone traded 3 times → 2 consecutive losses).
                    # Applies after ANY trade close (win or loss).
                    if (
                        settings.ZONE_COOLDOWN_HOURS > 0
                        and self._last_zone_mid > 0
                        and self._last_trade_close_ts is not None
                        and mtf.best_zone is not None
                    ):
                        hours_since = (ts - self._last_trade_close_ts).total_seconds() / 3600
                        if hours_since < settings.ZONE_COOLDOWN_HOURS:
                            proximity = abs(mtf.best_zone.mid - self._last_zone_mid) / self._last_zone_mid
                            if proximity < 0.005:  # within 0.5% of price = same zone
                                self._rejections["zone_cooldown"] = self._rejections.get("zone_cooldown", 0) + 1
                                continue

                    _stats: dict = {}
                    signal = generate_signal(
                        df_5m_w,
                        mtf,
                        current_price=float(row["close"]),
                        spread=settings.BACKTEST_SPREAD,
                        account_balance=self.risk.state.account_balance,
                        stats=_stats,
                    )
                    # Track rejection reason
                    if signal is None and "rejected_by" in _stats:
                        key = _stats["rejected_by"]
                        self._rejections[key] = self._rejections.get(key, 0) + 1

                    # Apply entry slippage (worsens fill vs close price)
                    if signal is not None:
                        slip = settings.BACKTEST_SLIPPAGE
                        if signal.direction == "buy":
                            signal.entry_price += slip
                        else:
                            signal.entry_price -= slip
                except Exception as e:
                    import traceback
                    log.warning(f"Signal gen error at {ts}: {e}\n{traceback.format_exc()}")
                    self._rejections["exception"] = self._rejections.get("exception", 0) + 1
                    continue

                if signal is None:
                    continue

                score, breakdown = score_signal(signal)
                if settings.FILTER_SCORE and not should_execute(score):
                    self._rejections["score"] = self._rejections.get("score", 0) + 1
                    continue

                ai_ok, prob = self.ai.should_trade(signal, breakdown)
                if settings.FILTER_AI and not ai_ok:
                    log.debug(f"AI filter blocked: prob={prob:.3f}")
                    continue

                # Open trade
                self.risk.register_trade_open()
                self._active_signal = signal
                self._current_sl = signal.sl
                self._mfe_best_price = signal.entry_price  # initialise MFE tracker

                zone = signal.zone
                rec = TradeRecord(
                    trade_id=signal.trade_id,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    lot_size=signal.lot_size,
                    entry_price=signal.entry_price,
                    entry_time=str(ts),
                    entry_timeframe=settings.LTF_TIMEFRAME,
                    entry_reason=signal.reason,
                    session=signal.session,
                    sl=signal.sl,
                    tp1=signal.tp1,
                    tp2=signal.tp2,
                    planned_rr=signal.rr,
                    has_ob=zone.ob is not None if zone else False,
                    has_fvg=zone.fvg is not None if zone else False,
                    has_confluence=zone.has_confluence if zone else False,
                    zone_kind=zone.kind if zone else "",
                    sweep_size=signal.sweep_size,
                    score=signal.score,
                    ai_probability=signal.ai_probability,
                    atr=signal.atr,
                )
                self._active_record = rec
                self.journal.open_trade(rec)

            # ── Equity tracking ────────────────────────────────────────────
            equity.append(self.risk.state.account_balance)
            timestamps.append(ts)

        # Force-close any open trade at end
        if self._active_signal is not None:
            last_row = self.df_5m.iloc[-1]
            self._close_active(float(last_row["close"]), "end_of_data", self.df_5m.index[-1])

        self.result.equity_curve = equity
        self.result.timestamps = timestamps
        self.result.max_drawdown = self._calc_max_drawdown(equity)
        self.result.sharpe_ratio = self._calc_sharpe(equity)  # uses SHARPE_BARS_PER_DAY + RISK_FREE_RATE

        elapsed = time.time() - t0
        log.info(
            f"Backtest complete in {elapsed:.1f}s | "
            + " | ".join(f"{k}={v}" for k, v in self.result.summary().items())
        )
        if self._rejections:
            total_rej = sum(self._rejections.values())
            breakdown = " | ".join(
                f"{k}={v}" for k, v in sorted(self._rejections.items(), key=lambda x: -x[1])
            )
            log.info(f"Signal rejections ({total_rej} total): {breakdown}")
        return self.result

    @staticmethod
    def _calc_max_drawdown(equity: list[float]) -> float:
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1)
        return float(dd.max())

    @staticmethod
    def _calc_sharpe(equity: list[float]) -> float:
        if len(equity) < 2:
            return 0.0
        returns = np.diff(equity) / np.array(equity[:-1])
        if returns.std() == 0:
            return 0.0
        # Annualise: 252 trading days × bars per day (5m default = 78)
        annualisation = np.sqrt(252 * settings.SHARPE_BARS_PER_DAY)
        risk_free_per_bar = settings.RISK_FREE_RATE / (252 * settings.SHARPE_BARS_PER_DAY)
        return float((returns.mean() - risk_free_per_bar) / returns.std() * annualisation)
