"""
run_live.py — Live trading runner for SMC Bot (MetaTrader 5 / Exness).

Connects to MT5, fetches historical bars for warmup, then waits for each
5-minute bar to close and runs the full SMC strategy pipeline on it.
All trade management (partial TP, break-even, trailing SL) is applied
bar-by-bar. Orders are placed and modified via the MT5Adapter.

Usage:
    python run_live.py                     # connect with .env credentials
    python run_live.py --warmup 10000      # override warmup bar count
    python run_live.py --dry-run           # signal generation only, no orders

Requirements:
    pip install MetaTrader5   (Windows only — MT5 terminal must be installed)

Exness demo server  : Exness-MT5Trial
Exness live servers : Exness-MT5Real, Exness-MT5Real2, Exness-MT5Real3
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# Project root on path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import settings
from mt5.connector import MT5Adapter, OrderResult
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

log = get_logger("run_live")

# ── Constants ────────────────────────────────────────────────────────────────

_BAR_POLL_SECONDS = 10      # how often to check if a new 5m bar has formed
_MAX_ROLLING_BARS = 15_000  # cap on in-memory 5m bars to avoid memory growth
_RECONNECT_RETRIES = 4


# ── LiveRunner ───────────────────────────────────────────────────────────────

class LiveRunner:
    """
    Mirrors BacktestEngine logic bar-by-bar, but routes order execution
    through MT5Adapter instead of simulating fills in-memory.

    State is kept in RAM; the MT5 terminal is the authoritative source
    for whether a position is open, and the journal/CSV track P&L history.
    """

    # Same warmup thresholds as BacktestEngine
    HTF_WARMUP = settings.HTF_WARMUP_BARS
    MTF_WARMUP = settings.MTF_WARMUP_BARS
    LTF_WARMUP = settings.LTF_WARMUP_BARS

    def __init__(self, adapter: MT5Adapter, dry_run: bool = False) -> None:
        self.adapter = adapter
        self.dry_run = dry_run

        # Rolling 5m DataFrame (most recent bars at tail)
        self._rolling_df: Optional[pd.DataFrame] = None

        # Active trade state — mirrors BacktestEngine fields
        self._active_signal: Optional[TradeSignal] = None
        self._active_ticket: Optional[str] = None      # MT5 position ticket (str)
        self._active_record: Optional[TradeRecord] = None
        self._partialled: bool = False
        self._be_moved: bool = False
        self._current_sl: float = 0.0
        self._remaining_lot_ratio: float = 1.0
        self._mfe_best_price: float = 0.0

        # Cooldown / zone tracking
        self._last_loss_ts: Optional[pd.Timestamp] = None
        self._last_zone_mid: float = 0.0
        self._last_trade_close_ts: Optional[pd.Timestamp] = None

        # HTF/MTF caches — recomputed only when a higher-TF bar forms
        self._htf_cache: Optional[HTFAnalysis] = None
        self._mtf_cache: Optional[MTFAnalysis] = None
        self._last_1h_bar: Optional[pd.Timestamp] = None
        self._last_15m_bar: Optional[pd.Timestamp] = None

        # Risk & journal
        acct = adapter.get_account_info()
        balance = acct.get("balance", settings.ACCOUNT_BALANCE)
        self.risk = RiskManager(RiskState(account_balance=balance))
        self.journal = TradeJournal()
        self.ai = ProbabilityModel()

        self._rejections: dict[str, int] = {}
        self._running = True  # set False on SIGINT

    # ── Warmup ─────────────────────────────────────────────────────────────

    def warmup(self, n_bars: int = None) -> bool:
        """Fetch historical 5m bars from MT5 to prime the strategy pipeline."""
        n = n_bars or settings.MT5_WARMUP_BARS
        log.info(f"Fetching {n} historical 5m bars for warmup…")

        try:
            import MetaTrader5 as mt5
            tf = mt5.TIMEFRAME_M5
        except ImportError:
            log.error("MetaTrader5 not installed")
            return False

        df = self.adapter.fetch_ohlcv(settings.MT5_SYMBOL, tf, n)
        if df.empty:
            log.error("Warmup failed — no data returned from MT5")
            return False

        self._rolling_df = df
        log.info(
            f"Warmup complete: {len(df)} bars | "
            f"{df.index[0]} → {df.index[-1]}"
        )
        return True

    # ── Bar helpers ────────────────────────────────────────────────────────

    def _get_slice(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[-n:] if len(df) >= n else df

    def _append_bar(self, row: pd.Series, ts: pd.Timestamp) -> None:
        """Append a new closed bar to the rolling buffer, trimming if needed."""
        new_row = pd.DataFrame([row], index=[ts])
        new_row.index.name = "time"
        self._rolling_df = pd.concat([self._rolling_df, new_row])
        if len(self._rolling_df) > _MAX_ROLLING_BARS:
            self._rolling_df = self._rolling_df.iloc[-_MAX_ROLLING_BARS:]

    # ── Trade management ───────────────────────────────────────────────────

    def _handle_partial_tp(self, ts: pd.Timestamp, close: float) -> None:
        sig = self._active_signal
        if self._partialled:
            return
        if not RiskManager.check_partial_tp(
            sig.entry_price, close, sig.tp1, sig.direction, self._partialled
        ):
            return

        pct = settings.PARTIAL_TP_PCT
        partial_lot = round(sig.lot_size * pct, 2)
        self._partialled = True

        if partial_lot < 0.01:
            # Lot too small to split — hold full position
            self._remaining_lot_ratio = 1.0
            self.journal.add_action(
                sig.trade_id, "partial_tp_skipped", sig.tp1, "lot too small"
            )
            log.info(f"Partial TP skipped (lot too small): {sig.trade_id[:8]}")
            return

        if not self.dry_run:
            result = self.adapter.close_order(self._active_ticket, partial_pct=pct)
        else:
            result = OrderResult(success=True, fill_price=sig.tp1)

        if not result.success:
            log.error(f"Partial TP close failed: {result.error}")
            return

        fill = result.fill_price or sig.tp1
        if sig.direction == "buy":
            partial_pips = (fill - sig.entry_price) / settings.POINT_VALUE
        else:
            partial_pips = (sig.entry_price - fill) / settings.POINT_VALUE
        partial_pnl = round(
            partial_pips * settings.POINT_VALUE * settings.CONTRACT_SIZE * partial_lot, 2
        )
        self.risk.state.update_balance(partial_pnl)
        self._remaining_lot_ratio = 1.0 - pct
        self.journal.add_action(
            sig.trade_id, "partial_tp", fill,
            f"{pct:.0%} closed @ {fill:.5f} P&L=${partial_pnl:+.2f}"
        )
        log.info(
            f"Partial TP: {sig.trade_id[:8]} | {pct:.0%} closed @ {fill:.5f} "
            f"| P&L ${partial_pnl:+.2f}"
        )

    def _handle_breakeven(self, ts: pd.Timestamp, close: float) -> None:
        sig = self._active_signal
        if self._be_moved or not settings.USE_BREAKEVEN:
            return

        # Same anchor logic as BacktestEngine — avoids slippage gap bug
        if settings.BREAKEVEN_TRIGGER_RR <= 1.0:
            _be_trigger = sig.tp1
        else:
            _t = (settings.BREAKEVEN_TRIGGER_RR - 1.0) / max(settings.MIN_RR - 1.0, 1e-9)
            _be_trigger = sig.tp1 + _t * (sig.tp2 - sig.tp1)

        if not RiskManager.check_breakeven(sig.entry_price, close, _be_trigger, sig.direction):
            return

        be_buffer = settings.BREAKEVEN_BUFFER_PIPS * settings.POINT_VALUE
        if sig.direction == "buy":
            new_be = sig.entry_price + be_buffer
            new_be = max(self._current_sl, new_be)
        else:
            new_be = sig.entry_price - be_buffer
            new_be = min(self._current_sl, new_be)

        ok = True
        if not self.dry_run:
            ok = self.adapter.modify_sl(self._active_ticket, new_be)

        if ok:
            self._current_sl = new_be
            self._be_moved = True
            self.journal.add_action(sig.trade_id, "break_even", new_be, str(ts))
            log.info(f"Break-even set: {sig.trade_id[:8]} | new SL={new_be:.5f}")

    def _handle_trailing_sl(self, ts: pd.Timestamp, close: float) -> None:
        if not self._be_moved:
            return
        sig = self._active_signal
        df_slice = self._get_slice(self._rolling_df, settings.ATR_TRAILING_BARS)
        atr_series = calculate_atr(df_slice)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else sig.atr
        new_sl = RiskManager.trailing_sl(self._current_sl, close, atr_val, sig.direction)
        if new_sl == self._current_sl:
            return

        ok = True
        if not self.dry_run:
            ok = self.adapter.modify_sl(self._active_ticket, new_sl)

        if ok:
            self._current_sl = new_sl
            self.journal.add_action(sig.trade_id, "trailing_sl", new_sl, str(ts))
            log.debug(f"Trailing SL: {sig.trade_id[:8]} → {new_sl:.5f}")

    def _close_trade(
        self, close_price: float, close_reason: str, ts: pd.Timestamp
    ) -> None:
        """Update journal and risk state when a trade closes."""
        sig = self._active_signal

        # MFE in pips
        if sig.direction == "buy":
            mfe_pips = max(0.0, (self._mfe_best_price - sig.entry_price) / settings.POINT_VALUE)
        else:
            mfe_pips = max(0.0, (sig.entry_price - self._mfe_best_price) / settings.POINT_VALUE)

        closed = self.journal.close_trade(
            trade_id=sig.trade_id,
            exit_price=close_price,
            exit_reason=close_reason,
            account_balance=self.risk.state.account_balance,
            exit_time=str(ts),
            max_favorable_excursion=mfe_pips,
        )
        if closed:
            remaining_lot = closed.lot_size * self._remaining_lot_ratio
            if self._remaining_lot_ratio != 1.0:
                closed.pnl_currency = round(
                    closed.pnl_pips * settings.POINT_VALUE * settings.CONTRACT_SIZE * remaining_lot, 2
                )
            pnl = closed.pnl_currency
            self.risk.register_trade_close(pnl)
            closed.account_balance_after = round(self.risk.state.account_balance, 2)

            outcome_str = f"{'WIN' if closed.outcome == 'win' else ('LOSS' if closed.outcome == 'loss' else 'BE')}"
            log.info(
                f"Trade closed [{outcome_str}]: {sig.trade_id[:8]} | "
                f"{sig.direction.upper()} @ {sig.entry_price:.5f} → {close_price:.5f} | "
                f"reason={close_reason} | P&L ${pnl:+.2f} | "
                f"balance={self.risk.state.account_balance:.2f}"
            )

            if closed.outcome == "loss":
                self._last_loss_ts = ts

        if sig.zone:
            self._last_zone_mid = sig.zone.mid
        self._last_trade_close_ts = ts

        # Reset active trade state
        self._active_signal = None
        self._active_ticket = None
        self._active_record = None
        self._partialled = False
        self._be_moved = False
        self._remaining_lot_ratio = 1.0
        self._mfe_best_price = 0.0

    # ── Per-bar step ────────────────────────────────────────────────────────

    def _step(self, ts: pd.Timestamp, row: pd.Series) -> None:
        lo = float(row["low"])
        hi = float(row["high"])
        close = float(row["close"])

        # ── Manage active trade ──────────────────────────────────────────
        if self._active_signal is not None:
            sig = self._active_signal

            # Sync position state from MT5
            pos = self.adapter.get_open_position(settings.MT5_SYMBOL) if not self.dry_run else {"ticket": self._active_ticket}

            if pos is None:
                # Position closed by MT5 (SL or TP2 hit natively)
                ticket_int = int(self._active_ticket)
                close_price, close_reason = self.adapter.get_position_close_info(
                    ticket_int, sig.timestamp
                )
                if close_price == 0.0:
                    # Fallback: estimate from our tracked SL/TP2
                    if sig.direction == "buy":
                        close_price = sig.tp2 if hi >= sig.tp2 else self._current_sl
                        close_reason = "tp2" if hi >= sig.tp2 else "sl"
                    else:
                        close_price = sig.tp2 if lo <= sig.tp2 else self._current_sl
                        close_reason = "tp2" if lo <= sig.tp2 else "sl"
                self._close_trade(close_price, close_reason, ts)
            else:
                # Position open — apply management

                # Update MFE
                if sig.direction == "buy":
                    self._mfe_best_price = max(self._mfe_best_price, hi)
                else:
                    self._mfe_best_price = min(self._mfe_best_price, lo)

                # Order: trailing → partial TP → break-even
                # (Trailing only after BE; partial TP and BE can fire same bar)
                self._handle_trailing_sl(ts, close)
                self._handle_partial_tp(ts, close)
                self._handle_breakeven(ts, close)

        # ── Look for new signal ──────────────────────────────────────────
        if self._active_signal is not None:
            return  # still in a trade

        # Loss cooldown
        if settings.LOSS_COOLDOWN_BARS > 0 and self._last_loss_ts is not None:
            bars_since = (ts - self._last_loss_ts) / pd.Timedelta("5min")
            if bars_since < settings.LOSS_COOLDOWN_BARS:
                self._rejections["loss_cooldown"] = self._rejections.get("loss_cooldown", 0) + 1
                return

        # Risk management gate
        if settings.FILTER_RISK_MANAGEMENT:
            allowed, reason = self.risk.can_trade(bar_date=ts.date())
            if not allowed:
                log.debug(f"Risk block: {reason}")
                return

        # Build sliced DataFrames
        df_4h = resample_ohlcv(self._rolling_df, "4H")
        df_1h = resample_ohlcv(self._rolling_df, "1H")
        df_15m = resample_ohlcv(self._rolling_df, "15m")
        df_5m_w = self._get_slice(self._rolling_df, self.LTF_WARMUP)

        df_4h = df_4h.iloc[-self.HTF_WARMUP:] if len(df_4h) > self.HTF_WARMUP else df_4h
        df_1h = df_1h.iloc[-self.HTF_WARMUP:] if len(df_1h) > self.HTF_WARMUP else df_1h
        df_15m = df_15m.iloc[-self.MTF_WARMUP:] if len(df_15m) > self.MTF_WARMUP else df_15m

        if len(df_4h) < 20 or len(df_1h) < 20 or len(df_15m) < 20 or len(df_5m_w) < 20:
            return

        try:
            # HTF cache — recompute only on new 1H bar
            current_1h_bar = df_1h.index[-1]
            if current_1h_bar != self._last_1h_bar or self._htf_cache is None:
                self._htf_cache = run_htf_bias(df_4h, df_1h, point_value=settings.POINT_VALUE)
                self._last_1h_bar = current_1h_bar
                self._mtf_cache = None  # invalidate when HTF changes

            # MTF cache — recompute only on new 15m bar
            current_15m_bar = df_15m.index[-1]
            if current_15m_bar != self._last_15m_bar or self._mtf_cache is None:
                self._mtf_cache = run_mtf_refinement(df_15m, self._htf_cache)
                self._last_15m_bar = current_15m_bar

            mtf = self._mtf_cache

            # Zone cooldown
            if (
                settings.ZONE_COOLDOWN_HOURS > 0
                and self._last_zone_mid > 0
                and self._last_trade_close_ts is not None
                and mtf.best_zone is not None
            ):
                hours_since = (ts - self._last_trade_close_ts).total_seconds() / 3600
                if hours_since < settings.ZONE_COOLDOWN_HOURS:
                    proximity = abs(mtf.best_zone.mid - self._last_zone_mid) / self._last_zone_mid
                    if proximity < 0.005:
                        self._rejections["zone_cooldown"] = self._rejections.get("zone_cooldown", 0) + 1
                        return

            # Live spread from MT5
            spread = self.adapter.get_spread(settings.MT5_SYMBOL)

            _stats: dict = {}
            signal = generate_signal(
                df_5m_w,
                mtf,
                current_price=close,
                spread=spread,
                account_balance=self.risk.state.account_balance,
                stats=_stats,
            )

            if signal is None:
                if "rejected_by" in _stats:
                    key = _stats["rejected_by"]
                    self._rejections[key] = self._rejections.get(key, 0) + 1
                return

        except Exception:
            log.warning(f"Signal gen error at {ts}:\n{traceback.format_exc()}")
            self._rejections["exception"] = self._rejections.get("exception", 0) + 1
            return

        # Score filter
        score, breakdown = score_signal(signal)
        if settings.FILTER_SCORE and not should_execute(score):
            self._rejections["score"] = self._rejections.get("score", 0) + 1
            return

        # AI filter
        ai_ok, prob = self.ai.should_trade(signal, breakdown)
        if settings.FILTER_AI and not ai_ok:
            log.debug(f"AI filter blocked: prob={prob:.3f}")
            self._rejections["ai"] = self._rejections.get("ai", 0) + 1
            return

        # ── Open trade ───────────────────────────────────────────────────
        log.info(
            f"SIGNAL: {signal.direction.upper()} {settings.MT5_SYMBOL} "
            f"| entry={signal.entry_price:.5f} SL={signal.sl:.5f} "
            f"TP1={signal.tp1:.5f} TP2={signal.tp2:.5f} "
            f"| lot={signal.lot_size} score={score:.0f} "
            f"| {signal.reason}"
        )

        if not self.dry_run:
            result = self.adapter.place_order(
                symbol=settings.MT5_SYMBOL,
                direction=signal.direction,
                lot_size=signal.lot_size,
                sl=signal.sl,
                tp=signal.tp2,  # MT5 holds TP2; we manage partial TP manually
                comment=f"smc_{signal.trade_id[:8]}",
            )
            if not result.success:
                log.error(f"Order rejected by MT5: {result.error}")
                return
            fill_price = result.fill_price or signal.entry_price
            self._active_ticket = result.order_id
        else:
            fill_price = signal.entry_price
            self._active_ticket = "DRY_RUN"
            log.info("[DRY RUN] Order not placed")

        # Apply slippage to signal entry for P&L tracking consistency
        signal.entry_price = fill_price

        self.risk.register_trade_open()
        self._active_signal = signal
        self._current_sl = signal.sl
        self._mfe_best_price = signal.entry_price

        zone = signal.zone
        rec = TradeRecord(
            trade_id=signal.trade_id,
            symbol=settings.MT5_SYMBOL,
            direction=signal.direction,
            lot_size=signal.lot_size,
            entry_price=fill_price,
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

    # ── Main loop ──────────────────────────────────────────────────────────

    def _wait_for_new_bar(self) -> tuple[pd.Timestamp, pd.Series]:
        """
        Poll MT5 every _BAR_POLL_SECONDS until a new 5m bar appears.
        Returns (bar_timestamp, OHLCV_series) of the just-closed bar.
        """
        import MetaTrader5 as mt5

        last_ts = self._rolling_df.index[-1] if self._rolling_df is not None else None

        while self._running:
            try:
                rates = mt5.copy_rates_from_pos(
                    settings.MT5_SYMBOL, mt5.TIMEFRAME_M5, 0, 3
                )
                if rates is None or len(rates) < 2:
                    time.sleep(_BAR_POLL_SECONDS)
                    continue

                df_new = pd.DataFrame(rates)
                df_new["time"] = pd.to_datetime(df_new["time"], unit="s", utc=True)
                df_new = df_new.set_index("time")
                df_new = df_new.rename(columns={"tick_volume": "volume"})
                df_new = df_new[["open", "high", "low", "close", "volume"]].astype(float)

                # The second-to-last bar is the most recently closed one
                closed_ts = df_new.index[-2]

                if last_ts is None or closed_ts > last_ts:
                    return closed_ts, df_new.iloc[-2]

            except Exception:
                log.warning(f"Bar poll error:\n{traceback.format_exc()}")

            time.sleep(_BAR_POLL_SECONDS)

        raise KeyboardInterrupt  # _running was set False

    def run(self, warmup_bars: int = None) -> None:
        """Main live trading loop — runs until SIGINT."""
        log.info("=" * 60)
        log.info("  SMC BOT — LIVE TRADING")
        log.info(f"  Symbol : {settings.MT5_SYMBOL}")
        log.info(f"  Mode   : {'DRY RUN (no orders)' if self.dry_run else 'LIVE'}")
        log.info(f"  Magic  : {settings.MT5_MAGIC}")
        log.info("=" * 60)

        if not self.warmup(warmup_bars):
            log.error("Warmup failed — cannot start live trading")
            return

        log.info("Waiting for next 5m bar close…")

        while self._running:
            try:
                # Reconnect if terminal dropped
                if not self.adapter.is_connected():
                    log.warning("MT5 connection lost — reconnecting…")
                    if not self.adapter.reconnect(_RECONNECT_RETRIES):
                        log.critical("Reconnect failed — stopping")
                        break
                    # Re-sync balance after reconnect
                    acct = self.adapter.get_account_info()
                    if acct:
                        self.risk.state.account_balance = acct["balance"]

                ts, row = self._wait_for_new_bar()
                log.debug(
                    f"Bar: {ts} | O={row['open']:.3f} H={row['high']:.3f} "
                    f"L={row['low']:.3f} C={row['close']:.3f}"
                )

                self._append_bar(row, ts)
                self._step(ts, row)

            except KeyboardInterrupt:
                break
            except Exception:
                log.error(f"Unhandled error in main loop:\n{traceback.format_exc()}")
                time.sleep(5)  # brief pause before retrying

        self._shutdown()

    def _shutdown(self) -> None:
        log.info("Shutting down…")
        if self._active_signal is not None and not self.dry_run:
            log.warning(
                f"Active trade {self._active_signal.trade_id[:8]} left open — "
                "MT5 SL/TP will manage it."
            )
        if self._rejections:
            total = sum(self._rejections.values())
            breakdown = " | ".join(
                f"{k}={v}" for k, v in sorted(self._rejections.items(), key=lambda x: -x[1])
            )
            log.info(f"Signal rejections ({total} total): {breakdown}")

        self.adapter.disconnect()
        log.info("Live runner stopped.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SMC Bot Live Trading (MT5 / Exness)")
    parser.add_argument(
        "--warmup", type=int, default=None,
        help=f"Historical 5m bars to fetch for warmup (default: {settings.MT5_WARMUP_BARS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run signal generation without placing real orders",
    )
    args = parser.parse_args()

    adapter = MT5Adapter()
    if not adapter.connect():
        log.critical("Failed to connect to MT5 — check .env credentials")
        sys.exit(1)

    runner = LiveRunner(adapter=adapter, dry_run=args.dry_run)

    # Graceful shutdown on CTRL+C or SIGTERM
    def _stop(sig, frame):
        log.info("Shutdown signal received")
        runner._running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    runner.run(warmup_bars=args.warmup)


if __name__ == "__main__":
    main()
