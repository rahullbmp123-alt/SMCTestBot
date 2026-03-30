"""
Trade Journal — logs every trade lifecycle event to CSV and JSON.

Logs:
  - Entry details (price, time, reason, RR)
  - Risk parameters (SL, TP, lots)
  - Trade management actions (break-even, partial TP, trailing SL)
  - Exit (price, time, reason)
  - Performance (P&L, RR achieved, duration)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_TRADES_CSV = settings.LOGS_DIR / "trades.csv"
_TRADES_JSON = settings.LOGS_DIR / "trades.json"

# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    # Identity
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    direction: str = ""     # "buy" / "sell"
    lot_size: float = 0.0

    # Entry
    entry_price: float = 0.0
    entry_time: str = ""
    entry_timeframe: str = ""
    entry_reason: str = ""
    session: str = ""

    # Risk
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    planned_rr: float = 0.0

    # Zone context
    has_ob: bool = False
    has_fvg: bool = False
    has_confluence: bool = False
    zone_kind: str = ""
    sweep_size: float = 0.0

    # Scoring
    score: float = 0.0
    ai_probability: float = 0.0

    # Management actions (list stored as JSON string)
    _actions: list = field(default_factory=list)

    # Exit
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""    # "tp1", "tp2", "sl", "trailing_sl", "manual"

    # Performance
    pnl_pips: float = 0.0
    pnl_currency: float = 0.0
    rr_achieved: float = 0.0
    duration_minutes: float = 0.0
    outcome: str = ""        # "win" / "loss" / "breakeven"

    # Max favorable excursion — how far price moved in trade's favour before close
    max_favorable_excursion: float = 0.0   # in pips (positive = moved toward TP)

    # Account balance after this trade closed
    account_balance_after: float = 0.0

    # ATR at entry
    atr: float = 0.0

    def add_action(self, action_type: str, price: float, time: str, note: str = "") -> None:
        self._actions.append({
            "type": action_type,
            "price": price,
            "time": time,
            "note": note,
        })

    def to_flat_dict(self) -> dict:
        d = asdict(self)
        d["actions"] = json.dumps(self._actions)
        d.pop("_actions", None)
        return d


# ── Journal class ─────────────────────────────────────────────────────────────

class TradeJournal:
    def __init__(self) -> None:
        self._records: list[TradeRecord] = []
        self._load_existing()

    # ── Persistence ────────────────────────────────────────────────────────
    def _load_existing(self) -> None:
        if _TRADES_CSV.exists():
            try:
                df = pd.read_csv(_TRADES_CSV)
                log.info(f"Journal: loaded {len(df)} existing trades from CSV")
            except Exception as e:
                log.warning(f"Journal: could not load existing trades: {e}")

    def _append_csv(self, record: TradeRecord) -> None:
        flat = record.to_flat_dict()
        df = pd.DataFrame([flat])
        header = not _TRADES_CSV.exists()
        df.to_csv(_TRADES_CSV, mode="a", index=False, header=header)

    def _append_json(self, record: TradeRecord) -> None:
        all_trades = []
        if _TRADES_JSON.exists():
            try:
                with open(_TRADES_JSON) as f:
                    all_trades = json.load(f)
            except Exception:
                pass
        all_trades.append(record.to_flat_dict())
        with open(_TRADES_JSON, "w") as f:
            json.dump(all_trades, f, indent=2, default=str)

    # ── Public API ─────────────────────────────────────────────────────────
    def open_trade(self, record: TradeRecord) -> TradeRecord:
        self._records.append(record)
        log.info(
            f"TRADE OPENED [{record.trade_id}] "
            f"{record.direction.upper()} {record.symbol} "
            f"@ {record.entry_price:.5f} | SL={record.sl:.5f} TP={record.tp2:.5f} "
            f"score={record.score:.1f} AI={record.ai_probability:.2f}"
        )
        return record

    def add_action(
        self,
        trade_id: str,
        action_type: str,
        price: float,
        note: str = "",
    ) -> None:
        rec = self._find(trade_id)
        if rec:
            rec.add_action(action_type, price, datetime.utcnow().isoformat(), note)
            log.info(f"Trade [{trade_id}] action: {action_type} @ {price:.5f} {note}")

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        account_balance: float,
        exit_time: str = None,
        max_favorable_excursion: float = 0.0,
    ) -> Optional[TradeRecord]:
        rec = self._find(trade_id)
        if rec is None:
            log.warning(f"Trade {trade_id} not found in journal")
            return None

        rec.exit_price = exit_price
        rec.exit_time = exit_time if exit_time is not None else datetime.utcnow().isoformat()
        rec.exit_reason = exit_reason
        rec.max_favorable_excursion = round(max_favorable_excursion, 1)
        rec.account_balance_after = round(account_balance, 2)

        # Calculate P&L
        if rec.direction == "buy":
            pips = (exit_price - rec.entry_price) / settings.POINT_VALUE
        else:
            pips = (rec.entry_price - exit_price) / settings.POINT_VALUE

        rec.pnl_pips = round(pips, 1)
        rec.pnl_currency = round(
            pips * settings.POINT_VALUE * settings.CONTRACT_SIZE * rec.lot_size, 2
        )

        # RR achieved
        risk_pips = abs(rec.entry_price - rec.sl) / settings.POINT_VALUE
        rec.rr_achieved = round(pips / risk_pips, 2) if risk_pips > 0 else 0.0

        # Duration
        try:
            entry_dt = datetime.fromisoformat(rec.entry_time)
            exit_dt = datetime.fromisoformat(rec.exit_time)
            rec.duration_minutes = round((exit_dt - entry_dt).total_seconds() / 60, 1)
        except Exception:
            pass

        # Outcome
        if rec.pnl_pips > 0:
            rec.outcome = "win"
        elif rec.pnl_pips < 0:
            rec.outcome = "loss"
        else:
            rec.outcome = "breakeven"

        self._append_csv(rec)
        self._append_json(rec)

        log.info(
            f"TRADE CLOSED [{trade_id}] outcome={rec.outcome} "
            f"pips={rec.pnl_pips:+.1f} P&L=${rec.pnl_currency:+.2f} "
            f"RR={rec.rr_achieved:.2f} duration={rec.duration_minutes}min"
        )
        return rec

    def load_all(self) -> pd.DataFrame:
        """Load all completed trades from CSV."""
        if not _TRADES_CSV.exists():
            return pd.DataFrame()
        return pd.read_csv(_TRADES_CSV)

    def _find(self, trade_id: str) -> Optional[TradeRecord]:
        for r in self._records:
            if r.trade_id == trade_id:
                return r
        return None

    @property
    def open_trades(self) -> list[TradeRecord]:
        return [r for r in self._records if not r.exit_price]
