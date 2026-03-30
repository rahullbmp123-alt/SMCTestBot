"""
MT5 Integration Module — Execution adapters for backtest and live trading.

BacktestAdapter  : no-op paper trading (used by the backtest engine).
MT5Adapter       : full MetaTrader 5 implementation for live/demo trading.

Tested against Exness MT5 demo (Exness-MT5Trial) and real (Exness-MT5Real*).

Requirements (Windows only — MT5 terminal must be installed):
    pip install MetaTrader5

Linux users need Wine + the MT5 Windows terminal or a Windows VM.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    error: Optional[str] = None


class ExecutionAdapter(ABC):
    """Abstract base — all execution adapters implement this interface."""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> OrderResult: ...

    @abstractmethod
    def modify_sl(self, order_id: str, new_sl: float) -> bool: ...

    @abstractmethod
    def close_order(self, order_id: str, partial_pct: float = 1.0) -> OrderResult: ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> tuple[float, float]: ...  # (bid, ask)

    @abstractmethod
    def get_spread(self, symbol: str) -> float: ...

    @abstractmethod
    def get_account_info(self) -> dict: ...


# ── BacktestAdapter ─────────────────────────────────────────────────────────

class BacktestAdapter(ExecutionAdapter):
    """
    Paper trading adapter — used during backtesting.
    All methods are no-ops that return success.
    """

    def connect(self) -> bool:
        log.info("BacktestAdapter: connected (paper mode)")
        return True

    def disconnect(self) -> None:
        log.info("BacktestAdapter: disconnected")

    def place_order(self, symbol, direction, lot_size, sl, tp, comment="") -> OrderResult:
        log.debug(f"[PAPER] place_order {direction} {lot_size} {symbol}")
        return OrderResult(success=True, order_id="PAPER_001", fill_price=None)

    def modify_sl(self, order_id, new_sl) -> bool:
        log.debug(f"[PAPER] modify_sl {order_id} → {new_sl:.5f}")
        return True

    def close_order(self, order_id, partial_pct=1.0) -> OrderResult:
        log.debug(f"[PAPER] close_order {order_id} ({partial_pct:.0%})")
        return OrderResult(success=True, fill_price=None)

    def get_current_price(self, symbol) -> tuple[float, float]:
        return 0.0, 0.0

    def get_spread(self, symbol) -> float:
        return 0.0001

    def get_account_info(self) -> dict:
        return {"balance": 10000.0, "equity": 10000.0, "margin_free": 9000.0}


# ── MT5Adapter ──────────────────────────────────────────────────────────────

class MT5Adapter(ExecutionAdapter):
    """
    MetaTrader 5 live execution adapter — Exness-compatible.

    Exness demo server  : Exness-MT5Trial
    Exness live servers : Exness-MT5Real, Exness-MT5Real2, Exness-MT5Real3 …

    Order filling strategy:
        Tries ORDER_FILLING_IOC first (works on most Exness account types).
        Falls back to ORDER_FILLING_FOK automatically on retcode 10030.

    Partial close:
        Sends a counter-direction market order for the fractional volume.
        MT5 merges it against the original ticket and reduces its volume.
    """

    # Max price deviation (points) — prevents requote rejection on fast markets
    _DEVIATION = 30

    def __init__(self, magic: int = None) -> None:
        from config import settings
        self._magic: int = magic if magic is not None else settings.MT5_MAGIC
        self._connected: bool = False
        self._mt5 = None  # MetaTrader5 module, imported lazily in connect()

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            log.error(
                "MetaTrader5 package not installed.\n"
                "Install with: pip install MetaTrader5\n"
                "Note: this package only works on Windows."
            )
            return False

        self._mt5 = mt5
        from config import settings

        # Initialize terminal
        path = settings.MT5_TERMINAL_PATH or None
        if not mt5.initialize(path=path):
            log.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

        # Login (skip if credentials not provided — uses already-open terminal session)
        if settings.MT5_LOGIN and settings.MT5_PASSWORD:
            ok = mt5.login(
                int(settings.MT5_LOGIN),
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER or None,
            )
            if not ok:
                log.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        info = mt5.account_info()
        if info is None:
            log.error("MT5 account_info() returned None — check credentials / terminal")
            mt5.shutdown()
            return False

        trade_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
        account_type = "DEMO" if info.trade_mode == trade_mode else "LIVE"
        self._connected = True
        log.info(
            f"MT5 connected | account={info.login} | server={info.server} | "
            f"balance={info.balance:.2f} {info.currency} | {account_type}"
        )
        return True

    def disconnect(self) -> None:
        if self._connected and self._mt5 is not None:
            self._mt5.shutdown()
            self._connected = False
            log.info("MT5 disconnected")

    def is_connected(self) -> bool:
        """Ping MT5 terminal to verify connection is still alive."""
        if not self._connected or self._mt5 is None:
            return False
        return self._mt5.terminal_info() is not None

    # ── Order placement ────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> OrderResult:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, error=f"No tick data for {symbol}")

        # Ensure the symbol is visible in Market Watch
        if not mt5.symbol_select(symbol, True):
            log.warning(f"symbol_select({symbol}) failed — proceeding anyway")

        if direction == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(float(lot_size), 2),
            "type": order_type,
            "price": price,
            "sl": round(float(sl), 5),
            "tp": round(float(tp), 5),
            "deviation": self._DEVIATION,
            "magic": self._magic,
            "comment": comment[:31],  # MT5 limit
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        # Exness sometimes requires FOK instead of IOC — retry once
        if result is not None and result.retcode == 10030:
            request["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment_r = result.comment if result else ""
            err = f"place_order failed: retcode={retcode} comment={comment_r}"
            log.error(err)
            return OrderResult(success=False, error=err)

        log.info(
            f"Order placed: {direction.upper()} {lot_size} {symbol} "
            f"@ {result.price:.5f} | SL={sl:.5f} TP={tp:.5f} | ticket={result.order}"
        )
        return OrderResult(
            success=True,
            order_id=str(result.order),
            fill_price=float(result.price),
        )

    # ── Modify SL/TP ──────────────────────────────────────────────────────

    def modify_sl(self, order_id: str, new_sl: float) -> bool:
        """Modify SL on an open position. Keeps existing TP unchanged."""
        mt5 = self._mt5
        ticket = int(order_id)

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            log.warning(f"modify_sl: position {ticket} not found")
            return False
        pos = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": round(float(new_sl), 5),
            "tp": pos.tp,  # keep existing TP
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            log.warning(f"modify_sl failed: ticket={ticket} retcode={retcode}")
            return False

        log.debug(f"SL modified: ticket={ticket} new_sl={new_sl:.5f}")
        return True

    # ── Close / partial close ──────────────────────────────────────────────

    def close_order(self, order_id: str, partial_pct: float = 1.0) -> OrderResult:
        """
        Close all or part of a position.

        partial_pct=1.0  → full close
        partial_pct=0.5  → close half (partial TP)
        """
        mt5 = self._mt5
        ticket = int(order_id)

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            # Already closed — not an error (may have been stopped by MT5)
            log.info(f"close_order: position {ticket} already closed")
            return OrderResult(success=True, fill_price=None)
        pos = positions[0]

        volume = round(pos.volume * partial_pct, 2)
        volume = max(volume, 0.01)  # MT5 minimum lot

        tick = mt5.symbol_info_tick(pos.symbol)
        if pos.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self._DEVIATION,
            "magic": self._magic,
            "comment": "partial_tp" if partial_pct < 1.0 else "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is not None and result.retcode == 10030:
            request["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment_r = result.comment if result else ""
            err = f"close_order failed: ticket={ticket} retcode={retcode} {comment_r}"
            log.error(err)
            return OrderResult(success=False, error=err)

        log.info(
            f"Position {'partial ' if partial_pct < 1.0 else ''}closed: "
            f"ticket={ticket} volume={volume} @ {result.price:.5f}"
        )
        return OrderResult(success=True, fill_price=float(result.price))

    # ── Price & account ────────────────────────────────────────────────────

    def get_current_price(self, symbol: str) -> tuple[float, float]:
        """Returns (bid, ask)."""
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0.0, 0.0
        return float(tick.bid), float(tick.ask)

    def get_spread(self, symbol: str) -> float:
        bid, ask = self.get_current_price(symbol)
        return round(ask - bid, 5)

    def get_account_info(self) -> dict:
        info = self._mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin_free": float(info.margin_free),
            "leverage": int(info.leverage),
            "currency": info.currency,
            "login": int(info.login),
            "server": info.server,
        }

    # ── MT5-specific helpers ───────────────────────────────────────────────

    def get_open_position(self, symbol: str) -> Optional[dict]:
        """
        Returns info dict for the first open position matching symbol + magic,
        or None if no such position exists.
        """
        positions = self._mt5.positions_get(symbol=symbol)
        if not positions:
            return None
        for pos in positions:
            if pos.magic == self._magic:
                return {
                    "ticket": int(pos.ticket),
                    "symbol": pos.symbol,
                    "volume": float(pos.volume),
                    "direction": "buy" if pos.type == 0 else "sell",
                    "price_open": float(pos.price_open),
                    "sl": float(pos.sl),
                    "tp": float(pos.tp),
                    "profit": float(pos.profit),
                }
        return None

    def get_position_close_info(
        self,
        ticket: int,
        from_ts: pd.Timestamp,
    ) -> tuple[float, str]:
        """
        After a position is detected as closed, look up the actual fill price
        and derive the close reason (tp2 / sl / be / unknown).

        Returns (close_price, reason).
        """
        mt5 = self._mt5
        from_dt = datetime.fromtimestamp(int(from_ts.timestamp()), tz=timezone.utc)
        to_dt = datetime.now(tz=timezone.utc)

        deals = mt5.history_deals_get(from_dt, to_dt)
        if deals:
            for deal in reversed(deals):
                if (
                    deal.position_id == ticket
                    and deal.entry == mt5.DEAL_ENTRY_OUT
                ):
                    return float(deal.price), _reason_from_deal(deal, mt5)

        return 0.0, "unknown"

    def fetch_ohlcv(self, symbol: str, timeframe_mt5: int, n_bars: int) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars from MT5 terminal.

        timeframe_mt5 : one of mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, etc.
        n_bars        : number of bars to fetch (0 = most recent)
        """
        rates = self._mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, n_bars)
        if rates is None or len(rates) == 0:
            log.warning(f"fetch_ohlcv: no data for {symbol}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        # Drop the last (incomplete) bar — it's still forming
        return df.iloc[:-1]

    def reconnect(self, retries: int = 4) -> bool:
        """Attempt to reconnect with exponential backoff."""
        self.disconnect()
        for attempt in range(1, retries + 1):
            log.info(f"Reconnect attempt {attempt}/{retries}…")
            if self.connect():
                return True
            wait = 2 ** attempt
            log.info(f"Waiting {wait}s before next attempt")
            time.sleep(wait)
        return False


# ── Helpers ─────────────────────────────────────────────────────────────────

def _reason_from_deal(deal, mt5) -> str:
    """Derive close reason string from a deal's reason code."""
    reason_map = {
        getattr(mt5, "DEAL_REASON_TP", 3): "tp2",
        getattr(mt5, "DEAL_REASON_SL", 4): "sl",
        getattr(mt5, "DEAL_REASON_CLIENT", 0): "manual",
        getattr(mt5, "DEAL_REASON_EXPERT", 2): "expert",
    }
    return reason_map.get(deal.reason, "unknown")
