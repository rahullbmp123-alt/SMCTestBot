"""
LTF (5m) Execution Engine — Layer 3 of the MTF model.

Conditions ALL must be met before a trade signal is generated:
  1. Price enters a refined zone
  2. Liquidity sweep occurs at zone level
  3. CHoCH on 5m confirms directional shift
  4. Optional candle confirmation (engulfing / pin bar)

Outputs a TradeSignal with full SL/TP/RR parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import uuid

import pandas as pd
import numpy as np

from core.market_structure import Bias, analyse as ms_analyse
from core.liquidity import build_liquidity_map, get_recent_sweep
from core.atr import calculate_atr, lot_size_from_risk
from strategy.mtf_refinement import MTFAnalysis, RefinedZone
from utils.session import get_session, is_tradeable_session
from utils.logger import get_logger
from config import settings

log = get_logger(__name__)


@dataclass
class TradeSignal:
    """A fully specified trade signal ready for scoring and execution."""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    direction: str = ""          # "buy" or "sell"
    entry_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0             # partial TP (1:1)
    tp2: float = 0.0             # full TP (1:2+)
    lot_size: float = 0.0
    rr: float = 0.0
    timestamp: Optional[pd.Timestamp] = None
    session: str = ""
    atr: float = 0.0
    zone: Optional[RefinedZone] = None
    reason: str = ""             # human-readable entry reason
    sweep_size: float = 0.0      # size of liquidity sweep in pips
    choch_confirmed: bool = False
    candle_confirmation: bool = False
    score: float = 0.0           # filled by scoring engine
    ai_probability: float = 0.0  # filled by AI filter

    @property
    def sl_pips(self) -> float:
        return abs(self.entry_price - self.sl) / settings.POINT_VALUE

    @property
    def tp2_pips(self) -> float:
        return abs(self.tp2 - self.entry_price) / settings.POINT_VALUE


def _check_choch(df_5m: pd.DataFrame, bias: Bias) -> bool:
    """Check if a CHoCH has occurred on the 5m chart in the expected direction."""
    ms = ms_analyse(df_5m, lookback=settings.SWING_LOOKBACK_LTF)
    for brk in reversed(ms.breaks[-settings.CHOCH_BREAKS_TO_CHECK:]):
        if brk.kind == "CHoCH":
            if bias == Bias.BULLISH and brk.direction == "bullish":
                return True
            if bias == Bias.BEARISH and brk.direction == "bearish":
                return True
    return False


def _check_candle_confirmation(df_5m: pd.DataFrame, direction: str) -> bool:
    """
    Optional confirmation patterns on the last closed candle:
      - Bullish engulfing / pin bar (hammer) for longs
      - Bearish engulfing / shooting star for shorts
    """
    if len(df_5m) < 2:
        return False

    last = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]
    if candle_range == 0:
        return False

    if direction == "buy":
        # Bullish engulfing: current bullish body fully engulfs previous bearish body
        engulfing = (
            prev["close"] < prev["open"]           # previous candle bearish
            and last["close"] > last["open"]        # current candle bullish
            and last["open"] <= prev["close"]       # current open at/below prev close
            and last["close"] >= prev["open"]       # current close at/above prev open
        )
        # Hammer: lower wick > 2× body, small upper wick
        lower_wick = min(last["open"], last["close"]) - last["low"]
        upper_wick = last["high"] - max(last["open"], last["close"])
        hammer = lower_wick > settings.CANDLE_WICK_RATIO * body and upper_wick < body
        return engulfing or hammer

    if direction == "sell":
        # Bearish engulfing: current bearish body fully engulfs previous bullish body
        engulfing = (
            prev["close"] > prev["open"]            # previous candle bullish
            and last["close"] < last["open"]        # current candle bearish
            and last["open"] >= prev["close"]       # current open at/above prev close
            and last["close"] <= prev["open"]       # current close at/below prev open
        )
        # Shooting star: upper wick > 2× body, small lower wick
        upper_wick = last["high"] - max(last["open"], last["close"])
        lower_wick = min(last["open"], last["close"]) - last["low"]
        star = upper_wick > settings.CANDLE_WICK_RATIO * body and lower_wick < body
        return engulfing or star

    return False


def _build_sl_tp(
    entry: float,
    direction: str,
    zone: RefinedZone,
    atr: float,
    min_rr: float,
) -> tuple[float, float, float]:
    """
    Calculate structure-based SL, TP1 (1:1), TP2 (based on min_rr).
    SL is placed just beyond the zone boundary with an ATR buffer.
    """
    buffer = atr * settings.SL_ATR_BUFFER

    if direction == "buy":
        sl = zone.entry_bottom - buffer
        risk = entry - sl
        tp1 = entry + risk
        tp2 = entry + risk * min_rr
    else:
        sl = zone.entry_top + buffer
        risk = sl - entry
        tp1 = entry - risk
        tp2 = entry - risk * min_rr

    return sl, tp1, tp2


def generate_signal(
    df_5m: pd.DataFrame,
    mtf: MTFAnalysis,
    current_price: float,
    spread: float = 0.0,
    account_balance: float = None,
    stats: dict | None = None,
) -> Optional[TradeSignal]:
    """
    Check all LTF execution conditions and return a TradeSignal if valid.
    Returns None if conditions not met.

    Args:
        stats: optional dict; if provided, the rejection reason is written
               to stats["rejected_by"] on every None return.
    """
    def _reject(reason: str):
        if stats is not None:
            stats["rejected_by"] = reason
        return None

    if mtf.best_zone is None:
        return _reject("no_zone")

    if account_balance is None:
        account_balance = settings.ACCOUNT_BALANCE

    # ── HTF confluence filter ──────────────────────────────────────────────
    # Only trade when BOTH 4H and 1H agree; a single-TF bias is unreliable
    if settings.FILTER_HTF_CONFLUENCE and not mtf.htf.bias_confirmed:
        return _reject("htf_unconfirmed")

    # ── Session filter ─────────────────────────────────────────────────────
    now = df_5m.index[-1].to_pydatetime()
    if settings.FILTER_SESSION and not is_tradeable_session(now):
        return _reject("session")

    # ── Spread filter ──────────────────────────────────────────────────────
    if settings.FILTER_SPREAD and spread > settings.SPREAD_LIMIT:
        return _reject("spread")

    zone = mtf.best_zone
    bias = mtf.htf.bias
    direction = "buy" if bias == Bias.BULLISH else "sell"

    # ── Condition 1: Price in zone ─────────────────────────────────────────
    if settings.FILTER_PRICE_IN_ZONE and not zone.price_in_zone(current_price):
        return _reject("price_not_in_zone")

    # ── Condition 1b: Zone entry half ──────────────────────────────────────
    # Enter only in the half of the zone closest to the structure boundary:
    #   bearish zone → upper half (near zone_top) → minimises SL distance
    #   bullish zone → lower half (near zone_bottom) → minimises SL distance
    # This ensures avg win ≥ avg loss at the configured RR.
    if settings.FILTER_ZONE_ENTRY_HALF:
        zone_mid = (zone.entry_top + zone.entry_bottom) / 2
        if direction == "sell" and current_price < zone_mid:
            return _reject("price_not_at_premium_half")
        if direction == "buy" and current_price > zone_mid:
            return _reject("price_not_at_discount_half")

    # ── Condition 2: Liquidity sweep ───────────────────────────────────────
    lm = build_liquidity_map(df_5m, point_value=settings.POINT_VALUE)
    sweep_kind = "SSL" if direction == "buy" else "BSL"
    recent_sweep = get_recent_sweep(lm, sweep_kind)
    if settings.FILTER_LIQUIDITY_SWEEP and (recent_sweep is None or not recent_sweep.swept):
        return _reject("no_sweep")

    # ── Condition 3: CHoCH confirmation ────────────────────────────────────
    choch = _check_choch(df_5m, bias)
    if settings.FILTER_CHOCH and not choch:
        return _reject("no_choch")

    # ── ATR ────────────────────────────────────────────────────────────────
    atr = float(calculate_atr(df_5m).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return _reject("zero_atr")

    # ── SL / TP ────────────────────────────────────────────────────────────
    sl, tp1, tp2 = _build_sl_tp(current_price, direction, zone, atr, settings.MIN_RR)

    # ── RR check ──────────────────────────────────────────────────────────
    risk = abs(current_price - sl)
    reward = abs(tp2 - current_price)
    if risk == 0:
        return _reject("zero_risk")
    if settings.FILTER_MIN_RR and (reward / risk) < settings.MIN_RR:
        log.debug(f"LTF: RR {reward/risk:.2f} below minimum {settings.MIN_RR} — skip")
        return _reject("min_rr")

    # ── Max SL distance filter ─────────────────────────────────────────────
    # Reject if the zone-based SL is unrealistically wide (e.g. a 200-bar-old
    # OB producing a $140 SL on a $10 account risk → lot=0.00).
    max_sl_distance = atr * settings.MAX_SL_ATR_MULT
    if risk > max_sl_distance:
        log.debug(
            f"LTF: SL too wide ({risk:.2f} > {max_sl_distance:.2f} = "
            f"{settings.MAX_SL_ATR_MULT}× ATR) — skip"
        )
        return _reject("sl_too_wide")

    # ── Lot size ───────────────────────────────────────────────────────────
    sl_pips = risk / settings.POINT_VALUE
    if settings.FIXED_LOT_SIZE > 0:
        # Fixed lot from env — trade this exact size regardless of balance or SL
        lot = settings.FIXED_LOT_SIZE
    else:
        lot = lot_size_from_risk(
            account_balance=account_balance,
            risk_pct=settings.RISK_PER_TRADE,
            sl_pips=sl_pips,
            pip_value=settings.POINT_VALUE,
            contract_size=settings.CONTRACT_SIZE,
            lot_tier_capital=settings.LOT_TIER_CAPITAL,
            min_lot=settings.MIN_LOT_SIZE,
        )

    # Reject if lot rounds to zero — would open a trade with no financial impact
    if lot == 0.0:
        log.debug(f"LTF: lot size rounds to zero (SL={risk:.4f}, balance={account_balance:.2f}) — skip")
        return _reject("zero_lot")

    # ── Actual risk check ──────────────────────────────────────────────────
    # Skip when FIXED_LOT_SIZE is set — user explicitly chose the size and
    # accepts the dollar risk at any SL distance.
    if settings.FIXED_LOT_SIZE == 0.0:
        actual_risk_usd = lot * sl_pips * settings.POINT_VALUE * settings.CONTRACT_SIZE
        max_allowed_usd = account_balance * settings.RISK_PER_TRADE * settings.MAX_RISK_TOLERANCE_MULT
        if actual_risk_usd > max_allowed_usd:
            log.debug(
                f"LTF: actual risk ${actual_risk_usd:.2f} > allowed ${max_allowed_usd:.2f} "
                f"({settings.MAX_RISK_TOLERANCE_MULT}× budget) — skip"
            )
            return _reject("risk_too_high")

    # ── Zone quality filter ────────────────────────────────────────────────
    # Reject standalone FVG-only zones (quality < 0.5 = no OB anchor or
    # no internal BOS confirmation). Forces at least OB + internal structure.
    if settings.MIN_ZONE_QUALITY > 0 and zone.quality_score < settings.MIN_ZONE_QUALITY:
        log.debug(
            f"LTF: zone quality {zone.quality_score:.2f} < min {settings.MIN_ZONE_QUALITY} "
            f"(ob={bool(zone.ob)}, fvg={bool(zone.fvg)}, confluence={zone.has_confluence}, "
            f"internal_bos={zone.internal_bos}) — skip"
        )
        return _reject("zone_quality_low")

    # ── Entry bar momentum filter ──────────────────────────────────────────
    # Reject if the entry bar closed against the trade direction.
    # For SELL: close must be in the lower portion of the bar range (bearish close).
    # For BUY: close must be in the upper portion of the bar range (bullish close).
    # Catches "immediate reversal" entries where price entered zone then closed wrong way.
    if settings.FILTER_ENTRY_BAR_CLOSE:
        _last = df_5m.iloc[-1]
        _bar_range = _last["high"] - _last["low"]
        if _bar_range > 0:
            _close_pos = (_last["close"] - _last["low"]) / _bar_range
            if direction == "sell" and _close_pos > (1.0 - settings.ENTRY_BAR_CLOSE_PCT):
                return _reject("entry_bar_bullish")
            if direction == "buy" and _close_pos < settings.ENTRY_BAR_CLOSE_PCT:
                return _reject("entry_bar_bearish")

    # ── Optional candle confirmation ───────────────────────────────────────
    candle_conf = _check_candle_confirmation(df_5m, direction)

    # ── Candle confirmation filter ─────────────────────────────────────────
    # Require a reversal candle pattern (engulfing / hammer / shooting star)
    # at the zone before entering. This adds a second price-action gate after
    # CHoCH, filtering out zones where price enters but shows no reversal body.
    if settings.FILTER_CANDLE_CONFIRMATION and not candle_conf:
        log.debug("LTF: no candle confirmation (engulfing/hammer/shooting star) — skip")
        return _reject("no_candle_conf")

    session = get_session(now)
    reason_parts = [
        f"zone_entry ({zone.kind})",
        f"sweep ({sweep_kind})",
        "choch_confirmed",
    ]
    if zone.ob:
        reason_parts.append("OB")
    if zone.fvg:
        reason_parts.append("FVG")
    if zone.has_confluence:
        reason_parts.append("confluence")
    if candle_conf:
        reason_parts.append("candle_conf")

    signal = TradeSignal(
        symbol=settings.SYMBOL,
        direction=direction,
        entry_price=current_price,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        lot_size=lot,
        rr=round(reward / risk, 2),
        timestamp=df_5m.index[-1],
        session=session,
        atr=atr,
        zone=zone,
        reason=" + ".join(reason_parts),
        sweep_size=recent_sweep.sweep_size if recent_sweep else 0.0,
        choch_confirmed=choch,
        candle_confirmation=candle_conf,
    )

    log.info(
        f"LTF SIGNAL: {direction.upper()} {settings.SYMBOL} | "
        f"entry={current_price:.5f} SL={sl:.5f} TP={tp2:.5f} "
        f"RR={signal.rr} lots={lot}"
    )
    return signal
