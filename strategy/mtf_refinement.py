"""
MTF (15m) Zone Refinement — Layer 2 of the MTF model.

Takes HTF bias and narrows down to high-probability entry zones:
  1. OB+FVG confluence on 15m
  2. Internal BOS/CHoCH validation
  3. Only zones aligned with HTF bias are kept
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.market_structure import MarketStructure, Bias, analyse as ms_analyse
from core.order_blocks import OrderBlock, detect_order_blocks, check_mitigation, get_valid_obs
from core.fvg import FairValueGap, detect_fvgs, check_fills, get_valid_fvgs, get_confluence_zone
from strategy.htf_bias import HTFAnalysis
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RefinedZone:
    """A 15m zone ready for LTF execution monitoring."""
    kind: str                          # "bullish" or "bearish"
    entry_top: float
    entry_bottom: float
    ob: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    has_confluence: bool = False       # OB + FVG overlap
    internal_bos: bool = False         # internal structure confirms direction
    timestamp: Optional[pd.Timestamp] = None
    quality_score: float = 0.0        # 0–1 internal ranking

    @property
    def mid(self) -> float:
        return (self.entry_top + self.entry_bottom) / 2

    def price_in_zone(self, price: float) -> bool:
        return self.entry_bottom <= price <= self.entry_top


@dataclass
class MTFAnalysis:
    htf: HTFAnalysis
    ms_15m: Optional[MarketStructure] = None
    refined_zones: list[RefinedZone] = field(default_factory=list)
    best_zone: Optional[RefinedZone] = None


def _internal_bos_confirmed(
    ms: MarketStructure, bias: Bias
) -> bool:
    """
    Check if the 15m internal structure has a BOS in the direction of HTF bias.
    """
    for brk in reversed(ms.breaks):
        if brk.kind == "BOS":
            if bias == Bias.BULLISH and brk.direction == "bullish":
                return True
            if bias == Bias.BEARISH and brk.direction == "bearish":
                return True
    return False


def _score_zone(zone: RefinedZone) -> float:
    score = 0.0
    if zone.ob:
        score += 0.4
    if zone.fvg:
        score += 0.3
    if zone.has_confluence:
        score += 0.2
    if zone.internal_bos:
        score += 0.1
    return round(score, 2)


def run_mtf_refinement(
    df_15m: pd.DataFrame,
    htf: HTFAnalysis,
    swing_lookback: int = None,
) -> MTFAnalysis:
    """
    Refine HTF bias into actionable 15m entry zones.
    """
    if swing_lookback is None:
        swing_lookback = settings.SWING_LOOKBACK_MTF
    result = MTFAnalysis(htf=htf)

    if htf.bias == Bias.NEUTRAL:
        log.warning("MTF: HTF bias is NEUTRAL — skipping refinement")
        return result

    bias_str = htf.bias.value

    # ── 15m structure ─────────────────────────────────────────────────────
    result.ms_15m = ms_analyse(df_15m, lookback=swing_lookback)

    # ── 15m OBs ───────────────────────────────────────────────────────────
    obs_15m = detect_order_blocks(df_15m, impulse_threshold=settings.OB_IMPULSE_THRESHOLD_MTF)
    obs_15m = check_mitigation(df_15m, obs_15m)
    valid_obs = get_valid_obs(obs_15m, bias_str)

    # ── 15m FVGs ──────────────────────────────────────────────────────────
    fvgs_15m = detect_fvgs(df_15m)
    fvgs_15m = check_fills(df_15m, fvgs_15m)
    valid_fvgs = get_valid_fvgs(fvgs_15m, bias_str)

    internal_bos = _internal_bos_confirmed(result.ms_15m, htf.bias)

    # ── Build zones ────────────────────────────────────────────────────────
    zones: list[RefinedZone] = []

    for ob in valid_obs:
        # Check if this OB is inside the HTF premium/discount range
        if settings.FILTER_PREMIUM_DISCOUNT and htf.premium_discount_range:
            lo, hi = htf.premium_discount_range
            if not (lo <= ob.mid <= hi):
                log.debug(f"MTF: OB {ob.mid:.2f} outside premium/discount range [{lo:.2f}-{hi:.2f}] — skip")
                continue   # OB outside HTF range — skip

        zone = RefinedZone(
            kind=bias_str,
            entry_top=ob.top,
            entry_bottom=ob.bottom,
            ob=ob,
            internal_bos=internal_bos,
            timestamp=ob.timestamp,
        )

        # Check confluence with any FVG
        for fvg in valid_fvgs:
            conf = get_confluence_zone(ob, fvg)
            if conf:
                zone.fvg = fvg
                zone.has_confluence = True
                zone.entry_bottom = conf[0]
                zone.entry_top = conf[1]
                break

        zone.quality_score = _score_zone(zone)
        zones.append(zone)

    # Standalone FVG zones (no OB overlap) — lower quality but still valid
    ob_fvg_used_ids = {id(z.fvg) for z in zones if z.fvg}
    for fvg in valid_fvgs:
        if id(fvg) in ob_fvg_used_ids:
            continue
        if settings.FILTER_PREMIUM_DISCOUNT and htf.premium_discount_range:
            lo, hi = htf.premium_discount_range
            if not (lo <= fvg.mid <= hi):
                log.debug(f"MTF: FVG {fvg.mid:.2f} outside premium/discount range [{lo:.2f}-{hi:.2f}] — skip")
                continue

        zone = RefinedZone(
            kind=bias_str,
            entry_top=fvg.top,
            entry_bottom=fvg.bottom,
            fvg=fvg,
            internal_bos=internal_bos,
            timestamp=fvg.timestamp,
        )
        zone.quality_score = _score_zone(zone)
        zones.append(zone)

    # Drop zones older than OB_ZONE_MAX_AGE_BARS (stale zones cause price_not_in_zone
    # rejections and entries at levels price has long moved away from)
    if len(df_15m) > 0:
        cutoff_ts = df_15m.index[-settings.OB_ZONE_MAX_AGE_BARS] if len(df_15m) > settings.OB_ZONE_MAX_AGE_BARS else df_15m.index[0]
        zones = [z for z in zones if z.timestamp is not None and z.timestamp >= cutoff_ts]

    # Sort by recency first (most recent zone = most relevant), quality as tiebreaker.
    # A fresh zone at quality 0.4 is preferable to a 90-bar-old zone at quality 0.9.
    result.refined_zones = sorted(
        zones,
        key=lambda z: (z.timestamp or pd.Timestamp.min, z.quality_score),
        reverse=True,
    )
    result.best_zone = result.refined_zones[0] if result.refined_zones else None

    log.debug(
        f"MTF: {len(result.refined_zones)} zones found, "
        f"bias={bias_str}, internal_bos={internal_bos}"
    )
    return result
