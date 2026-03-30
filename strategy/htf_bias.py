"""
HTF Bias Engine — Layer 1 of the MTF model.

Processes 4H and 1H data to determine the overall directional bias
and identify key HTF liquidity zones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.market_structure import MarketStructure, Bias, analyse as ms_analyse
from core.liquidity import LiquidityMap, build_liquidity_map
from core.order_blocks import OrderBlock, detect_order_blocks, check_mitigation, get_valid_obs
from core.fvg import FairValueGap, detect_fvgs, check_fills, get_valid_fvgs
from core.atr import calculate_atr
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class HTFAnalysis:
    """Complete HTF output — fed downstream to MTF refinement."""
    bias: Bias = Bias.NEUTRAL
    bias_confirmed: bool = False   # True only when BOTH 4H and 1H agree on direction
    primary_tf: str = "4H"
    secondary_tf: str = "1H"
    ms_primary: Optional[MarketStructure] = None
    ms_secondary: Optional[MarketStructure] = None
    liquidity_primary: Optional[LiquidityMap] = None
    liquidity_secondary: Optional[LiquidityMap] = None
    valid_obs: list[OrderBlock] = field(default_factory=list)
    valid_fvgs: list[FairValueGap] = field(default_factory=list)
    premium_discount_range: Optional[tuple[float, float]] = None  # (low, high)


def _calc_premium_discount(
    ms: MarketStructure,
) -> Optional[tuple[float, float]]:
    """
    Premium / Discount range based on the last swing high-low leg.
    Discount zone = below 50% of range (buy zone in bullish market)
    Premium zone  = above 50% of range (sell zone in bearish market)
    """
    if not ms.swing_highs or not ms.swing_lows:
        return None

    sh = ms.swing_highs[-1].price
    sl = ms.swing_lows[-1].price
    return min(sl, sh), max(sl, sh)


def run_htf_bias(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    primary_tf: str = "4H",
    secondary_tf: str = "1H",
    swing_lookback: int = None,
    point_value: float = None,
) -> HTFAnalysis:
    """
    Run full HTF bias analysis.

    Args:
        df_primary   : 4H OHLCV DataFrame
        df_secondary : 1H OHLCV DataFrame
        primary_tf   : label for primary TF
        secondary_tf : label for secondary TF
        swing_lookback: bars each side for swing detection
        point_value  : pip / point size

    Returns:
        HTFAnalysis object with bias, zones, OBs, FVGs
    """
    if swing_lookback is None:
        swing_lookback = settings.SWING_LOOKBACK_HTF
    if point_value is None:
        point_value = settings.POINT_VALUE
    result = HTFAnalysis(primary_tf=primary_tf, secondary_tf=secondary_tf)

    # ── Market structure ───────────────────────────────────────────────────
    result.ms_primary = ms_analyse(df_primary, lookback=swing_lookback)
    result.ms_secondary = ms_analyse(df_secondary, lookback=swing_lookback)

    # Bias: both TFs agree → confirmed; one TF → unconfirmed (filtered by FILTER_HTF_CONFLUENCE)
    if (
        result.ms_primary.bias == result.ms_secondary.bias
        and result.ms_primary.bias != Bias.NEUTRAL
    ):
        result.bias = result.ms_primary.bias
        result.bias_confirmed = True
        log.debug(f"HTF bias CONFIRMED: {result.bias.value} (both TFs agree)")
    elif result.ms_primary.bias != Bias.NEUTRAL:
        result.bias = result.ms_primary.bias
        log.debug(f"HTF bias from primary only (unconfirmed): {result.bias.value}")
    elif result.ms_secondary.bias != Bias.NEUTRAL:
        result.bias = result.ms_secondary.bias
        log.debug(f"HTF bias from secondary only (unconfirmed): {result.bias.value}")
    else:
        result.bias = Bias.NEUTRAL
        log.debug("HTF bias: NEUTRAL — no trades will be taken")

    # ── Liquidity ──────────────────────────────────────────────────────────
    result.liquidity_primary = build_liquidity_map(df_primary, point_value=point_value)
    result.liquidity_secondary = build_liquidity_map(df_secondary, point_value=point_value)

    # ── Order blocks ───────────────────────────────────────────────────────
    bias_str = result.bias.value
    obs_primary = detect_order_blocks(df_primary)
    obs_primary = check_mitigation(df_primary, obs_primary)
    obs_secondary = detect_order_blocks(df_secondary)
    obs_secondary = check_mitigation(df_secondary, obs_secondary)

    result.valid_obs = (
        get_valid_obs(obs_primary, bias_str) + get_valid_obs(obs_secondary, bias_str)
    )

    # ── Fair Value Gaps ────────────────────────────────────────────────────
    fvgs_primary = detect_fvgs(df_primary)
    fvgs_primary = check_fills(df_primary, fvgs_primary)
    fvgs_secondary = detect_fvgs(df_secondary)
    fvgs_secondary = check_fills(df_secondary, fvgs_secondary)

    result.valid_fvgs = (
        get_valid_fvgs(fvgs_primary, bias_str) + get_valid_fvgs(fvgs_secondary, bias_str)
    )

    # ── Premium / Discount ─────────────────────────────────────────────────
    result.premium_discount_range = _calc_premium_discount(result.ms_primary)

    return result
