"""
Trade Setup Scoring Engine (0–100).

Each trade signal is scored across 7 dimensions:
  1. HTF bias alignment        → max 20
  2. Liquidity sweep strength  → max 20
  3. CHoCH clarity             → max 15
  4. OB / FVG quality          → max 15
  5. Session timing            → max 10
  6. ATR / volatility          → max 10
  7. Clean structure           → max 10
"""
from __future__ import annotations

from dataclasses import dataclass

from strategy.ltf_execution import TradeSignal
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ScoreBreakdown:
    htf_alignment: float = 0.0
    sweep_strength: float = 0.0
    choch_clarity: float = 0.0
    ob_fvg_quality: float = 0.0
    session_timing: float = 0.0
    atr_volatility: float = 0.0
    clean_structure: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.htf_alignment
            + self.sweep_strength
            + self.choch_clarity
            + self.ob_fvg_quality
            + self.session_timing
            + self.atr_volatility
            + self.clean_structure
        )

    def to_dict(self) -> dict:
        return {
            "htf_alignment": self.htf_alignment,
            "sweep_strength": self.sweep_strength,
            "choch_clarity": self.choch_clarity,
            "ob_fvg_quality": self.ob_fvg_quality,
            "session_timing": self.session_timing,
            "atr_volatility": self.atr_volatility,
            "clean_structure": self.clean_structure,
            "total": self.total,
        }


def _score_htf_alignment(signal: TradeSignal) -> float:
    """
    20 pts — both 4H and 1H confirm?
    Both TF agree = 20, only primary = 12, only secondary = 8.
    We use zone quality as a proxy here.
    """
    if signal.zone is None:
        return 0.0
    q = signal.zone.quality_score  # 0–1
    return round(q * 20, 1)


def _score_sweep_strength(signal: TradeSignal) -> float:
    """
    20 pts — bigger sweep = more liquidity taken = stronger signal.
    Thresholds configured via SWEEP_HIGH/MED/LOW/MIN_PIPS in .env.
    """
    pips = signal.sweep_size
    if pips >= settings.SWEEP_HIGH_PIPS:
        return 20.0
    if pips >= settings.SWEEP_MED_PIPS:
        return 15.0
    if pips >= settings.SWEEP_LOW_PIPS:
        return 10.0
    if pips >= settings.SWEEP_MIN_PIPS:
        return 5.0
    return 0.0


def _score_choch_clarity(signal: TradeSignal) -> float:
    """
    15 pts — CHoCH confirmed on LTF + optional candle confirmation.
    """
    score = 0.0
    if signal.choch_confirmed:
        score += 10.0
    if signal.candle_confirmation:
        score += 5.0
    return score


def _score_ob_fvg_quality(signal: TradeSignal) -> float:
    """
    15 pts — OB alone = 8, FVG alone = 6, confluence = 15.
    """
    if signal.zone is None:
        return 0.0
    if signal.zone.has_confluence:
        return 15.0
    if signal.zone.ob and signal.zone.fvg:
        return 12.0
    if signal.zone.ob:
        return 8.0
    if signal.zone.fvg:
        return 6.0
    return 0.0


def _score_session(signal: TradeSignal) -> float:
    """
    10 pts — overlap > newyork > london > off.
    """
    mapping = {"overlap": 10.0, "newyork": 8.0, "london": 7.0, "asian": 5.0, "off": 0.0}
    return mapping.get(signal.session, 0.0)


def _score_atr(signal: TradeSignal) -> float:
    """
    10 pts — ATR should be >= threshold (trending market).
    Proportional scoring relative to threshold.
    """
    if signal.atr <= 0:
        return 0.0
    ratio = signal.atr / settings.ATR_THRESHOLD
    if ratio >= 2.0:
        return 10.0
    if ratio >= 1.5:
        return 7.5
    if ratio >= 1.0:
        return 5.0
    return 2.0


def _score_clean_structure(signal: TradeSignal) -> float:
    """
    10 pts — zone internal_bos (validates no chop) + RR quality.
    """
    score = 0.0
    if signal.zone and signal.zone.internal_bos:
        score += 5.0
    # RR bonus: ≥ 3.0 = full; ≥ 2.5 = partial
    if signal.rr >= 3.0:
        score += 5.0
    elif signal.rr >= 2.5:
        score += 3.0
    elif signal.rr >= 2.0:
        score += 1.0
    return score


def score_signal(signal: TradeSignal) -> tuple[float, ScoreBreakdown]:
    """
    Score a trade signal. Returns (total_score, breakdown).
    """
    bd = ScoreBreakdown(
        htf_alignment=_score_htf_alignment(signal),
        sweep_strength=_score_sweep_strength(signal),
        choch_clarity=_score_choch_clarity(signal),
        ob_fvg_quality=_score_ob_fvg_quality(signal),
        session_timing=_score_session(signal),
        atr_volatility=_score_atr(signal),
        clean_structure=_score_clean_structure(signal),
    )
    total = bd.total
    signal.score = total

    decision = "EXECUTE" if total >= settings.MIN_SCORE_EXECUTE else (
        "OPTIONAL" if total >= settings.MIN_SCORE_OPTIONAL else "SKIP"
    )
    log.info(f"Score: {total:.1f}/100 → {decision} | {bd.to_dict()}")
    return total, bd


def should_execute(score: float) -> bool:
    return score >= settings.MIN_SCORE_EXECUTE
