"""
Feature engineering for the AI probability model.
Converts a TradeSignal into a feature vector for ML inference.
"""
from __future__ import annotations

import numpy as np

from strategy.ltf_execution import TradeSignal
from strategy.scoring import ScoreBreakdown

# Ordered feature names — must match training exactly
FEATURE_NAMES = [
    "sweep_size_pips",
    "atr_ratio",          # atr / atr_threshold
    "rr_ratio",
    "session_encoded",    # 0=off, 1=london, 2=newyork, 3=overlap
    "score_total",
    "score_htf",
    "score_sweep",
    "score_choch",
    "score_ob_fvg",
    "score_session",
    "score_structure",
    "has_ob",
    "has_fvg",
    "has_confluence",
    "has_internal_bos",
    "candle_confirmation",
    "choch_confirmed",
    "lot_size",
]

_SESSION_MAP = {"off": 0, "london": 1, "newyork": 2, "overlap": 3}


def build_feature_vector(
    signal: TradeSignal,
    breakdown: ScoreBreakdown,
    atr_threshold: float,
) -> np.ndarray:
    zone = signal.zone

    features = [
        signal.sweep_size,
        signal.atr / atr_threshold if atr_threshold > 0 else 0.0,
        signal.rr,
        _SESSION_MAP.get(signal.session, 0),
        breakdown.total,
        breakdown.htf_alignment,
        breakdown.sweep_strength,
        breakdown.choch_clarity,
        breakdown.ob_fvg_quality,
        breakdown.session_timing,
        breakdown.clean_structure,
        float(zone.ob is not None) if zone else 0.0,
        float(zone.fvg is not None) if zone else 0.0,
        float(zone.has_confluence) if zone else 0.0,
        float(zone.internal_bos) if zone else 0.0,
        float(signal.candle_confirmation),
        float(signal.choch_confirmed),
        signal.lot_size,
    ]
    return np.array(features, dtype=np.float32)
