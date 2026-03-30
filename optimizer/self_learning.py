"""
Self-Learning & Adaptive Optimisation System.

Analyses completed trade logs and:
  1. Computes per-session and per-setup win rates
  2. Adjusts scoring weights for high/low performers
  3. Auto-tunes parameters (RR, ATR threshold)
  4. Reinforcement-style reward/penalty for setups
  5. Safety: minimum trades before changes, drawdown protection

All learned parameters are saved to optimizer/learned_params.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ai.feature_engineering import FEATURE_NAMES
from ai.probability_model import ProbabilityModel
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_PARAMS_FILE = Path("optimizer") / "learned_params.json"
_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class LearnedParams:
    """Runtime-adjustable parameters that the self-learning system can modify."""
    min_score_execute: float = float(settings.MIN_SCORE_EXECUTE)
    ai_probability_threshold: float = settings.AI_PROBABILITY_THRESHOLD
    min_rr: float = settings.MIN_RR
    atr_threshold_multiplier: float = 1.0   # multiplied against base ATR threshold
    risk_per_trade: float = settings.RISK_PER_TRADE

    # Session weights (multiplied into session score)
    session_weights: dict = field(default_factory=lambda: {
        "london": 1.0, "newyork": 1.0, "overlap": 1.0, "off": 0.0
    })

    # Setup weights: "ob+fvg", "ob_only", "fvg_only"
    setup_weights: dict = field(default_factory=lambda: {
        "ob+fvg": 1.0, "ob_only": 0.8, "fvg_only": 0.6
    })

    version: int = 0

    def save(self) -> None:
        with open(_PARAMS_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.info(f"Learned params saved → {_PARAMS_FILE} (v{self.version})")

    @classmethod
    def load(cls) -> "LearnedParams":
        if _PARAMS_FILE.exists():
            try:
                with open(_PARAMS_FILE) as f:
                    data = json.load(f)
                p = cls(**data)
                log.info(f"Learned params loaded (v{p.version})")
                return p
            except Exception as e:
                log.warning(f"Could not load learned params: {e}")
        return cls()


class SelfLearningSystem:
    """
    Analyses trade history and adapts parameters.
    """

    def __init__(self) -> None:
        self.params = LearnedParams.load()
        self.ai = ProbabilityModel()

    # ── Analysis ───────────────────────────────────────────────────────────

    def analyse_trades(self, df: pd.DataFrame) -> dict:
        """
        Returns performance breakdown by session and setup type.
        df must have columns: outcome, session, has_ob, has_fvg, has_confluence, pnl_currency
        """
        if len(df) < settings.MIN_TRADES_BEFORE_ADAPT:
            log.warning(
                f"Only {len(df)} trades — need {settings.MIN_TRADES_BEFORE_ADAPT} before adapting"
            )
            return {}

        analysis: dict = {}

        # By session
        for session in ["london", "newyork", "overlap"]:
            sub = df[df["session"] == session]
            if len(sub) == 0:
                continue
            wins = (sub["outcome"] == "win").sum()
            analysis[f"win_rate_{session}"] = wins / len(sub)
            analysis[f"avg_rr_{session}"] = sub["rr_achieved"].mean()

        # By setup type
        for setup, mask in [
            ("ob+fvg", df["has_confluence"] == True),
            ("ob_only", (df["has_ob"] == True) & (df["has_confluence"] == False)),
            ("fvg_only", (df["has_fvg"] == True) & (df["has_ob"] == False)),
        ]:
            sub = df[mask]
            if len(sub) == 0:
                continue
            wins = (sub["outcome"] == "win").sum()
            analysis[f"win_rate_{setup}"] = wins / len(sub)

        log.info(f"Analysis complete: {analysis}")
        return analysis

    # ── Adaptive logic ─────────────────────────────────────────────────────

    def adapt(self, df: pd.DataFrame) -> LearnedParams:
        """
        Update params based on trade performance.
        Only modifies params when evidence is statistically meaningful.
        """
        if len(df) < settings.MIN_TRADES_BEFORE_ADAPT:
            return self.params

        # Safety: check drawdown — halt adaptation if too deep
        if "pnl_currency" in df.columns:
            cumulative = df["pnl_currency"].cumsum()
            peak = cumulative.cummax()
            dd = (peak - cumulative) / peak.replace(0, np.nan)
            max_dd = dd.max()
            if max_dd >= settings.MAX_DRAWDOWN_PROTECTION:
                log.warning(f"Adaptation skipped: max drawdown {max_dd:.1%} too high")
                return self.params

        stats = self.analyse_trades(df)
        if not stats:
            return self.params

        # ── Session weight adjustment ──────────────────────────────────────
        for session in ["london", "newyork", "overlap"]:
            key = f"win_rate_{session}"
            if key in stats:
                wr = stats[key]
                if wr < 0.35:
                    self.params.session_weights[session] *= 0.85   # reduce
                    log.info(f"Reducing {session} weight (WR={wr:.1%})")
                elif wr > 0.60:
                    self.params.session_weights[session] = min(
                        1.5, self.params.session_weights[session] * 1.1
                    )
                    log.info(f"Increasing {session} weight (WR={wr:.1%})")

        # ── Setup weight adjustment ────────────────────────────────────────
        for setup in ["ob+fvg", "ob_only", "fvg_only"]:
            key = f"win_rate_{setup}"
            if key in stats:
                wr = stats[key]
                if wr < 0.35:
                    self.params.setup_weights[setup] = max(0.3, self.params.setup_weights[setup] * 0.9)
                elif wr > 0.60:
                    self.params.setup_weights[setup] = min(1.5, self.params.setup_weights[setup] * 1.05)

        # ── RR / threshold tuning ──────────────────────────────────────────
        overall_wr = (df["outcome"] == "win").sum() / len(df)
        if overall_wr < 0.40:
            # Raise quality bar
            self.params.min_score_execute = min(80, self.params.min_score_execute + 2)
            self.params.ai_probability_threshold = min(0.85, self.params.ai_probability_threshold + 0.02)
            log.info(f"Raised quality bar: score≥{self.params.min_score_execute}, AI≥{self.params.ai_probability_threshold:.2f}")
        elif overall_wr > 0.65 and len(df) > 50:
            # Can loosen slightly
            self.params.min_score_execute = max(65, self.params.min_score_execute - 1)

        # ── Retrain AI model ───────────────────────────────────────────────
        if len(df) >= settings.AI_MIN_SAMPLES_TO_TRAIN and all(
            c in df.columns for c in FEATURE_NAMES + ["outcome"]
        ):
            log.info("Retraining AI probability model with new data...")
            self.ai.train_from_df(df[FEATURE_NAMES + ["outcome"]])

        self.params.version += 1
        self.params.save()
        return self.params

    # ── Reinforcement-style reward / penalty ──────────────────────────────

    def apply_reinforcement(self, df: pd.DataFrame) -> None:
        """
        Adjust internal weights based on reward/penalty logic.
        Win = reward → increase feature weights.
        Loss = penalty → decrease feature weights.
        This feeds back into the scoring system via LearnedParams.
        """
        if "outcome" not in df.columns or len(df) < 10:
            return

        recent = df.tail(20)   # look at last 20 trades
        recent_wr = (recent["outcome"] == "win").sum() / len(recent)

        if recent_wr < 0.35:
            # Consecutive losses → tighten risk
            self.params.risk_per_trade = max(0.005, self.params.risk_per_trade * 0.9)
            log.warning(f"Recent WR {recent_wr:.0%} — reducing risk to {self.params.risk_per_trade:.1%}")
        elif recent_wr > 0.65 and self.params.risk_per_trade < settings.RISK_PER_TRADE:
            # Good streak → restore risk
            self.params.risk_per_trade = min(settings.RISK_PER_TRADE, self.params.risk_per_trade * 1.05)
            log.info(f"Recent WR {recent_wr:.0%} — restoring risk to {self.params.risk_per_trade:.1%}")

        self.params.save()
