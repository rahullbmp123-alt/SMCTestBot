"""
AI Probability Model — scikit-learn Random Forest classifier.

Predicts the probability of a trade winning based on signal features.
Model is retrained automatically as new trade data accumulates.
"""
from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from ai.feature_engineering import FEATURE_NAMES, build_feature_vector
from strategy.ltf_execution import TradeSignal
from strategy.scoring import ScoreBreakdown
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_MODEL_FILE = settings.MODELS_DIR / "probability_model.pkl"


class ProbabilityModel:
    """
    Wraps a scikit-learn pipeline for trade outcome prediction.

    Uses a calibrated Random Forest to output well-calibrated probabilities.
    Falls back to score-based heuristic if not enough data to train.
    """

    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False
        self.n_samples: int = 0
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────
    def _load(self) -> None:
        if _MODEL_FILE.exists():
            try:
                with open(_MODEL_FILE, "rb") as f:
                    data = pickle.load(f)
                self.pipeline = data["pipeline"]
                self.n_samples = data.get("n_samples", 0)
                self.is_trained = True
                log.info(f"AI model loaded (n_samples={self.n_samples})")
            except Exception as e:
                log.warning(f"Could not load model: {e}")

    def _save(self) -> None:
        with open(_MODEL_FILE, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "n_samples": self.n_samples}, f)
        log.info(f"AI model saved → {_MODEL_FILE}")

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train on feature matrix X and binary labels y.
        Returns cross-validated accuracy.
        """
        if len(X) < settings.AI_MIN_SAMPLES_TO_TRAIN:
            log.warning(
                f"Only {len(X)} samples — minimum is {settings.AI_MIN_SAMPLES_TO_TRAIN}. "
                "Skipping training."
            )
            return 0.0

        base = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        calibrated = CalibratedClassifierCV(base, cv=5, method="isotonic")
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ])
        self.pipeline.fit(X, y)
        self.n_samples = len(X)
        self.is_trained = True
        self._save()

        # CV score on training set
        cv_scores = cross_val_score(
            Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
            X, y, cv=min(5, len(X) // 10 or 2), scoring="accuracy"
        )
        acc = float(cv_scores.mean())
        log.info(f"AI model trained — CV accuracy: {acc:.3f} (n={len(X)})")
        return acc

    def train_from_df(self, df: pd.DataFrame) -> float:
        """Train from a DataFrame with FEATURE_NAMES columns + 'outcome' column."""
        if "outcome" not in df.columns:
            raise ValueError("DataFrame must have 'outcome' column")
        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df["outcome"].values.astype(int)
        return self.train(X, y)

    # ── Inference ──────────────────────────────────────────────────────────
    def predict_proba(
        self,
        signal: TradeSignal,
        breakdown: ScoreBreakdown,
    ) -> float:
        """Return probability of success (0–1)."""
        if not self.is_trained or self.pipeline is None:
            # Fallback heuristic: normalise score to 0–1
            prob = breakdown.total / 100.0
            log.debug(f"AI not trained — heuristic probability: {prob:.3f}")
            return prob

        features = build_feature_vector(signal, breakdown, settings.ATR_THRESHOLD)
        X = features.reshape(1, -1)
        prob = float(self.pipeline.predict_proba(X)[0][1])
        log.debug(f"AI probability: {prob:.3f}")
        return prob

    def should_trade(
        self,
        signal: TradeSignal,
        breakdown: ScoreBreakdown,
    ) -> tuple[bool, float]:
        """Return (allow_trade, probability)."""
        prob = self.predict_proba(signal, breakdown)
        signal.ai_probability = prob
        return prob >= settings.AI_PROBABILITY_THRESHOLD, prob
