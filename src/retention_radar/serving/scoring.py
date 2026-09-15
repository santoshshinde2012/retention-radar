"""Score a feature frame with any ``predict_proba`` estimator + optional calibrator."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.retention_radar.protocols import Calibrator, ProbabilisticClassifier


class CalibratedScorer:
    """Depend on Protocols so Dummy / LogReg / XGB can swap without UI changes."""

    def __init__(
        self,
        classifier: ProbabilisticClassifier,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.classifier = classifier
        self.calibrator = calibrator

    def raw_positive(self, X: Any) -> np.ndarray:
        """Return raw P(churn) for each row."""
        proba = np.asarray(self.classifier.predict_proba(X), dtype=float)
        return proba[:, 1]

    def score(self, X: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Return ``(raw, calibrated_or_none, display)`` positive-class vectors."""
        raw = self.raw_positive(X)
        if self.calibrator is None:
            return raw, None, raw
        calibrated = np.asarray(self.calibrator.transform(raw), dtype=float).ravel()
        return raw, calibrated, calibrated
