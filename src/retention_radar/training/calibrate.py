"""Probability calibration on the validation set (isotonic or sigmoid).

Fits a post-hoc calibrator on XGBoost validation probabilities so reported
churn probabilities are closer to true frequencies (lower Brier score).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.retention_radar import config


class ProbabilityCalibrator:
    """Thin wrapper: transform raw positive-class probabilities → calibrated."""

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "sigmoid"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, y_true, y_prob) -> "ProbabilityCalibrator":
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(y_prob, y_true)
        else:
            self._platt = LogisticRegression(
                max_iter=200, random_state=config.RANDOM_SEED
            )
            self._platt.fit(y_prob.reshape(-1, 1), y_true)
        return self

    def transform(self, y_prob) -> np.ndarray:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        if self.method == "isotonic":
            if self._iso is None:
                raise RuntimeError("Calibrator not fitted")
            return np.clip(self._iso.predict(y_prob), 0.0, 1.0)
        if self._platt is None:
            raise RuntimeError("Calibrator not fitted")
        return self._platt.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    def predict_proba_positive(self, y_prob) -> np.ndarray:
        return self.transform(y_prob)


def fit_calibrator(
    y_val,
    raw_val_prob,
    method: str | None = None,
) -> ProbabilityCalibrator:
    """Fit a validation-set calibrator (isotonic or sigmoid)."""
    method = method or config.CALIBRATION_METHOD
    cal = ProbabilityCalibrator(method=method)
    cal.fit(y_val, raw_val_prob)
    return cal


def brier(y_true, y_prob) -> float:
    """Brier score (lower is better)."""
    return float(brier_score_loss(y_true, y_prob))


def save_calibrator(
    calibrator: ProbabilityCalibrator, path: Path | None = None
) -> Path:
    path = path or config.CALIBRATOR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrator": calibrator, "method": calibrator.method}, path)
    return path


def load_calibrator(path: Path | None = None) -> ProbabilityCalibrator | None:
    path = path or config.CALIBRATOR_PATH
    if not path.exists():
        return None
    payload = joblib.load(path)
    if isinstance(payload, dict) and "calibrator" in payload:
        return payload["calibrator"]
    if isinstance(payload, ProbabilityCalibrator):
        return payload
    return None
