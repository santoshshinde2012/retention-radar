"""Small serving/training Protocols (ISP) — no fat interfaces.

Callers depend on these abstractions so Dummy, LogReg, and XGBoost can swap
at score time without the UI knowing which estimator was trained.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class FeatureTransformer(Protocol):
    """Map a raw user table or dict onto the 22-column model contract."""

    def prepare_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return encoded ``X`` and label ``y`` for training."""
        ...

    def row_to_feature_frame(self, row: dict | pd.Series) -> pd.DataFrame:
        """Return a 1-row feature frame for scoring."""
        ...


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    """Anything with sklearn-style ``predict_proba`` (Dummy, LogReg, XGB)."""

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return ``(n, 2)`` class probabilities; column 1 is P(churn)."""
        ...


@runtime_checkable
class Calibrator(Protocol):
    """Post-hoc map from raw positive-class probabilities to calibrated ones."""

    def transform(self, y_prob: Any) -> np.ndarray:
        """Return calibrated probabilities in ``[0, 1]``."""
        ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """HITL action mapping. Implementations must keep ``auto_action: none``."""

    def decide(self, prob: float, threshold: float, band: str) -> dict[str, Any]:
        """Return action, rationale, and auto_action fields."""
        ...
