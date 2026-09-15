"""Feature preparation: encode plan_tier and build X / y matrices."""

from __future__ import annotations

import pandas as pd

from src.retention_radar import config


def encode_plan_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Map plan_tier string → ordinal plan_tier_code (free=0 … enterprise=3)."""
    out = df.copy()
    unknown = set(out["plan_tier"].unique()) - set(config.PLAN_TIER_MAP)
    if unknown:
        raise ValueError(f"Unknown plan_tier values: {unknown}")
    out["plan_tier_code"] = out["plan_tier"].map(config.PLAN_TIER_MAP).astype(int)
    return out


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for sklearn / XGBoost."""
    encoded = encode_plan_tier(df)
    X = encoded[config.MODEL_FEATURE_COLUMNS].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any():
        raise ValueError(
            f"NaNs in features after encoding: {X.isna().sum()[X.isna().sum() > 0]}"
        )
    y = encoded[config.TARGET_COLUMN].astype(int)
    return X, y


def row_to_feature_frame(row: dict | pd.Series) -> pd.DataFrame:
    """Convert a single user dict/Series into a 1-row feature DataFrame."""
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)
    df = pd.DataFrame([data])
    encoded = encode_plan_tier(df)
    X = encoded[config.MODEL_FEATURE_COLUMNS].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any():
        nan_cols = X.isna().sum()[X.isna().sum() > 0]
        raise ValueError(
            "NaNs in serve features after encoding (fail loud — "
            f"synthetic generator emits no missingness): {nan_cols.to_dict()}"
        )
    return X


class DefaultFeatureTransformer:
    """``FeatureTransformer`` adapter over the module-level helpers."""

    def prepare_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return prepare_xy(df)

    def row_to_feature_frame(self, row: dict | pd.Series) -> pd.DataFrame:
        return row_to_feature_frame(row)
