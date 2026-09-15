"""Load and lightly validate the users table (synthetic or lakehouse gold export)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from retention_radar import config


def _required_columns() -> list[str]:
    return config.ID_COLUMNS + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]


def resolve_users_csv(path: Path | None = None) -> Path:
    """Pick lakehouse export vs synthetic users.csv based on CHURN_DATA_SOURCE."""
    if path is not None:
        return Path(path)
    source = (config.CHURN_DATA_SOURCE or "auto").lower()
    lake = config.LAKEHOUSE_FEATURES_CSV
    synth = config.USERS_CSV
    if source == "lakehouse":
        return lake
    if source == "synthetic":
        return synth
    # auto
    return lake if lake.exists() else synth


def resolve_santosh_json(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    source = (config.CHURN_DATA_SOURCE or "auto").lower()
    lake = config.LAKEHOUSE_SANTOSH_JSON
    synth = config.SANTOSH_JSON
    if source == "lakehouse":
        return lake
    if source == "synthetic":
        return synth
    return lake if lake.exists() else synth


def load_users(path: Path | None = None) -> pd.DataFrame:
    """Read users table and run basic validation checks."""
    csv_path = resolve_users_csv(path)
    if not csv_path.exists():
        hint = (
            "Run: python -m retention_radar.cli.generate_data"
            if csv_path == config.USERS_CSV
            else "Copy lakehouse exports into data/external/ (see scripts/sync_lakehouse_exports.sh)"
        )
        raise FileNotFoundError(f"Missing {csv_path}. {hint}")

    df = pd.read_csv(csv_path)
    # Drop lake-only metadata if someone passes a raw gold dump
    drop_cols = [c for c in ("city", "feature_as_of", "built_at") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    validate_users(df)
    return df


def validate_users(df: pd.DataFrame) -> None:
    """Raise ValueError if the frame looks broken."""
    missing = [c for c in _required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("users dataframe is empty")

    nan_counts = df[_required_columns()].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        raise ValueError(
            "NaNs in required columns (synthetic/lakehouse gold has none by design; "
            f"fail loud): {nan_cols.to_dict()}"
        )

    if df["user_id"].duplicated().any():
        raise ValueError("Duplicate user_id values found")

    bad_plans = set(df["plan_tier"].unique()) - set(config.PLAN_TIER_ORDER)
    if bad_plans:
        raise ValueError(f"Unknown plan_tier values: {bad_plans}")

    for col in (
        "failed_requests_rate",
        "weekend_usage_ratio",
        "feature_adoption_score",
        "seat_utilization",
    ):
        if not df[col].between(0, 1).all():
            raise ValueError(f"{col} must be in [0, 1]")

    if not df["engagement_trend"].between(0, 5).all():
        raise ValueError("engagement_trend must be in [0, 5]")

    if (df["spend_usd_last_30d"] < 0).any():
        raise ValueError("spend_usd_last_30d must be >= 0")

    if (df["days_until_renewal"] < 0).any():
        raise ValueError("days_until_renewal must be >= 0")

    if (df["agent_runs_last_30d"] < 0).any():
        raise ValueError("agent_runs_last_30d must be >= 0")

    if (df["ide_plugin_sessions_last_30d"] < 0).any():
        raise ValueError("ide_plugin_sessions_last_30d must be >= 0")

    if not set(df[config.TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("churned must be 0/1")

    rate = float(df[config.TARGET_COLUMN].mean())
    if rate < 0.05 or rate > 0.55:
        raise ValueError(f"Unexpected churn rate {rate:.3f}; regenerate data?")


def sync_lakehouse_exports(
    source_dir: Path,
    dest_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Copy churn_user_features.csv + santosh_inference_record.json into data/external/."""
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir) if dest_dir is not None else config.EXTERNAL_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_csv = source_dir / "churn_user_features.csv"
    src_json = source_dir / "santosh_inference_record.json"
    if not src_csv.exists() or not src_json.exists():
        raise FileNotFoundError(
            f"Expected {src_csv.name} and {src_json.name} under {source_dir}"
        )
    dest_csv = dest_dir / "churn_user_features.csv"
    dest_json = dest_dir / "santosh_inference_record.json"
    shutil.copy2(src_csv, dest_csv)
    shutil.copy2(src_json, dest_json)
    return dest_csv, dest_json


def main() -> None:
    csv_path = resolve_users_csv()
    df = load_users(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"CHURN_DATA_SOURCE={config.CHURN_DATA_SOURCE}")
    print(f"Churn rate: {df[config.TARGET_COLUMN].mean():.3f}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
