"""Shared train/val/test split helper (seed 42, stratified)."""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.retention_radar import config


def stratified_train_val_test(X, y):
    """Return X_train, X_val, X_test, y_train, y_val, y_test."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    relative_val = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val,
        random_state=config.RANDOM_SEED,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
