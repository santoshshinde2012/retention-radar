"""PSI drift, τ-aligned bands and τ-based evaluation (2026-10 fixes)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from retention_radar import config
from retention_radar.evaluation.slices import slice_metrics_by_plan_tier
from retention_radar.serving.drift import compare_stats, population_stability_index
from retention_radar.training.train import compute_feature_stats, psi_reference_bins


def test_psi_is_zero_on_the_reference_and_large_on_a_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    bins = psi_reference_bins(ref)
    assert population_stability_index(ref.to_numpy(), bins["edges"], bins["ref_frac"]) == pytest.approx(0.0, abs=1e-9)
    shifted = rng.normal(1.0, 1, 5000)
    assert population_stability_index(shifted, bins["edges"], bins["ref_frac"]) > 0.25


def test_psi_bins_handle_ties_exactly():
    ref = pd.Series([0] * 700 + [1] * 200 + [2] * 100, dtype=float)
    bins = psi_reference_bins(ref)
    assert sum(bins["ref_frac"]) == pytest.approx(1.0)
    assert len(bins["ref_frac"]) == len(bins["edges"]) + 1
    assert population_stability_index(ref.to_numpy(), bins["edges"], bins["ref_frac"]) == pytest.approx(0.0, abs=1e-9)


def _frame(n, shift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(10 + shift, 2, n) for c in config.MODEL_FEATURE_COLUMNS})


def test_compare_stats_uses_psi_and_keeps_se_z_as_context():
    ref_df = _frame(3000)
    reference = compute_feature_stats(ref_df)
    cur_df = _frame(3000, seed=2)
    current = {c: {"count": len(cur_df), "mean": float(cur_df[c].mean()), "std": float(cur_df[c].std(ddof=0))}
               for c in cur_df.columns}
    values = {c: cur_df[c].to_numpy() for c in cur_df.columns}
    report = compare_stats(reference, current, current_values=values)
    assert report["method"] == "psi" and report["severity"] == "ok"
    row = report["features"][0]
    assert {"psi", "smd", "mean_z_se"} <= set(row)

    drifted = _frame(3000, shift=2.0, seed=3)
    cur2 = {c: {"count": len(drifted), "mean": float(drifted[c].mean()), "std": float(drifted[c].std(ddof=0))}
            for c in drifted.columns}
    rep2 = compare_stats(reference, cur2, current_values={c: drifted[c].to_numpy() for c in drifted.columns})
    assert rep2["severity"] == "severe" and rep2["n_major"] >= 1


def test_compare_stats_falls_back_to_smd_without_bins():
    ref_df = _frame(500)
    reference = {c: {k: v for k, v in s.items() if k != "psi_bins"} for c, s in compute_feature_stats(ref_df).items()}
    current = {c: {"count": 500, "mean": reference[c]["mean"] + reference[c]["std"], "std": reference[c]["std"]}
               for c in reference}
    rep = compare_stats(reference, current)
    assert rep["method"] == "smd" and rep["severity"] == "severe"


def test_slice_metrics_require_the_serving_threshold():
    assert "threshold" in inspect.signature(slice_metrics_by_plan_tier).parameters
    assert inspect.signature(slice_metrics_by_plan_tier).parameters["threshold"].default is inspect._empty
    block = slice_metrics_by_plan_tier([0, 1, 1, 0], [0.1, 0.4, 0.2, 0.3], ["free"] * 4, threshold=0.34)
    assert block["threshold"] == 0.34
    assert block["by_plan_tier"]["free"]["recall"] == 0.5
