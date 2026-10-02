# Checklist for changes

## Reproducibility

- [ ] Keep `RANDOM_SEED = 42`, or say why it changed.
- [ ] Use `./scripts/run_all.sh` so generate, train, evaluate and packets stay in step.
- [ ] Cite `models/metrics.json`, not numbers copied from prose.
- [ ] Run `make reproduce` after anything that could change the model.
- [ ] Run tests with `CHURN_DATA_SOURCE=synthetic` so a local lakehouse export cannot change them.

## The label

- [ ] `churned` means a voluntary lapse at this renewal.
- [ ] Failed cards go to dunning and stay out of the model table.
- [ ] Cancels scheduled before T-7 go to the cancel flow and stay out of the model table.
- [ ] Every feature must exist in the live record at T-7. Nothing known only after the renewal.

## Evaluation

- [ ] Optuna, the calibrator and τ use validation only. Test is read once.
- [ ] Report the whole ladder, including Dummy and logistic regression.
- [ ] An AUC near 1.0 on this data means a leak. Find it before going further.
- [ ] Unknown `plan_tier` (including `teams`) is held, never silently encoded.

## Calibration and threshold

- [ ] Raw `predict_proba` is inflated by `scale_pos_weight`. Bands and the policy use the calibrated value.
- [ ] If you change the calibrator, check that no subscriber gets exactly 0.000 and that the queue does not tie.
- [ ] τ is the best-F1 threshold on validation. Say so wherever it is used.

## Policy

- [ ] `auto_action` stays `none` in `serving/policy.py`. The service suggests; a lifecycle tool sends.
- [ ] Playbook effects in `config.PLAYBOOKS` are assumptions. Do not present expected values or simulated lift as results.
- [ ] Keep the holdout. Without it, no playbook can be measured.
- [ ] Only `personal_email` has a person in it, and only for Ultra.
- [ ] SHAP drivers describe the model, not the subscriber's reasons, and not what an action would change.

## Drift

- [ ] `python -m retention_radar.cli.drift_check` compares the active table with `models/feature_stats.json`. Use `--strict` to fail a pipeline.
- [ ] Do not compare a lakehouse export with a leftover synthetic `data/raw/renewals_t7.csv`.

## Synthetic data

- [ ] README, model card and UI say the data is synthetic with no real PII.
- [ ] No ROI or fairness claims. Slices by `plan_tier` are for inspection only.

## Train and serve

- [ ] Serving loads the bundle; it never fits an encoder or runs Optuna.
- [ ] Never train inside Streamlit.
- [ ] The FastAPI app has no auth. Keep it on localhost.

## Lakehouse

- [ ] The published numbers are the synthetic run. A lakehouse run writes under `artifacts/lakehouse_run/`; do not copy it into `models/`.
- [ ] The lakehouse export must match the v2 contract (24 fields + `churned`, `hero_inference_record.json`).

## Docs and CI

- [ ] `pytest -q` passes before a PR.
- [ ] After a change to metrics, run `python -m retention_radar.cli.docs_gen` and update `results/`. Not after a lakehouse run.
- [ ] Engineering docs go under `docs/`, analysis under `results/`.

Related: [architecture.md](../architecture.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) ·
[data-foundation-lakehouse.md](../data/data-foundation-lakehouse.md)
