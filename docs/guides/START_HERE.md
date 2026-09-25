# Start here — Retention Radar teaching trilogy

Public FOSS path for the Medium / DET series. **Clone two public repos, run one synthetic command.** Do **not** clone any private article workspace.

## 1. Clone

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
git clone https://github.com/santoshshinde2012/local-data-lakehouse.git
cd retention-radar
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## 2. One synthetic command (published seed-42 ladder)

```bash
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Expect Santosh: raw ≈ **0.043** → calibrated ≈ **0.016** → band **low** → HITL **monitor** (`auto_action: none`).

Cite [`models/metrics.json`](../../models/metrics.json) · [results/BENCHMARKS.md](../../results/BENCHMARKS.md).

Verify the whole trilogy locally in one command (reproduce + tests + CLI/API/UI + lakehouse, committed files untouched):

```bash
make e2e-local   # picks up ../local-data-lakehouse automatically
```

## 3. Optional lakehouse gold path

Needs the lakehouse checkout beside this repo (or pass the path):

```bash
./scripts/run_lakehouse_e2e.sh ../local-data-lakehouse
```

Lakehouse train/eval writes **only** under `artifacts/lakehouse_run/` (`RETENTION_RADAR_ARTIFACT_DIR`). Committed `models/` and `docs/MODEL_CARD.md` stay the published synthetic serve bundle. Dual-world cite: [`results/lakehouse-e2e-summary.json`](../../results/lakehouse-e2e-summary.json).


## 3b. Optional FOSS production-shaped path (local)

Batch-score gold features with the committed seed-42 serve bundle (HITL only — `auto_action: none`):

```bash
python -m retention_radar.cli.batch_score \
  --csv data/external/churn_user_features.csv \
  --out artifacts/predictions/scores.csv
```

Tiny fixture (tests): `tests/fixtures/batch/tiny_features.csv`.

Append a HITL review-log row (predict → act):

```bash
python -m retention_radar.cli.hitl_log \
  --from-packet artifacts/santosh_decision_packet.json \
  --reviewer you --action-taken monitor --notes "ok"
```

Template + schema: [`configs/templates/hitl_review_log.csv`](../../configs/templates/hitl_review_log.csv) · [`configs/hitl_review_log.schema.json`](../../configs/hitl_review_log.schema.json).

Close the loop once labels arrive (outcome write-back — observed churn per band / action; descriptive only):

```bash
python -m retention_radar.cli.hitl_outcomes --labels data/raw/users.csv
# → artifacts/hitl_outcomes.csv + artifacts/hitl_outcomes.json
```

Thin local FastAPI (teaching-only, **no auth**). Preferred route `POST /v1/churn/score` (conceptual alias `POST /v1/churn:score`):

```bash
uvicorn retention_radar.serving.api:app --app-dir src --port 8000
# curl -s localhost:8000/v1/churn/score -H 'content-type: application/json' -d @data/raw/santosh_shinde.json
```

## 4. UI / live demo

```bash
make ui   # streamlit run app/streamlit_app.py — loads committed models only (no fit on load)
```

**Live demo:** TBD — Streamlit Community Cloud / HF Space  
(Exact deploy steps when ready: [DEPLOY_LATER.md](DEPLOY_LATER.md); full free-platform map: [e2e-free-platforms.md](e2e-free-platforms.md))

## 5. Articles

Narrative articles live in a **separate internal** workspace. This public repo is code + benchmarks + results only — never a private clone instruction for readers.

## See also (related teaching repos)

- [medallion-write-back-loop](https://github.com/santoshshinde2012/medallion-write-back-loop) — medallion write-back teaching loop (not wired into this E2E)
- [churn-vs-risk-poc](https://github.com/santoshshinde2012/churn-vs-risk-poc) — churn vs risk POC (not wired into this E2E)
