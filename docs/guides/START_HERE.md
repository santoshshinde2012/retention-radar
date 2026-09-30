# Start here

Clone the repo, run one command, then read the results.

## 1. Clone and install

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## 2. Run the pipeline

```bash
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

This generates 8,000 renewals (seed 42), trains and calibrates the models, and writes
decision packets for the two worked examples. Maya should come out at calibrated 0.153,
band medium, action `limit_reset`. Arjun should come out at 0.023, band low, `no_action`.

Numbers to cite: [`models/metrics.json`](../../models/metrics.json). How to read them:
[results/BENCHMARKS.md](../../results/BENCHMARKS.md).

To check everything at once (reproduction, tests, CLI, API, UI, and the lakehouse path if
it is cloned beside this repo) without touching committed files:

```bash
make e2e-local
```

## 3. Walk one renewal day

```bash
make use-cases
```

Scores a day of T-7 renewals into an action queue, holds invalid records, imports what
the messaging tool sent, and reports lift against the holdout. Details:
[data/use_cases/README.md](../../data/use_cases/README.md).

Single pieces:

```bash
# Score a CSV into a ranked action queue (invalid rows go to *_rejected.csv)
python -m retention_radar.cli.batch_score --csv data/raw/renewals_t7.csv \
  --out artifacts/predictions/scores.csv

# Log what was done for one subscriber
python -m retention_radar.cli.action_log \
  --from-packet artifacts/maya_decision_packet.json \
  --executed-by lifecycle_tool --action-taken limit_reset

# Thin local API (no auth)
uvicorn retention_radar.serving.api:app --app-dir src --port 8000
curl -s localhost:8000/v1/churn/score -H 'content-type: application/json' \
  -d @data/raw/subscribers/maya.json
```

Action-log template and schema: [`configs/templates/action_log.csv`](../../configs/templates/action_log.csv) ·
[`configs/action_log.schema.json`](../../configs/action_log.schema.json).

## 4. Lakehouse path (optional)

Needs a [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse)
checkout that exports the v2 renewal contract:

```bash
./scripts/run_lakehouse_e2e.sh ../local-data-lakehouse
```

It trains under `artifacts/lakehouse_run/` only. See
[data-foundation-lakehouse.md](../data/data-foundation-lakehouse.md).

## 5. UI

```bash
make ui
```

Loads the committed models; nothing is trained on page load. There is no hosted demo yet;
deploy steps are in [DEPLOY_LATER.md](DEPLOY_LATER.md).

## What to read

1. [USE_CASE.md](../USE_CASE.md): the business problem and its sources.
2. [results/WORKED_EXAMPLES.md](../../results/WORKED_EXAMPLES.md): Maya and Arjun.
3. [results/BENCHMARKS.md](../../results/BENCHMARKS.md): the ladder, calibration, the policy.
4. [MODEL_CARD.md](../MODEL_CARD.md): intended use and limits.
