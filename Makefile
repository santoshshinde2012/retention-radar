.PHONY: setup run run-lakehouse test lint infer ui api use-cases build-use-cases reproduce e2e-local docs-results

# Python 3.12+ is required (committed bundle trained with XGBoost 3.4.1).
PYTHON ?= $(shell command -v python3.12 || command -v python3)
# Use the project venv when it exists, so targets work without `source .venv/bin/activate`.
BIN := $(if $(wildcard .venv/bin/python),.venv/bin/,)
PY := $(BIN)python
export PYTHONPATH := src

setup:
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12+ required, got " + sys.version.split()[0] + " — run: make setup PYTHON=python3.12")'
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . ruff

run:
	CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh

run-lakehouse:
	./scripts/run_lakehouse_e2e.sh $(LAKEHOUSE_ROOT)

test:
	CHURN_DATA_SOURCE=synthetic $(PY) -m pytest -q

lint:
	$(PY) -m ruff check src/ tests/ app/

infer:
	$(PY) -m retention_radar.cli.single_record --user santosh --out artifacts/santosh_decision_packet.json

ui:
	$(PY) -m streamlit run app/streamlit_app.py

api:
	$(PY) -m uvicorn retention_radar.serving.api:app --host 127.0.0.1 --port 8000

# Walk the service on data/use_cases/: queue → packets → held records → reviews → outcomes.
use-cases:
	PYTHON=$(PY) ./scripts/run_use_cases.sh

# Rebuild data/use_cases/ from seed-42 data/raw/users.csv + the committed bundle.
build-use-cases:
	CHURN_DATA_SOURCE=synthetic $(PY) -m retention_radar.cli.generate_data
	CHURN_DATA_SOURCE=synthetic $(PY) -m retention_radar.cli.build_use_cases

# Retrain into artifacts/repro/ (committed models/ untouched) and diff against models/metrics.json.
reproduce:
	RETENTION_RADAR_ARTIFACT_DIR=artifacts/repro CHURN_DATA_SOURCE=synthetic N_USERS=5000 N_OPTUNA_TRIALS=20 ./scripts/run_all.sh
	$(PY) -m retention_radar.cli.check_reproduction --artifact-dir artifacts/repro

# Everything end to end (reproduce, tests, CLI/API/UI surfaces, optional lakehouse); see scripts/run_local_e2e.sh.
e2e-local:
	PYTHON=$(PY) ./scripts/run_local_e2e.sh

docs-results:
	mkdir -p results/plots
	cp artifacts/roc_curve.png artifacts/pr_curve.png artifacts/calibration_curve.png artifacts/confusion_matrix.png artifacts/threshold_f1.png results/plots/
