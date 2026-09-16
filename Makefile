.PHONY: setup run run-lakehouse test infer ui api docs-results

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt && pip install -e .

run:
	CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh

run-lakehouse:
	./scripts/run_lakehouse_e2e.sh

test:
	CHURN_DATA_SOURCE=synthetic PYTHONPATH=src pytest -q

infer:
	PYTHONPATH=src python -m retention_radar.cli.single_record --user santosh --out artifacts/santosh_decision_packet.json

ui:
	PYTHONPATH=src streamlit run app/streamlit_app.py

api:
	PYTHONPATH=src uvicorn retention_radar.serving.api:app --host 127.0.0.1 --port 8000

docs-results:
	mkdir -p results/plots
	cp artifacts/roc_curve.png artifacts/pr_curve.png artifacts/calibration_curve.png artifacts/confusion_matrix.png artifacts/threshold_f1.png results/plots/
