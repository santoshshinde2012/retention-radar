.PHONY: setup run run-lakehouse test infer ui docs-results

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh

run-lakehouse:
	./scripts/run_lakehouse_e2e.sh

test:
	CHURN_DATA_SOURCE=synthetic PYTHONPATH=. pytest -q

infer:
	PYTHONPATH=. python -m src.single_record --user santosh --out artifacts/santosh_decision_packet.json

ui:
	PYTHONPATH=. streamlit run app/streamlit_app.py

docs-results:
	mkdir -p docs/results
	cp artifacts/roc_curve.png artifacts/pr_curve.png artifacts/calibration_curve.png artifacts/confusion_matrix.png artifacts/threshold_f1.png docs/results/
