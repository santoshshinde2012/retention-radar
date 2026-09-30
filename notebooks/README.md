# notebooks/

Exploration only. Import the installed package; code that matters goes in `src/retention_radar/`.

| Notebook | Purpose |
|----------|---------|
| [`01_explore_renewals.ipynb`](01_explore_renewals.ipynb) | Load `renewals_t7.csv` and `renewals_all.csv`, count outcomes and routes, lapse rate by tenure and by cap hits, score Maya and Arjun with the committed bundle |

Run `CHURN_DATA_SOURCE=synthetic python -m retention_radar.cli.generate_data` first; the
CSVs are not committed. Published charts are in `results/plots/`.
