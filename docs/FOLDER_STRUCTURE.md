# Folder structure — Retention Radar

Teaching / ML use-case layout. **Code + benchmarks + results analysis** live here (public code home).  
Articles are authored separately (**internal**); readers should not be pointed at an articles GitHub repo.  
Data foundation / SoR → [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse).

```text
retention-radar/
├── README.md                 # use-case overview + repo map
├── LICENSE, Makefile, pyproject.toml, requirements.txt, runtime.txt, .gitignore
├── docs/
│   ├── ARCHITECTURE.md       # SOLID / train≠serve map
│   ├── MODEL_CARD.md         # metrics + intended use (from metrics.json)
│   ├── FOLDER_STRUCTURE.md   # this file
│   ├── GETTING_STARTED.md
│   ├── BEST_PRACTICES.md, data-dictionary.md, e2e-free-platforms.md
│   ├── data-foundation-lakehouse.md, santosh-case-study.md, single-record-checklist.md
├── src/retention_radar/      # SOLID package (data, features, training, evaluation, serving)
├── src/*.py                  # thin shims (`python -m src.train`, …)
├── app/streamlit_app.py      # serve-only UI
├── scripts/                  # run_all, sync_lakehouse, run_lakehouse_e2e
├── schemas/
├── models/                   # committed seed-42 joblibs + metrics.json + feature_*
├── data/raw/, data/external/
├── results/                  # analysis of results (benchmarks + Santosh + plots)
│   ├── README.md, BENCHMARKS.md, SANTOSH_ANALYSIS.md
│   ├── plots/                # ROC/PR/calibration/confusion/threshold
│   └── *.json                # sample packet + optional lakehouse summary
├── artifacts/                # runtime dumps (gitignored except .gitkeep)
├── tests/
└── .github/workflows/
```

## Folder purposes

| Path | Purpose |
|------|---------|
| `src/retention_radar/` | Implementation packages (SOLID boundaries) |
| `src/*.py` | CLI/compat shims so `python -m src.*` stays stable |
| `app/` | Streamlit — **serve-only** |
| `scripts/` | E2E + lakehouse sync helpers (`make run` / `make run-lakehouse`) |
| `models/` | Seed-42 serve bundle — **commit** for Cloud |
| `results/` | Committed analysis + charts (not runtime) |
| `artifacts/` | Local evaluate / infer dumps |
| `docs/` | Engineering docs + model card + architecture |
| `data/external/` | Lakehouse gold sync (CSVs gitignored) |

**Charts:** `artifacts/` is what evaluate writes. `results/plots/` is the committed pack. Refresh with `make docs-results`.

More: [GETTING_STARTED.md](GETTING_STARTED.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [../results/README.md](../results/README.md)
