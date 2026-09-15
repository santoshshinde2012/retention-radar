## Unreleased

- Restructure to production folder layout: only `src/retention_radar/` under `src/`, CLI under `retention_radar.cli`, nested `docs/{guides,data,case-study}/`, runtime-only `artifacts/`.
- Add CatBoost (default) to the honest bake-off ladder; keep calibrated Optuna XGBoost as serving hero.
- Document algorithm landscape (IN vs DEFER: TabPFN, survival, conformal, uplift, …).

# Changelog

## 0.1.0 — 2026-09-15

- Initial public teaching repo: full E2E Retention Radar codebase (train → calibrate → serve → Santosh HITL).
- Package layout under `src/retention_radar/` with CLI under `src/retention_radar/cli/`.
- Committed seed-42 serve bundle in `models/`.
- Engineering docs: `ARCHITECTURE.md` (SOLID), `MODEL_CARD.md`, minimal `docs/`.
- Dual-path data: synthetic generator + [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) gold sync.
- CI: GitHub Actions synthetic smoke + pytest.
