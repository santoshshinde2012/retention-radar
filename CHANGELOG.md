# Changelog

## 0.1.0 — 2026-09-15

- Initial public teaching repo: full E2E Retention Radar codebase (train → calibrate → serve → Santosh HITL).
- Package layout under `src/retention_radar/` with `src/*.py` shims.
- Committed seed-42 serve bundle in `models/`.
- Engineering docs: `ARCHITECTURE.md` (SOLID), `MODEL_CARD.md`, minimal `docs/`.
- Dual-path data: synthetic generator + [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) gold sync.
- CI: GitHub Actions synthetic smoke + pytest.
