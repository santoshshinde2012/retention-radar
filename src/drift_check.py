"""CLI shim: ``python -m src.drift_check``."""

from src.retention_radar.serving.drift import *  # noqa: F403
from src.retention_radar.serving.drift import main

if __name__ == "__main__":
    raise SystemExit(main())
