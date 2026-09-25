"""CLI: ``python -m retention_radar.cli.drift_check``."""

from retention_radar.serving.drift import *  # noqa: F403
from retention_radar.serving.drift import main

if __name__ == "__main__":
    raise SystemExit(main())
