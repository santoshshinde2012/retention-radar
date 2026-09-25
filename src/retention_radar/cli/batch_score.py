"""CLI: ``python -m retention_radar.cli.batch_score``."""

from retention_radar.serving.batch_score import *  # noqa: F403
from retention_radar.serving.batch_score import main

if __name__ == "__main__":
    raise SystemExit(main())
