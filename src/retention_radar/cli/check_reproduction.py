"""CLI: ``python -m retention_radar.cli.check_reproduction``."""

from retention_radar.evaluation.reproduce import *  # noqa: F403
from retention_radar.evaluation.reproduce import main

if __name__ == "__main__":
    raise SystemExit(main())
