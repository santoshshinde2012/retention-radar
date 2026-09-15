"""CLI: ``python -m retention_radar.cli.benchmark``."""

from retention_radar.evaluation.benchmark import *  # noqa: F403
from retention_radar.evaluation.benchmark import main

if __name__ == "__main__":
    main()
