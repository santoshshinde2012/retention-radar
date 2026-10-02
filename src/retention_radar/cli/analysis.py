"""CLI: ``python -m retention_radar.cli.analysis``."""

from retention_radar.evaluation.analysis import *  # noqa: F403
from retention_radar.evaluation.analysis import main

if __name__ == "__main__":
    main()
