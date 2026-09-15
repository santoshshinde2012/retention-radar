"""CLI: ``python -m retention_radar.cli.evaluate``."""

from retention_radar.evaluation.evaluate import *  # noqa: F403
from retention_radar.evaluation.evaluate import main

if __name__ == "__main__":
    main()
