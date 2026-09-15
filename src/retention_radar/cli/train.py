"""CLI: ``python -m retention_radar.cli.train``."""

from retention_radar.training.train import *  # noqa: F403
from retention_radar.training.train import main

if __name__ == "__main__":
    main()
