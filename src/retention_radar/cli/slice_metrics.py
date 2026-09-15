"""CLI: ``python -m retention_radar.cli.slice_metrics``."""

from retention_radar.evaluation.slices import *  # noqa: F403
from retention_radar.evaluation.slices import main

if __name__ == "__main__":
    main()
