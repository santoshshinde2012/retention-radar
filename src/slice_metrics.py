"""CLI shim: ``python -m src.slice_metrics``."""

from src.retention_radar.evaluation.slices import *  # noqa: F403
from src.retention_radar.evaluation.slices import main

if __name__ == "__main__":
    main()
