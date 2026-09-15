"""CLI shim: ``python -m src.benchmark``."""

from src.retention_radar.evaluation.benchmark import *  # noqa: F403
from src.retention_radar.evaluation.benchmark import main

if __name__ == "__main__":
    main()
