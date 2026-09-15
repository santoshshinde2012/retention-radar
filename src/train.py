"""CLI shim: ``python -m src.train``."""

from src.retention_radar.training.train import *  # noqa: F403
from src.retention_radar.training.train import main

if __name__ == "__main__":
    main()
