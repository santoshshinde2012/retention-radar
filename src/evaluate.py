"""CLI shim: ``python -m src.evaluate``."""

from src.retention_radar.evaluation.evaluate import *  # noqa: F403
from src.retention_radar.evaluation.evaluate import main

if __name__ == "__main__":
    main()
