"""CLI shim: ``python -m src.ingest``."""

from src.retention_radar.data.ingest import *  # noqa: F403
from src.retention_radar.data.ingest import main

if __name__ == "__main__":
    main()
