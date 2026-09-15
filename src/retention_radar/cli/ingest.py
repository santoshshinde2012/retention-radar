"""CLI: ``python -m retention_radar.cli.ingest``."""

from retention_radar.data.ingest import *  # noqa: F403
from retention_radar.data.ingest import main

if __name__ == "__main__":
    main()
