"""CLI: ``python -m retention_radar.cli.hitl_log``."""

from retention_radar.serving.hitl_log import *  # noqa: F403
from retention_radar.serving.hitl_log import main

if __name__ == "__main__":
    main()
