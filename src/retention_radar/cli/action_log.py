"""CLI: ``python -m retention_radar.cli.action_log``."""

from retention_radar.serving.action_log import *  # noqa: F403
from retention_radar.serving.action_log import main

if __name__ == "__main__":
    main()
