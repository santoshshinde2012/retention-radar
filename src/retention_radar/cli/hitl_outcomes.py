"""CLI: ``python -m retention_radar.cli.hitl_outcomes``."""

from retention_radar.serving.outcomes import *  # noqa: F403
from retention_radar.serving.outcomes import main

if __name__ == "__main__":
    main()
