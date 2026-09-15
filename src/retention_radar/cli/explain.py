"""CLI: ``python -m retention_radar.cli.explain``."""

from retention_radar.serving.explain import *  # noqa: F403
from retention_radar.serving.explain import main

if __name__ == "__main__":
    main()
