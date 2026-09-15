"""CLI: ``python -m retention_radar.cli.generate_data``."""

from retention_radar.data.generate import *  # noqa: F403
from retention_radar.data.generate import main

if __name__ == "__main__":
    main()
