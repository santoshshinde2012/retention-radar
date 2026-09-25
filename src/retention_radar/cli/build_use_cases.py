"""CLI: ``python -m retention_radar.cli.build_use_cases``."""

from retention_radar.data.use_cases import *  # noqa: F403
from retention_radar.data.use_cases import main

if __name__ == "__main__":
    raise SystemExit(main())
