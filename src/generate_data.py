"""CLI shim: ``python -m src.generate_data``."""

from src.retention_radar.data.generate import *  # noqa: F403
from src.retention_radar.data.generate import main

if __name__ == "__main__":
    main()
