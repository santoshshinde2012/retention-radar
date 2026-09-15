"""CLI shim: ``python -m src.docs_gen``."""

from src.retention_radar.docs_gen import *  # noqa: F403
from src.retention_radar.docs_gen import main

if __name__ == "__main__":
    main()
