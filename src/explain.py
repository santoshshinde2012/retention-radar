"""CLI shim: ``python -m src.explain``."""

from src.retention_radar.serving.explain import *  # noqa: F403
from src.retention_radar.serving.explain import main

if __name__ == "__main__":
    main()
