"""CLI shim: ``python -m src.infer``."""

from src.retention_radar.serving.infer import *  # noqa: F403
from src.retention_radar.serving.policy import risk_band  # noqa: F401
from src.retention_radar.serving.infer import main

if __name__ == "__main__":
    main()
