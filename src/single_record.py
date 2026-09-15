"""CLI shim: ``python -m src.single_record``."""

from src.retention_radar.serving.packet import *  # noqa: F403
from src.retention_radar.serving.policy import hitl_action, risk_band  # noqa: F401
from src.retention_radar.serving.packet import main

if __name__ == "__main__":
    main()
