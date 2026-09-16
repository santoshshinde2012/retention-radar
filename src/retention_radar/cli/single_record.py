"""CLI: ``python -m retention_radar.cli.single_record``."""

from retention_radar.serving.packet import *  # noqa: F403
from retention_radar.serving.packet import main
from retention_radar.serving.policy import hitl_action, risk_band  # noqa: F401

if __name__ == "__main__":
    main()
