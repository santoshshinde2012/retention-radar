"""CLI: ``python -m retention_radar.cli.infer``."""

from retention_radar.serving.infer import *  # noqa: F403
from retention_radar.serving.infer import main
from retention_radar.serving.policy import risk_band  # noqa: F401

if __name__ == "__main__":
    main()
