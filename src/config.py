"""Compatibility alias — canonical module is ``src.retention_radar.config``."""

from src.retention_radar.config import *  # noqa: F403
from src.retention_radar import config as _config
import sys

sys.modules[__name__] = _config
