"""
Structured logging via loguru.
Single setup call at import time.
In production: swap stdout sink for JSON → ELK / Datadog.
"""

import sys
from pathlib import Path
from loguru import logger

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Remove default loguru handler
logger.remove()

# Console — human readable during development
logger.add(
    sys.stdout,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    ),
    colorize=True,
)

# File — rotate daily, retain 7 days, compress
logger.add(
    "logs/arof_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="7 days",
    compression="gz",
    format="{time} | {level} | {name}:{line} | {message}",
)

__all__ = ["logger"]
