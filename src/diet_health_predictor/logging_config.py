"""
Logging Configuration
====================

Centralized logging setup for the entire application.
Configures logging based on environment settings.
"""

import logging
import sys
from pathlib import Path

from diet_health_predictor.config import get_settings


def setup_logging():
    """Configure logging for the application"""
    settings = get_settings()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Set log level based on settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_dir / f"{settings.environment}.log")
    file_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging configured for {settings.environment} environment")


# Configure logging when module is imported
setup_logging()
