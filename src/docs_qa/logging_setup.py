"""One logging setup shared by the API server and the ingestion CLI."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_LEVEL = logging.INFO


def configure_logging() -> None:
    """Send the app's log lines to the console. Uvicorn only sets this up for its own loggers."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
