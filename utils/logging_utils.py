import logging
import os


def setup_logging(log_file=None):
    """Timestamped messages to the console (stderr, so they appear immediately) and optionally to a file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers, force=True,
    )
