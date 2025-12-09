import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir='logs', log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance with both console and file handlers.

    This function ensures that a log directory exists and generates a log file
    named with the current date if no specific file is provided.

    Parameters
    ---
    name : str
        The name of the logger (usually __name__).
    log_dir : str, optional
        Directory to save log files. Defaults to "logs".
    log_file : str, optional
        Specific path for the log file. If None, a default name based on the date is generated.
    level : int, optional
        The logging level (e.g., logging.INFO). Defaults to logging.INFO.

    Returns
    ---
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(__file__).parent.parent.parent / 'logs'

    # File handler
    if log_file is None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # File name: logs/training_2023-10-27.log
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"training_{current_date}.log")
    else:
        # If user provides specific path, ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Disable propagate to prevent logs from being pushed to root logger (avoid duplicate printing if root logger also has handler)
    logger.propagate = False

    return logger
