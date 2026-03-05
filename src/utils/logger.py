import logging
import os
from datetime import datetime
import sys

def setup_logger(name='hier_bert_logger', log_file=None):
    """
    Sets up a production-grade logger that writes to both the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if instantiated multiple times
    if logger.hasHandlers():
        return logger

    # Log Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create File Handler (if logs directory exists)
    if log_file is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logging to {log_file}: {e}")

    return logger

# Default logger instance
logger = setup_logger()
