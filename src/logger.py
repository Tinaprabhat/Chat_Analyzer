import logging
import os
from datetime import datetime
from src.config import LOG_FILE

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ChatPersonaRAG")


def log_update(message: str):
    """Log an update and print confirmation."""
    logger.info(message)
    print("update logged successfully")
