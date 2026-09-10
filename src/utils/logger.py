import logging
import os
from datetime import datetime

from src.config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Fungsi ini dipanggil di setiap file lain yang butuh mencatat log.
    Contoh pakai: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # cegah duplikat handler kalau fungsi ini dipanggil berkali-kali
    if not logger.handlers:
        log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d')}.log")

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # simpan log ke file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # tampilkan juga log di terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger