"""
Logger utility: menampilkan log di terminal (stdout) dan file.
Berguna untuk debugging di Colab dan lokal.
"""
import logging
import sys
from pathlib import Path

# Path log file (di data/ agar konsisten dengan struktur project)
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "app.log"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_logger_configured = False


def setup_logger(name: str = "app", level=logging.INFO) -> logging.Logger:
    """Setup logger dengan output ke terminal dan file."""
    global _logger_configured

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Handler 1: Terminal (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Handler 2: File (untuk Colab & arsip)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _logger_configured = True
    return logger


def get_logger(name: str = "app") -> logging.Logger:
    """Ambil logger. Auto-setup jika belum dikonfigurasi."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def get_log_path() -> Path:
    """Path file log untuk dibaca di Colab."""
    return LOG_FILE
