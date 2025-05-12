import logging
import logging.handlers
from pathlib import Path


def setup_logging(
	log_file: str | None = None,
	level: int = logging.INFO,
	max_bytes: int = 10 * 1024 * 1024,  # 10MB
	backup_count: int = 5,
) -> None:
	"""
	Set up logging configuration for the application.

	Args:
	    log_file: Path to the log file. If None, logs will only go to console.
	    level: Logging level (default: INFO)
	    max_bytes: Maximum size of each log file in bytes
	    backup_count: Number of backup log files to keep
	"""  # noqa: E101
	# Create logs directory if it doesn't exist
	if log_file:
		log_path = Path(log_file)
		log_path.parent.mkdir(parents=True, exist_ok=True)

	# Create formatters
	console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

	# Create console handler
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(console_formatter)
	console_handler.setLevel(level)

	# Create root logger
	root_logger = logging.getLogger()
	root_logger.setLevel(level)
	root_logger.addHandler(console_handler)

	# Add file handler if log file is specified
	if log_file:
		file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
		file_handler.setFormatter(file_formatter)
		file_handler.setLevel(level)
		root_logger.addHandler(file_handler)

	# Set logging level for specific modules
	logging.getLogger("urllib3").setLevel(logging.WARNING)
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("httpcore").setLevel(logging.WARNING)
