import logging
import colorlog
from typing import Optional


class AILogger:
    _instance: Optional['AILogger'] = None
    _logger: Optional[logging.Logger] = None
    _verbose: bool = False

    _allow_debug: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AILogger, cls).__new__(cls)
            cls._setup_logger()
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        # Create logger
        cls._logger = logging.getLogger("ai_module")
        cls._logger.setLevel(logging.INFO)

        # Create console handler with color formatting
        console_handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(formatter)
        cls._logger.addHandler(console_handler)

    @classmethod
    def set_verbose(cls, verbose: bool):
        cls._verbose = verbose
        cls._allow_debug = verbose  # Enable debug messages when verbose mode is on
        if cls._logger:
            cls._logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    @classmethod
    def debug(cls, message: str):
        if cls._allow_debug and cls._logger:
            cls._logger.debug(message)

    @classmethod
    def info(cls, message: str):
        if cls._verbose and cls._logger:
            cls._logger.info(message)

    @classmethod
    def warning(cls, message: str):
        if cls._logger:
            cls._logger.warning(message)

    @classmethod
    def error(cls, message: str):
        if cls._logger:
            cls._logger.error(message)


# Create global logger instance
logger = AILogger()
