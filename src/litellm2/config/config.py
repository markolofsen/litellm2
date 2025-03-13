import os
import logging
from litellm.caching.caching import LiteLLMCacheType
import litellm
import pathlib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Root directory and paths configuration
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def setup_logging():
    """Setup logging directory"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger.info(f"Logging directory configured at {LOGS_DIR}")


def save_response_log(response_data: dict, model_name: str) -> str:
    """Save response data to a log file.

    Args:
        response_data: Dictionary containing response data
        model_name: Name of the model used

    Returns:
        str: Path to the saved log file
    """
    setup_logging()

    # Sanitize model name for filename
    model_name = (model_name or 'unknown').replace('/', '_')

    # Generate filename with timestamp
    timestamp = datetime.now()
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{model_name}.json"
    filepath = os.path.join(LOGS_DIR, filename)

    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=2, default=str)

    logger.info(f"Response log saved to {filepath}")
    return filepath

# Configure caching


def setup_caching():
    litellm.enable_cache(
        type=LiteLLMCacheType.DISK,
        disk_cache_dir=os.path.join(ROOT_DIR, ".cache"),
    )
    logger.debug("LiteLLM cache configured - using disk cache")


# Default configuration
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = None
DEFAULT_CACHE_PROMPT = False
DEFAULT_max_budget = None
