"""
Advanced AI agent with various tools and categories
"""

# Import standard libraries
from src.litellm2.agents import SimpleAgent
from smolagents.tools import tool
import os
import sys
import datetime
import random
import json
import math
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================
# TOOL DEFINITIONS
# ==============================================

# ----- Date and Time Tools -----


@tool
def date_time(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time in the specified format.

    Args:
        format_string: Format string for the date/time (default: "%Y-%m-%d %H:%M:%S")
    """
    now = datetime.datetime.now()
    return now.strftime(format_string)


@tool
def timestamp() -> str:
    """
    Get the current Unix timestamp (seconds since January 1, 1970).
    """
    return str(int(datetime.datetime.now().timestamp()))

# ----- Math Tools -----


@tool
def random_number(min_val: str = "1", max_val: str = "100") -> str:
    """
    Generate a random number between min and max (inclusive).

    Args:
        min_val: Minimum value (default: "1")
        max_val: Maximum value (default: "100")
    """
    try:
        min_num = int(min_val)
        max_num = int(max_val)
        if min_num > max_num:
            min_num, max_num = max_num, min_num
        return str(random.randint(min_num, max_num))
    except ValueError:
        return "Error: Please provide valid integer values for min and max."


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate
    """
    # Define a safe subset of math functions
    safe_dict = {
        'abs': abs, 'round': round,
        'min': min, 'max': max,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'sqrt': math.sqrt, 'pow': math.pow,
        'pi': math.pi, 'e': math.e
    }

    try:
        # Use a restricted global environment for basic operations
        return str(eval(expression, {"__builtins__": {}}, safe_dict))
    except Exception as e:
        return f"Error calculating expression: {e}"

# ----- Text Processing Tools -----


@tool
def word_count(text: str) -> str:
    """
    Count the number of words in the text.

    Args:
        text: Text to count words in
    """
    words = text.split()
    return f"Word count: {len(words)}"


@tool
def reverse_text(text: str) -> str:
    """
    Reverse the input text.

    Args:
        text: Text to reverse
    """
    return text[::-1]


@tool
def text_stats(text: str) -> str:
    """
    Get statistics about the text.

    Args:
        text: Text to analyze
    """
    stats = {
        "characters": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()) or 1,
        "uppercase_letters": sum(1 for c in text if c.isupper()),
        "lowercase_letters": sum(1 for c in text if c.islower()),
        "digits": sum(1 for c in text if c.isdigit()),
        "spaces": sum(1 for c in text if c.isspace()),
    }
    return json.dumps(stats, indent=2)

# ----- Utility Tools -----


@tool
def echo(text: str) -> str:
    """
    Simply echo back the input text.

    Args:
        text: Text to echo back
    """
    return text


@tool
def help_info(topic: str = "") -> str:
    """
    Get help information about a topic.

    Args:
        topic: Topic to get help for (leave empty for general help)
    """
    topics = {
        "": "Available help topics: tools, date, math, text, agent",
        "tools": "Tools are functions that can be called by name. Use 'help' to see all available tools.",
        "date": "Date tools include date_time and timestamp for working with date and time.",
        "math": "Math tools include calculate and random_number for calculations.",
        "text": "Text tools include word_count, reverse_text, and text_stats for text processing.",
        "agent": "This is an AI agent equipped with tools to help with various tasks.",
    }

    return topics.get(topic.lower(), f"Unknown topic: {topic}")

# ----- API/External Tools -----


@tool
def weather_api(city: str) -> str:
    """
    Get current weather information for a city.

    Args:
        city: City name for weather information
    """
    # Simulate weather API call with simple response
    try:
        weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy"]
        temperatures = range(5, 35)

        # Deterministic but random-looking result for same city
        city_hash = sum(ord(c) for c in city.lower())
        random.seed(city_hash)

        condition = weather_conditions[city_hash % len(weather_conditions)]
        temp = random.choice(temperatures)
        humidity = random.randint(30, 90)

        return (
            f"Weather for {city}:\n"
            f"Temperature: {temp}°C\n"
            f"Condition: {condition}\n"
            f"Humidity: {humidity}%"
        )
    except Exception as e:
        logger.error(f"Error getting weather for {city}: {e}")
        return f"Unable to get weather information for {city}."

# ==============================================
# AGENT IMPLEMENTATION
# ==============================================


class AdvancedAgent(SimpleAgent):
    """An extended version of SimpleAgent with more tools"""

    def __init__(self):
        """
        Initialize the advanced agent with pre-installed tools

        Args:
            api_base: API base URL for the language model
            model_id: Model ID for the language model
        """
        # Initialize with empty tools list
        super().__init__()

        # Add custom tools using the add_tools method
        self.add_tools([
            date_time,
            timestamp,
            random_number,
            calculate,
            word_count,
            reverse_text,
            text_stats,
            echo,
            help_info,
            weather_api
        ])

    def run_demo(self):
        """
        Run a demonstration of the agent with its tools
        """
        # Add name and description attributes
        self.name = "Advanced Agent"
        self.description = "An AI agent with various tools for data manipulation, math operations, and more."

        # Launch Gradio UI
        self.launch_gradio_ui()


if __name__ == "__main__":
    # When run directly, create an agent and run demo
    agent = AdvancedAgent()
    agent.run_demo()
