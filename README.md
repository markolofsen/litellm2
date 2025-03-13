![LiteLLM2](https://raw.githubusercontent.com/markolofsen/litellm2/main/assets/cover.png)

# LiteLLM2 🚀

A powerful AI agent framework with structured Pydantic response handling and LLM integration capabilities.

## Overview 🔍

LiteLLM2 is built on top of [litellm](https://pypi.org/project/litellm/) and focuses on **typesafe, structured responses** through Pydantic models. Key features:

- **Structured Pydantic responses** ✅
- Flexible LLM client integration 🔌
- Budget management and caching 💰
- Advanced agent system with tools 🛠️

### Tech Stack 🔋

- **[Pydantic](https://docs.pydantic.dev/)**: Type-safe data handling
- **[LiteLLM](https://litellm.ai/)**: Core LLM routing
- **[OpenRouter](https://openrouter.ai/)**: Default model provider

## Installation 📦

```bash
pip install litellm2
```

Set up your API key:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

## Configuration 🔧

LiteLLM2 uses a flexible configuration system through the `Request` class:

```python
from litellm2 import Request

config = Request(
    # Model settings
    model="openrouter/openai/gpt-4o-mini",  # Model identifier
    answer_model=YourModel,                  # Pydantic model for responses (required)
    temperature=0.7,                         # Response randomness (0.0-1.0)
    max_tokens=500,                          # Maximum response length

    # Performance
    online=True,                            # Enable web search
    cache_prompt=True,                      # Cache identical prompts
    max_budget=0.05,                      # Maximum cost per request

    # Debug options
    verbose=True,                           # Detailed output
    logs=False                              # Enable logging
)
```

## Message Types 💬

LiteLLM2 supports 4 types of messages with automatic formatting for any data type:

```python
# 1. System Messages - Set AI's behavior and context
client.msg.add_message_system("You are a helpful assistant.")
client.msg.add_message_system({
    "role": "expert",
    "expertise": ["python", "data analysis"]
})

# 2. User Messages - Send queries or inputs
client.msg.add_message_user("Analyze this data")
client.msg.add_message_user({
    "query": "analyze trends",
    "metrics": ["users", "revenue"]
})

# 3. Assistant Messages - Add AI responses or context
client.msg.add_message_assistant("Based on the data...")
client.msg.add_message_assistant([
    "Point 1: Growth is steady",
    "Point 2: Conversion improved"
])

# 4. Block Messages - Add structured data with tags
# Code blocks
client.msg.add_message_block("CODE", """
def hello(): print("Hello, World!")
""")

# Data structures
client.msg.add_message_block("DATA", {
    "users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}],
    "metrics": {"total": 100, "active": 80}
})

# DataFrames and custom classes
import pandas as pd
from dataclasses import dataclass

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
client.msg.add_message_block("DATAFRAME", df)

@dataclass
class Stats:
    total: int
    active: int
client.msg.add_message_block("STATS", Stats(100, 80))
```

Supported data types (automatically formatted):
- Basic types (str, int, float, bool)
- Collections (lists, dicts, sets)
- Pandas DataFrames
- Pydantic models
- Dataclasses
- Custom objects with __str__
- NumPy arrays
- JSON-serializable objects

## Quick Start ⚡

### Basic Usage Example

```python
from pydantic import Field
from typing import List, Optional, Any
from litellm2 import Request, LiteLLMClient
from drf_pydantic import BaseModel


class CustomAnswer(BaseModel):
    """Example custom answer model."""
    content: str = Field(..., description="The main content")
    keywords: List[str] = Field(default_factory=list, description="Keywords extracted from the response")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis")


class TextAnalyzer:
    """Service for analyzing text using AI with structured responses."""

    def __init__(self):
        """Initialize the text analyzer with configuration."""
        self.config = Request(
            # Model configuration
            model="openrouter/openai/gpt-4o-mini-2024-07-18",
            answer_model=CustomAnswer,  # Required: Defines response structure
            temperature=0.7,
            max_tokens=500,

            # Performance features
            online=True,              # Enable web search capability
            cache_prompt=False,       # Disable prompt caching
            max_budget=0.05,        # Set maximum budget per request

            # Debugging options
            verbose=True,             # Enable detailed output
            logs=False                # Enable logging
        )
        self.client = LiteLLMClient(self.config)

    def analyze_text(self, text: str, data_list: Any) -> CustomAnswer:
        """
        Analyze the provided text and return structured insights.

        Args:
            text (str): The text to analyze

        Returns:
            CustomAnswer: Structured analysis results
        """
        # Set up the conversation context
        self.client.msg.add_message_system(
            "You are an AI assistant that provides structured analysis with keywords and sentiment."
        )

        self.client.msg.add_message_block('DATA', data_list)

        # Add the text to analyze
        self.client.msg.add_message_user(f"Analyze the following text: '{text}'")

        # Generate and return structured response
        return self.client.generate_response()


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = TextAnalyzer()

    data_list = [
        {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com"
        },
        {
            "name": "Jane Smith",
            "age": 25,
            "email": "jane.smith@example.com"
        }
    ]

    result = analyzer.analyze_text("John Doe is 30 years old and works at Google.", data_list)

    print('CONFIG:')
    print(analyzer.client.config.model_dump_json(indent=2))

    print('-' * 100)

    print('META:')
    print(analyzer.client.meta.model_dump_json(indent=2))

    print('*' * 100)

    print('RESULT:')
    print(result.model_dump_json(indent=2))
```

Key components in this example:
- `CustomAnswer`: Pydantic model defining the response structure
- `Request` configuration: Model settings, performance features, and debugging options
- Message types: system, user, and block messages for structured input
- Response handling: Typed responses with JSON output

---

### Django Integration Example

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import serializers
from drf_pydantic import BaseModel
from pydantic import Field
from typing import List
from litellm2 import Request, LiteLLMClient

class FeedbackAnalysis(BaseModel):
    summary: str = Field(..., description="Summary of the feedback")
    sentiment: str = Field(..., description="Detected sentiment")
    key_points: List[str] = Field(..., description="Key points from the feedback")

class FeedbackResponseSerializer(serializers.Serializer):
    answer = FeedbackAnalysis.drf_serializer()

class FeedbackView(APIView):
    def post(self, request):
        feedback = request.data.get('feedback', '')

        client = LiteLLMClient(Request(
            model="openrouter/openai/gpt-4o-mini",
            temperature=0.3,
            answer_model=FeedbackAnalysis
        ))

        client.msg.add_message_system("You are a feedback analysis expert.")
        client.msg.add_message_user(feedback)

        response: FeedbackAnalysis = client.generate_response()

        serializer = FeedbackResponseSerializer(data={
            "answer": response.model_dump()
        })
        serializer.is_valid(raise_exception=True)

        return Response(serializer.data)
```

Key features:
- Seamless integration with Django REST framework
- Automatic serialization of Pydantic models
- Type-safe response handling
- Built-in validation

---

### Agent System Example

```python
from litellm2.agents import SimpleAgent
from litellm2.utils.tools import Tool
from pydantic import BaseModel, Field
from typing import List
import datetime

class AgentResponse(BaseModel):
    answer: str = Field(..., description="The main answer")
    reasoning: str = Field(..., description="The reasoning process")
    tools_used: List[str] = Field(default_factory=list)

def get_current_time() -> str:
    """Get the current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_text(text: str) -> str:
    """Process text input."""
    return text.upper()

class MyAgent(SimpleAgent):
    def __init__(self):
        super().__init__(answer_model=AgentResponse)

        # Add tools
        self.add_tool(Tool(
            name="get_time",
            description="Get current time",
            func=get_current_time
        ))
        self.add_tool(Tool(
            name="process_text",
            description="Convert text to uppercase",
            func=process_text
        ))

# Usage
agent = MyAgent()
result = agent.run("What time is it and convert 'hello world' to uppercase")
```

Key features:
- Custom tools integration
- Structured responses with Pydantic
- Automatic tool selection and execution
- Type-safe tool inputs and outputs

---

## About 👥

Developed by [Unrealos Inc.](https://unrealos.com/) - We create innovative AI-powered solutions for business.

## License 📝

MIT License - see the LICENSE file for details.

## Credits ✨

- Developed by [Unrealos Inc.](https://unrealos.com/)
