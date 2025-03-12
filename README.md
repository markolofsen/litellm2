![LiteLLM2](https://raw.githubusercontent.com/markolofsen/litellm2/main/assets/cover.png)

# LiteLLM2 🚀

A powerful and flexible AI agent framework with structured Pydantic response handling and LLM integration capabilities.

## Overview 🔍

LiteLLM2 is built on top of the popular [litellm](https://pypi.org/project/litellm/) library and aims to simplify working with LLMs through **typesafe, structured responses**. It provides a comprehensive framework for creating AI agents with various capabilities:

- **Structured Pydantic responses** (core feature) ✅
- Flexible LLM client integration with multiple providers 🔌
- Budget management and caching for cost control 💰
- Advanced agent system with built-in tool integration 🛠️

The main differentiator of LiteLLM2 is its focus on structured data handling through Pydantic models, ensuring type safety and predictable response formats.

### Powerful Technology Stack 🔋

LiteLLM2 combines several powerful technologies into one seamless package:

- **[Pydantic](https://docs.pydantic.dev/)**: For structured, type-safe data handling and validation
- **[LiteLLM](https://litellm.ai/)**: Providing the core LLM routing capabilities
- **[OpenRouter](https://openrouter.ai/)**: Default model provider with access to numerous models (though any other provider compatible with LiteLLM can be used)

This powerful combination delivers a flexible, robust framework for building AI applications with structured outputs and reliable performance.

---

## Installation and Setup 📦

### Installation

```bash
pip install litellm2
```

### API Key Setup

Before using LiteLLM2, you need to set up your API key:

```bash
# In your .env file
OPENROUTER_API_KEY=your_api_key_here
```

Or set it as an environment variable:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

---

## Quick Start ⚡

Here's a simple example to get you started with structured responses:

```python
from pydantic import BaseModel, Field
from litellm2 import Request, LiteLLMClient

# Define a structured response model
class RoboticsAnswer(BaseModel):
    explanation: str = Field(..., description="Detailed explanation of the answer")
    laws: list[str] = Field(..., description="List of robotics laws")

# Initialize client with basic configuration and required answer model
client = LiteLLMClient(Request(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0.7,
    answer_model=RoboticsAnswer  # Always required
))

# Add a user message
client.msg.add_message_user("What are the three laws of robotics?")

# Generate and print structured response
response: RoboticsAnswer = client.generate_response()
print(f"Explanation: {response.explanation}")
print("\nLaws:")
for i, law in enumerate(response.laws, 1):
    print(f"{i}. {law}")
```

---

## Core Concepts 🧠

### Configuration ⚙️

LiteLLM2 is configured through the `Request` class with these key options:

```python
config = Request(
    # Required parameters
    model="openrouter/openai/gpt-4o-mini",   # Model identifier
    answer_model=CustomAnswer,                # Pydantic model for structured responses (REQUIRED)

    # Model behavior
    temperature=0.7,                          # Controls randomness (0.0 to 1.0)
    max_tokens=500,                           # Maximum tokens in the response

    # Cost and performance features
    online=False,                             # Enable web search (OpenRouter only)
    cache_prompt=True,                        # Cache identical prompts
    budget_limit=0.05,                        # Maximum budget in dollars

    # Debugging
    verbose=True,                             # Enable verbose output
    logs=True                                 # Enable logging to file
)
```

> **Important:** The `answer_model` parameter is mandatory as LiteLLM2 is specifically designed for structured responses using Pydantic models.

### Specifying Model Providers 🔄

LiteLLM2 supports various model providers through the OpenRouter integration. The `model` parameter uses the format `openrouter/{provider}/{model_name}`.

For a complete list of supported models and their capabilities, see the [OpenRouter documentation](https://docs.litellm.ai/docs/providers/openrouter).

### Message Handling 💬

Build complex prompts with different message types:

```python
# Define a response model for our data analysis
class AnalysisResult(BaseModel):
    trend_description: str = Field(..., description="Description of the observed trend")
    growth_rate: float = Field(..., description="Calculated growth rate")
    forecast: str = Field(..., description="Future forecast based on the trend")

# Initialize client with required answer model
client = LiteLLMClient(Request(
    model="openrouter/openai/gpt-4o-mini",
    answer_model=AnalysisResult
))

# Add different types of messages
client.msg.add_message_system("You are a helpful AI assistant specialized in data analysis.")
client.msg.add_message_user("Analyze the growth trend in this data: [10, 15, 22, 35, 42]")
client.msg.add_message_assistant("I'll analyze this data step by step.")
client.msg.add_message_block("DATA", "The year-over-year growth is approximately 40%.")
```

> 💡 **Pro Tip:** Don't worry about extra whitespace in your prompts! LiteLLM2 automatically trims all messages to ensure clean, consistent prompts to the LLM.

Available message methods:
- **`add_message_user(message: str)`**: Add a user message
- **`add_message_system(message: str)`**: Add a system message (for instructions)
- **`add_message_assistant(message: str)`**: Add an assistant message (for context)
- **`add_message_block(tag: str, message: str)`**: Add a tagged block of content

#### Understanding `add_message_block`

The `add_message_block` method is a powerful feature that allows you to include structured, tagged content in your prompts. This is especially useful for:

- Including different types of content (code, data, JSON, etc.) with clear boundaries
- Helping the LLM distinguish between different parts of your input
- Creating multi-part prompts with clear sections

**How it works:**

When you call `add_message_block("TAG", "content")`, the library formats the content with special markers that help the LLM understand the structure:

```python
# Example usage
client.msg.add_message_system("You are a data analysis assistant.")
client.msg.add_message_user("Analyze this data and the SQL query results together.")
client.msg.add_message_block("CSV_DATA", "date,value\n2023-01,10\n2023-02,15\n2023-03,22\n2023-04,35\n2023-05,42")
client.msg.add_message_block("SQL_RESULTS", "| product_id | sales_count | revenue |\n|------------|-------------|----------|\n| A001 | 243 | $4,860 |\n| B002 | 157 | $3,925 |")
```

**Example of the resulting prompt format:**

The above message blocks get transformed into a structured prompt that looks similar to this:

```
SYSTEM: You are a data analysis assistant.

USER: Analyze this data and the SQL query results together.

[CSV_DATA]
date,value
2023-01,10
2023-02,15
2023-03,22
2023-04,35
2023-05,42
[/CSV_DATA]

[SQL_RESULTS]
| product_id | sales_count | revenue |
|------------|-------------|----------|
| A001 | 243 | $4,860 |
| B002 | 157 | $3,925 |
[/SQL_RESULTS]
```

This approach has several benefits:

1. **Clear boundaries**: The LLM can easily identify different blocks of information with opening and closing tags
2. **Improved context understanding**: Tags help the model understand what kind of data it's looking at
3. **Better structured responses**: The clearer input structure leads to better structured outputs
4. **Language-specific handling**: For code blocks, you can specify the language for proper formatting

**When to use it:**

Use `add_message_block` when you need to include:
- Structured data (CSV, JSON, tables)
- Code snippets with syntax highlighting
- Multiple distinct text sources
- Any content that should be visually separated in the prompt

### Generating Responses 🔄

Generate structured responses after adding messages:

```python
# Generate a structured response
response: AnalysisResult = client.generate_response()

# Access typed fields with autocompletion support
print(f"Trend: {response.trend_description}")
print(f"Growth Rate: {response.growth_rate}%")
print(f"Forecast: {response.forecast}")

# Access metadata about the request
print('Token usage:', client.meta.token_count)
print('Response time:', client.meta.response_time_seconds)
```

---

## Structured Responses with Pydantic 📊

LiteLLM2's core feature is working with structured data models. Every response is parsed into your defined Pydantic model:

```python
from pydantic import Field, BaseModel
from typing import List, Optional
from litellm2 import Request, LiteLLMClient

class CustomAnswer(BaseModel):
    """Example custom answer model."""
    content: str = Field(..., description="The main content")
    keywords: List[str] = Field(default_factory=list, description="Keywords extracted")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis")

# Initialize client with the custom answer model
config = Request(
    model="openrouter/openai/gpt-4o-mini",
    answer_model=CustomAnswer  # Required parameter
)
client = LiteLLMClient(config)

# Add user message
client.msg.add_message_user("Analyze this customer feedback: 'The product was fantastic!'")

# Generate typed response
response: CustomAnswer = client.generate_response()

# Access structured fields with type hints and autocompletion
print(f"Content: {response.content}")
print(f"Keywords: {', '.join(response.keywords)}")
print(f"Sentiment: {response.sentiment}")
```

> ✨ **Magic of Schema Handling:** LiteLLM2 automatically sends your Pydantic schema to the LLM and deserializes the JSON response into the exact schema you provided via the `answer_model` parameter. No manual parsing or complex prompt engineering required!

> **Important:** Using type annotations like `response: CustomAnswer = client.generate_response()` provides IDE autocompletion and type checking, which is a core benefit of using LiteLLM2.

### Best Practices for Response Models ✅

For optimal results with your Pydantic models:

1. Use descriptive field names and types
2. Add detailed descriptions using the `Field()` parameter
3. Use appropriate data types (str, int, float, bool, list, etc.)
4. Set default values for optional fields
5. Keep models focused on the specific data you need

Example of a well-structured model:

```python
class ProductRecommendation(BaseModel):
    product_name: str = Field(..., description="The name of the recommended product")
    reasoning: str = Field(..., description="Why this product is recommended")
    price_range: str = Field(..., description="Expected price range (e.g., '$10-$20')")
    rating: float = Field(..., description="Predicted rating out of 5")
    best_for: List[str] = Field(..., description="Types of users this is best for")
    alternatives: List[str] = Field(default_factory=list, description="Alternative products")
```

---

## Advanced Features 🔧

### Request and Response Data 📈

Access detailed information about requests and responses:

```python
# After generating a response
meta_data = client.meta.model_dump()
config_data = client.config.model_dump()
```

**Available Data:**

- **Metadata (`client.meta`)**: Performance and usage statistics
  - Request timestamp and response time
  - Model used and cache status
  - Token counts (prompt, completion, total)

- **Configuration (`client.config`)**: Current settings
  - Temperature, max tokens, model ID
  - Online mode, caching, and budget settings
  - Verbosity and logging options

This data is useful for tracking costs, optimizing prompts, monitoring performance, and maintaining configuration snapshots.

### Error Handling ⚠️

Handle various error scenarios:

```python
try:
    response = client.generate_response()
except RateLimitError:
    print("Rate limit exceeded, try again later")
except BudgetExceededError:
    print("Budget limit exceeded")
except AuthenticationError:
    print("API key is invalid")
except APIConnectionError:
    print("Could not connect to the API")
except Exception as e:
    print(f"An error occurred: {str(e)}")
```

---

## Integrations 🔌

### Django Integration 🎯

LiteLLM2 works seamlessly with Django through `drf-pydantic`:

```python
from rest_framework import serializers
from django.db import models
from litellm2 import Request, LiteLLMClient
from drf_pydantic import BaseModel  # Use drf_pydantic.BaseModel for Django
from pydantic import Field
from typing import List, Optional

# Define Pydantic model using drf_pydantic.BaseModel
class CustomAnswer(BaseModel):
    content: str = Field(..., description="The main content")
    keywords: List[str] = Field(default_factory=list, description="Keywords extracted")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis")

# Example Django model
class AIResponse(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    prompt = models.TextField()

# Example serializer
class CustomAnswerSerializer(serializers.ModelSerializer):
    """Serializer for the CustomAnswer model."""
    answer = CustomAnswer.drf_serializer()  # Special method from drf_pydantic

    class Meta:
        model = AIResponse
        fields = ['prompt', 'answer', 'created_at']
```

#### Complete Django View Example

Here's a complete example showing how to use LiteLLM2 in a Django view and return the results through a serializer:

```python
# In models.py
from django.db import models

class FeedbackAnalysis(models.Model):
    feedback = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    # Store metadata about the analysis
    model_used = models.CharField(max_length=100)
    response_time = models.FloatField()
    token_count = models.IntegerField(default=0)

# In serializers.py
from rest_framework import serializers
from drf_pydantic import BaseModel
from pydantic import Field
from typing import List

# Pydantic model for LLM response
class FeedbackInsights(BaseModel):
    summary: str = Field(..., description="Summary of the feedback")
    sentiment: str = Field(..., description="Detected sentiment (positive/negative/neutral)")
    key_points: List[str] = Field(..., description="Key points from the feedback")
    improvement_suggestions: List[str] = Field(default_factory=list,
                                            description="Suggestions for improvement")

# Request serializer
class FeedbackRequestSerializer(serializers.Serializer):
    feedback = serializers.CharField(help_text="Customer feedback to analyze")

# Response serializer with LLM output
class FeedbackResponseSerializer(serializers.Serializer):
    request_id = serializers.UUIDField()
    created_at = serializers.DateTimeField()
    model_used = serializers.CharField()

    # These fields will be populated from the LLM response
    summary = serializers.CharField()
    sentiment = serializers.CharField()
    key_points = serializers.ListField(child=serializers.CharField())
    improvement_suggestions = serializers.ListField(child=serializers.CharField())

    # Metadata about the request
    response_time = serializers.FloatField()
    token_count = serializers.IntegerField()

# In views.py
import uuid
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from litellm2 import Request, LiteLLMClient
from .models import FeedbackAnalysis
from .serializers import (FeedbackRequestSerializer,
                         FeedbackResponseSerializer,
                         FeedbackInsights)

class FeedbackAnalysisView(APIView):
    """
    API endpoint for analyzing customer feedback using LiteLLM2.
    """

    def post(self, request):
        # Validate request
        serializer = FeedbackRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Extract feedback from validated data
        feedback = serializer.validated_data['feedback']

        try:
            # Initialize LiteLLM client with our custom response model
            client = LiteLLMClient(Request(
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.3,  # Lower temperature for more factual analysis
                answer_model=FeedbackInsights  # Our pydantic model
            ))

            # Add messages to the client
            client.msg.add_message_system(
                "You are an expert in analyzing customer feedback. "
                "Extract key insights, determine sentiment, and suggest improvements."
            )
            client.msg.add_message_user(feedback)

            # Generate response from LLM
            llm_response = client.generate_response()

            # Create a record in the database
            request_id = uuid.uuid4()
            timestamp = timezone.now()

            # Store the analysis in the database
            analysis = FeedbackAnalysis.objects.create(
                feedback=feedback,
                model_used=client.meta.model_used,
                response_time=client.meta.response_time_seconds,
                token_count=client.meta.token_count.total_tokens
            )

            # Prepare response data
            response_data = {
                'request_id': request_id,
                'created_at': timestamp,
                'model_used': client.meta.model_used,

                # LLM response fields
                'summary': llm_response.summary,
                'sentiment': llm_response.sentiment,
                'key_points': llm_response.key_points,
                'improvement_suggestions': llm_response.improvement_suggestions,

                # Metadata
                'response_time': client.meta.response_time_seconds,
                'token_count': client.meta.token_count.total_tokens
            }

            # Serialize the response
            response_serializer = FeedbackResponseSerializer(data=response_data)
            response_serializer.is_valid(raise_exception=True)

            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': f'Analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# In urls.py
from django.urls import path
from .views import FeedbackAnalysisView

urlpatterns = [
    path('analyze-feedback/', FeedbackAnalysisView.as_view(), name='analyze-feedback'),
]
```

This example demonstrates:

1. A Pydantic model (`FeedbackInsights`) for structuring the LLM response
2. Django serializers for request validation and response formatting
3. A Django model for storing analysis results
4. A complete APIView that:
   - Validates the incoming request
   - Sends the feedback to LiteLLM2 for analysis
   - Stores the results in the database
   - Returns a structured API response

You can test this endpoint with:

```bash
curl -X POST http://localhost:8000/api/analyze-feedback/ \
     -H "Content-Type: application/json" \
     -d '{"feedback": "The product is great but the customer service could be improved."}'
```

---

## Working with Agents 🤖

LiteLLM2 includes a robust agent system that extends LLM capabilities with tools.

### Comprehensive Agent Example

The following example demonstrates how to create and use agents with custom tools and structured responses:

```python
from litellm2.agents import SimpleAgent, AdvancedAgent
from litellm2.utils.tools import Tool
from smolagents.tools import tool
from pydantic import BaseModel, Field
import datetime

# 1. Define a structured response model for the agent
class AgentResponse(BaseModel):
    answer: str = Field(..., description="The main answer to the query")
    reasoning: str = Field(..., description="The reasoning process")
    timestamp: str = Field(..., description="When the response was generated")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in processing")

# 2. Create custom tools
@tool
def date_time(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current date and time in the specified format."""
    now = datetime.datetime.now()
    return now.strftime(format_string)

def text_processor(input_text: str) -> str:
    """A tool that processes text input."""
    return f"PROCESSED: {input_text.upper()}"

# 3. Create a custom agent class
class MyAgent(SimpleAgent):
    def __init__(self):
        # Initialize with our structured response model
        super().__init__(answer_model=AgentResponse)

        # Add tools to the agent
        self.add_tools([date_time])
        self.add_tool(Tool(
            name="text_processor",
            description="Processes text in all uppercase",
            func=text_processor
        ))

# 4. Usage examples

# Create our custom agent
agent = MyAgent()

# Run a query that might use the date tool
result = agent.run("What's today's date and can you process the text 'hello world'?")

# Access structured fields from the response
print(f"Answer: {result.answer}")
print(f"Reasoning: {result.reasoning}")
print(f"Response time: {result.timestamp}")
print(f"Tools used: {', '.join(result.tools_used)}")

# For quick demos or exploration, use the included UI
# agent.run_demo()  # Launches a Gradio interface if available
```

This example shows:
1. Creating a structured response model for agent outputs
2. Defining custom tools using both the decorator and Tool class approaches
3. Building a custom agent that extends SimpleAgent
4. Running queries and accessing typed results

---

## About 👥

Developed by [Unrealos Inc.](https://unrealos.com/) - We create innovative SaaS and PaaS solutions powered by AI for business. Our expertise includes:
- AI-powered business solutions
- SaaS platforms
- PaaS infrastructure
- Custom enterprise software

## License 📝

MIT License - see the LICENSE file for details.

## Credits ✨

- Developed by [Unrealos Inc.](https://unrealos.com/)
