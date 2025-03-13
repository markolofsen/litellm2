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
            budget_limit=0.05,        # Set maximum budget per request

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
