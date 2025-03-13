from pydantic import Field
from typing import List, Optional

from src.litellm2 import Request, LiteLLMClient
from drf_pydantic import BaseModel


class CustomAnswer(BaseModel):
    """Example custom answer model."""
    content: str = Field(..., description="The main content")
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords extracted from the response"
    )
    sentiment: Optional[str] = Field(None, description="Sentiment analysis")


def test_litellm_client_initialization():
    """Test that the client can be initialized with a configuration."""
    # Create client with default configuration and custom answer model
    config = Request(
        temperature=0.7,
        max_tokens=500,
        model="openrouter/openai/gpt-4o-mini",
        online=False,
        cache_prompt=True,
        max_budget=0.05,
        answer_model=CustomAnswer,
        verbose=True,
        logs=True,
    )

    # Initialize typed client with CustomAnswer
    client = LiteLLMClient(config)

    # Test that the client was initialized correctly
    assert client is not None
    assert client.config.temperature == 0.7
    assert client.config.model == "openrouter/openai/gpt-4o-mini"
    assert client.config.online is False


def test_client_add_message():
    """Test that messages can be added to the client."""
    config = Request(
        model="openrouter/openai/gpt-4o-mini",
        answer_model=CustomAnswer,
        online=False,
        temperature=0.7,
        max_tokens=500,
        budget_limit=0.05,
    )

    client = LiteLLMClient(config)

    # Add a message
    test_message = "This is a test message"
    client.msg.add_message_user(test_message)

    # Check if the message was added - pass the answer_model parameter
    messages = client.msg.get_messages(answer_model=CustomAnswer)
    assert len(messages) > 0
    assert any(msg.get("content") == test_message for msg in messages)
