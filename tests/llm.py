from pydantic import Field
from typing import List, Optional

from src.litellm2 import Request, LiteLLMClient
from drf_pydantic import BaseModel


class CustomAnswer(BaseModel):
    """Example custom answer model."""
    content: str = Field(..., description="The main content")
    keywords: List[str] = Field(default_factory=list, description="Keywords extracted from the response")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis")


def main():

    # Create client with default configuration and custom answer model
    config = Request(
        temperature=0.7,
        max_tokens=500,
        model="openrouter/openai/gpt-4o-mini",
        online=False,
        cache_prompt=True,
        budget_limit=0.05,
        answer_model=CustomAnswer,
        verbose=True,
        logs=True,
    )

    # Initialize typed client with CustomAnswer
    client = LiteLLMClient(config)

    # Add a message
    client.msg.add_message_user("""
        You are a helpful assistant that can answer questions and help with tasks.
        You are given a question and you need to answer it.
        You are also given a list of keywords and you need to use them to answer the question.
        You are also given a sentiment and you need to use it to answer the question.
        You are also given a content and you need to use it to answer the question.
    """)

    try:
        # Generate response
        response: CustomAnswer = client.generate_response()

        # Print current response details
        print('Meta:', client.meta.model_dump())
        print('Request:', client.config.model_dump())

        if response is not None:
            print('Response:', response.sentiment)
        else:
            print('Error: Failed to generate response')

    except Exception as e:
        print(f'Error: {str(e)}')


if __name__ == "__main__":
    main()
