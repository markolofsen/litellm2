import time
from typing import TypeVar, Type, Any, Tuple
import json

import litellm
from litellm import completion
from litellm.exceptions import (
    RateLimitError, Timeout, APIConnectionError,
    BudgetExceededError, AuthenticationError
)

from ..models.base_models import Request, Meta, BaseModel, FullResponse
from ..config.config import setup_caching
from .message_handler import MessageHandler
from ..utils.logger import logger

T = TypeVar('T', bound=BaseModel)


class LiteLLMClient():
    def __init__(self, config: Request):
        # Initialize caching
        setup_caching()

        # Initialize message handler
        self.msg = MessageHandler()

        # Store configuration
        self.config = config

        self.meta = Meta(
            request_timestamp=time.time(),
            model_used=None,
            cache_hit=None,
            response_time_seconds=None,
            token_count=None
        )

        self._answer_model = config.answer_model

        # Set debug verbosity
        if config.verbose:
            logger.set_verbose(config.verbose)

        litellm.logging = config.verbose
        litellm.max_budget = config.max_budget
        litellm.caching = config.cache_prompt

        logger.info("LiteLLMClient initialized")

    @property
    def answer_model(self) -> Type[BaseModel]:
        return self._answer_model

    def _extract_first_json_object(self, text: str) -> Tuple[str, int, int]:
        """Extract the first complete JSON object from text.

        Args:
            text: Text containing one or more JSON objects

        Returns:
            Tuple of (json_content, start_index, end_index)
            If no valid JSON found, returns ("", -1, -1)
        """
        text = text.strip()
        stack = []
        in_string = False
        escape_char = False
        start_idx = -1

        for i, char in enumerate(text):
            # Handle string literals
            if char == '"' and not escape_char:
                in_string = not in_string
            elif char == '\\' and not escape_char:
                escape_char = True
                continue

            if not in_string:
                if char == '{':
                    if not stack:  # First opening brace
                        start_idx = i
                    stack.append(char)
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        if not stack:  # Complete object found
                            try:
                                # Validate that this is actually valid JSON
                                json_str = text[start_idx:i + 1]
                                json.loads(json_str)  # Test if it's valid JSON
                                return json_str, start_idx, i + 1
                            except json.JSONDecodeError:
                                # Not valid JSON, continue searching
                                start_idx = -1
                                continue

            escape_char = False

        return "", -1, -1

    def generate_response(self) -> T:
        """Generate a response using LiteLLM.

        Returns:
            An instance of the answer_model class or None if all attempts fail.
        """
        # Create a clean copy of the request
        request_params = self.config.model_dump()

        logger.debug(f"Request parameters: {request_params}")

        attempt = 0
        models_to_try = [self.config.model] + (self.config.fallbacks or [])

        # Make sure messages are not empty
        if not self.config.messages:
            self.config.messages = self.msg.get_messages(self.config.answer_model)
            logger.debug(f"Using messages from message handler: {self.config.messages}")

        if not self.config.messages:
            raise ValueError("Messages are empty")

        # Add :online to the model if it's not already there and online mode is requested
        if self.config.online and not self.config.model.endswith(":online"):
            self.config.model = f"{self.config.model}:online"
            logger.warning("Switched to online mode")

        # Create metadata object to track response metrics
        self.meta = Meta(
            request_timestamp=time.time(),
            model_used=None,
            cache_hit=None,
            response_time_seconds=None,
            token_count=None
        )

        response_content = None

        for model in models_to_try:
            try:
                logger.info(f"Sending request to LiteLLM with model {model}")

                # Start timing the response
                start_time = time.time()

                # Extract only needed parameters for LiteLLM
                llm_response = completion(
                    model=model,
                    messages=self.config.messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=False,
                    caching=self.config.cache_prompt,
                    logprobs=False,
                )

                # Calculate response time
                response_time = time.time() - start_time
                logger.debug(f"Raw LLM response: {llm_response}")
                logger.info(f"Response time: {response_time:.2f} seconds")

                # Update metadata
                self.meta.response_time_seconds = round(response_time, 3)
                self.meta.model_used = model
                cache_condition = response_time < 0.5 if self.config.cache_prompt else False
                self.meta.cache_hit = cache_condition

                # Extract content from response
                try:
                    # Use ANY type to handle both dict and list responses
                    llm_resp: Any = llm_response
                    content_text = llm_resp.choices[0].message.content
                except Exception as e:
                    logger.warning(f"Error extracting content from response: {e}")
                    if isinstance(llm_response, dict) and "choices" in llm_response:
                        choices = llm_response["choices"][0]
                        message = choices.get("message", {})
                        content_text = message.get("content", "")
                    else:
                        logger.error("No valid response content found")
                        continue

                if not content_text:
                    logger.warning("Empty content received")
                    continue

                logger.debug(f"Extracted content text: {content_text}")

                # Process the content based on the requested answer model
                try:
                    # Extract the first valid JSON object
                    json_content, start_idx, end_idx = self._extract_first_json_object(content_text)

                    if not json_content or start_idx == -1:
                        logger.warning("No valid JSON object found in response")
                        logger.debug(f"Raw content: {content_text}")
                        raise ValueError("No valid JSON content found", content_text)

                    # Validate and create a response model object
                    model_cls = self.answer_model
                    response_content = model_cls.model_validate_json(json_content)

                    # Save to log
                    if self.config.logs:
                        full_response = FullResponse.create(
                            request=self.config,
                            meta=self.meta,
                            response=response_content
                        )
                        full_response.save_to_log()
                        logger.info("Response saved to log")

                    return response_content

                except Exception as e:
                    logger.error(f"Error parsing JSON content: {e}")
                    continue

            except AuthenticationError:
                logger.error("Authentication failed: Invalid API key.")
                raise ValueError("Invalid API key")
            except (RateLimitError, Timeout, APIConnectionError, BudgetExceededError) as e:
                logger.warning(f"Error with model {model}: {e}. Trying fallback...")
                attempt += 1
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                continue

        # Calculate total time even for failed requests
        total_time = time.time() - (self.meta.request_timestamp or time.time())
        time_rounded = round(total_time, 3) if total_time > 0 else None
        self.meta.response_time_seconds = time_rounded

        logger.error("All attempts failed to generate a valid response")
        return None
