import time
import os
from typing import TypeVar, Type
from litellm import completion
from litellm.exceptions import RateLimitError, Timeout, APIConnectionError, BudgetExceededError, AuthenticationError

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

        # Set logging verbosity
        logger.set_verbose(config.verbose)
        logger.info("LiteLLMClient initialized")

    @property
    def answer_model(self) -> Type[BaseModel]:
        return self._answer_model

    def generate_response(self) -> Type[T]:
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

        if self.config.online:
            self.config.model = f"{self.config.model}:online"
            # self.config.cache_prompt = False
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
                self.meta.cache_hit = response_time < 0.5 if self.config.cache_prompt else False

                # Extract content from response
                try:
                    content_text = llm_response.choices[0].message.content
                except Exception as e:
                    logger.warning(f"Error extracting content from response: {e}")
                    if isinstance(llm_response, dict) and "choices" in llm_response:
                        content_text = llm_response["choices"][0].get("message", {}).get("content", "")
                    else:
                        logger.error("No valid response content found")
                        continue

                if not content_text:
                    logger.warning("Empty content received")
                    continue

                logger.debug(f"Extracted content text: {content_text}")

                # Process the content based on the requested answer model
                try:
                    json_content = content_text.strip()[content_text.strip().find('{'):content_text.strip().rfind('}') + 1]
                    if not json_content:
                        raise ValueError("No valid JSON content found", content_text)

                    try:
                        response_content = self.answer_model.model_validate_json(json_content)
                    except Exception as e:
                        logger.error(f"Error validating JSON content: {e}")
                        raise ValueError("Invalid JSON content", json_content)

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
                logger.warning(f"Error with model {model}: {e}. Trying next fallback if available...")
                attempt += 1
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                continue

        # Calculate total time even for failed requests
        total_time = time.time() - (self.meta.request_timestamp or time.time())
        self.meta.response_time_seconds = round(total_time, 3) if total_time > 0 else None

        logger.error("All attempts failed to generate a valid response")
        return None
