import logging
import requests
from litellm.llms.custom_llm import CustomLLM
import litellm

logger = logging.getLogger(__name__)


class CustomProvider(CustomLLM):
    def completion(self, model: str, messages: list, **kwargs):
        url = "https://***.unrealos.com/mistral"
        headers = {"Content-Type": "application/json"}

        # Convert messages to a single prompt
        prompt = "\n".join([m["content"] for m in messages])

        payload = {"prompt": prompt}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()

            # Format response to match expected structure
            result = response.json()
            return {
                "choices": [{
                    "message": {
                        "content": result["response"],
                        "role": "assistant"
                    }
                }]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Custom provider error: {str(e)}")
            raise Exception(f"Custom provider error: {str(e)}")


# Initialize and register custom provider
custom_provider = CustomProvider()
litellm.custom_provider_map = [
    {"provider": "custom_provider", "custom_handler": custom_provider}
]
