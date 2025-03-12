import json
from typing import List
from drf_pydantic import BaseModel

from ..utils.logger import logger


class MessageHandler:
    def __init__(self):
        self.messages_user = []
        self.messages_assistant = []
        self.messages_system = []
        self.messages_block = []

    def add_message_user(self, message: str):
        self.messages_user.append(message)

    def add_message_assistant(self, message: str):
        self.messages_assistant.append(message)

    def add_message_system(self, message: str):
        self.messages_system.append(message)

    def add_message_block(self, tag: str, message: str):
        tag = tag.upper()
        self.messages_block.append(f"[{tag}] {message} [/{tag}]")

    def _get_schema_str(self, answer_model: BaseModel) -> str:
        """Generate schema instructions for the model."""
        schema = answer_model.model_json_schema()
        # Remove metadata that might confuse the AI
        schema.pop('title', None)
        schema.pop('type', None)

        response = f"""
        Response:
        - Do not return the schema itself, return only the JSON object based on the schema.
        - Use snake_case for the keys of the JSON object.
        [SCHEMA]
        {json.dumps(schema)}
        [/SCHEMA]

        """
        return self.trim_message(response)

    def trim_message(self, message: str) -> str:
        """Trim spaces from each line in the message."""
        return "\n".join([line.strip() for line in message.split("\n")])

    def get_messages(self, answer_model) -> List[dict]:
        """Get all messages in the correct order with schema instructions."""
        messages = [
            {"role": "user", "content": self._get_schema_str(answer_model)}
        ]

        for message in self.messages_system:
            messages.append({"role": "system", "content": self.trim_message(message)})

        for message in self.messages_assistant:
            messages.append({"role": "assistant", "content": self.trim_message(message)})

        user_messages = [
            *self.messages_user,
            *self.messages_block,
        ]
        user_messages = [self.trim_message(message) for message in user_messages]
        messages.append({"role": "user", "content": "\n".join(user_messages)})

        logger.info(f"Messages: {messages}")

        # trim spaces from each line
        return messages
