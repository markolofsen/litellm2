import json
from typing import List, Any, Optional
from drf_pydantic import BaseModel
import yaml
import re

from ..utils.logger import logger


class MessageFormatError(Exception):
    """Raised when message format is not suitable for conversion."""
    pass


class MessageHandler:
    # Allowed types for messages
    ALLOWED_TYPES = (str, int, float, bool, dict, list)

    def __init__(self):
        self.messages_user = []
        self.messages_assistant = []
        self.messages_system = []
        self.messages_block = []

    def _validate_message(self, message: Any) -> None:
        """Validate if message type is supported.

        Args:
            message: Message to validate

        Raises:
            MessageFormatError: If message type is not supported
        """
        if not isinstance(message, self.ALLOWED_TYPES):
            type_name = type(message).__name__
            allowed_types = ', '.join(t.__name__ for t in self.ALLOWED_TYPES)
            raise MessageFormatError(
                f"Unsupported message type: {type_name}. "
                f"Allowed types are: {allowed_types}"
            )

        # For collections, validate each item
        if isinstance(message, (list, dict)):
            if isinstance(message, list):
                items = message
            else:
                items = message.values()

            for item in items:
                if not isinstance(item, self.ALLOWED_TYPES):
                    type_name = type(item).__name__
                    raise MessageFormatError(
                        f"Unsupported type in collection: {type_name}"
                    )

    def _format_data(self, data: Any, index: Optional[int] = None) -> str:
        """Format data in a readable way using YAML.

        Args:
            data: Data to format (dict, list, or any other type)
            index: Optional index for list items

        Returns:
            Formatted string representation of the data

        Raises:
            MessageFormatError: If data type is not supported
        """
        self._validate_message(data)

        if isinstance(data, (dict, list)):
            # Convert to YAML with custom style
            yaml_str = yaml.dump(
                data,
                default_flow_style=False,  # Use block style
                allow_unicode=True,        # Support Unicode
                sort_keys=False,           # Preserve key order
                indent=2                   # 2-space indentation
            )
            return yaml_str.strip()
        else:
            # For simple types
            return str(data)

    def _add_message(
        self,
        message: Any,
        target_list: List[str],
        split_lists: bool = False
    ) -> None:
        """Internal method to add a message to a specific message list.

        Args:
            message: Any type of message to add
            target_list: List where to add the message
            split_lists: Whether to split lists into separate messages

        Raises:
            MessageFormatError: If message type is not supported
        """
        self._validate_message(message)

        if split_lists and isinstance(message, list):
            for item in message:
                formatted_message = self._format_data(item)
                target_list.append(formatted_message)
        else:
            formatted_message = self._format_data(message)
            target_list.append(formatted_message)

    def add_message_user(self, message: Any) -> None:
        """Add a user message with intelligent formatting.

        Args:
            message: Any type of message that will be converted to a readable format
        """
        self._add_message(message, self.messages_user, split_lists=True)

    def add_message_assistant(self, message: Any) -> None:
        """Add an assistant message with intelligent formatting.

        Args:
            message: Any type of message that will be converted to a readable format
        """
        self._add_message(message, self.messages_assistant, split_lists=True)

    def add_message_system(self, message: Any) -> None:
        """Add a system message with intelligent formatting.

        Args:
            message: Any type of message that will be converted to a readable format
        """
        self._add_message(message, self.messages_system, split_lists=True)

    def add_message_block(self, tag: str, message: Any) -> None:
        """Add a tagged message block with intelligent string conversion.
        If message is a list, it will create separate numbered blocks for each item.

        Args:
            tag: The tag to wrap the message in (will be converted to uppercase)
            message: Any type of message that will be converted to a readable format
        """
        tag = tag.upper()

        # If message is a list, create separate blocks for each item
        if isinstance(message, list):
            for i, item in enumerate(message, 1):
                msg = self._format_data(item)
                self.messages_block.append(
                    f"[{tag}_{i}]\n{msg}\n[/{tag}_{i}]"
                )
        else:
            msg = self._format_data(message)
            self.messages_block.append(f"[{tag}]\n{msg}\n[/{tag}]")

    def _get_schema_str(self, answer_model: BaseModel) -> str:
        """Generate schema instructions for the model."""
        schema = answer_model.model_json_schema()
        # Remove metadata that might confuse the AI
        schema.pop('title', None)
        schema.pop('type', None)

        response = f"""
        Response:
        - Return only ONE clean JSON object based on the schema.
        - No code blocks, no extra text, just the JSON object.
        - Make sure the JSON is valid and properly formatted.
        - Do not return the schema itself, return only the JSON object based on the schema.
        [SCHEMA]
        {json.dumps(schema)}
        [/SCHEMA]

        """
        return self.trim_message(response)

    def trim_message(self, message: str) -> str:
        """Trim all types of whitespace from the message using regex.

        Args:
            message: Message to trim

        Returns:
            Message with trimmed whitespace
        """
        # Remove leading/trailing whitespace from each line
        # and collapse multiple spaces into single space
        lines = message.split('\n')
        trimmed_lines = [
            re.sub(r'\s+', ' ', line.strip())
            for line in lines
        ]
        # Remove empty lines at start and end
        while trimmed_lines and not trimmed_lines[0]:
            trimmed_lines.pop(0)
        while trimmed_lines and not trimmed_lines[-1]:
            trimmed_lines.pop()
        return '\n'.join(trimmed_lines)

    def get_messages(self, answer_model) -> List[dict]:
        """Get all messages in the correct order with schema instructions."""
        messages = [
            {"role": "user", "content": self._get_schema_str(answer_model)}
        ]

        for message in self.messages_system:
            messages.append(
                {"role": "system", "content": self.trim_message(message)}
            )

        for message in self.messages_assistant:
            messages.append(
                {"role": "assistant", "content": self.trim_message(message)}
            )

        user_messages = [
            *self.messages_user,
            *self.messages_block,
        ]
        user_messages = [self.trim_message(message) for message in user_messages]
        messages.append({"role": "user", "content": "\n".join(user_messages)})

        logger.debug(f"Messages: {messages}")

        # trim spaces from each line
        return messages
