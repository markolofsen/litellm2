from typing import List, Optional, Type, TypeVar, Generic
from pydantic import Field, BaseModel

import json
from datetime import datetime

from ..config.config import save_response_log

T = TypeVar('T', bound=BaseModel)


class Request(BaseModel):
    """Request parameters."""
    model: str = Field(default="gpt-4o", description="Model name")
    online: bool = Field(default=False, description="Use online model")
    messages: List[dict] = Field(default=[], description="List of messages for the chat")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens")
    fallbacks: Optional[List[str]] = Field(default=[], description="Fallback models in case of failure")
    stream: Optional[bool] = Field(default=False, description="Enable streaming response")
    cache_prompt: Optional[bool] = Field(default=False, description="Cache prompt for reuse")
    max_budget: Optional[float] = Field(None, description="Budget limit in USD for the request")
    answer_model: Type[BaseModel] = Field(..., description="Custom model for the answer content")
    verbose: Optional[bool] = Field(default=False, description="Enable verbose logging")
    logs: Optional[bool] = Field(default=False, description="Enable logging to file")

    def model_dump_json(self, **kwargs) -> str:
        data = self.model_dump()
        # Convert answer_model class to its name for JSON serialization
        data['answer_model'] = self.answer_model.__name__
        return json.dumps(data, **kwargs)


class Meta(BaseModel):
    """Metadata about the response processing."""
    response_time_seconds: Optional[float] = Field(None, description="Time taken to generate response in seconds")
    cache_hit: Optional[bool] = Field(None, description="Whether the response was retrieved from cache")
    model_used: Optional[str] = Field(None, description="The model that actually generated the response")
    token_count: Optional[int] = Field(None, description="Number of tokens in the response")
    request_timestamp: Optional[float] = Field(None, description="Timestamp when request was sent")

    def __str__(self):
        """String representation for easy logging and display."""
        cache_status = "Yes" if self.cache_hit else "No" if self.cache_hit is False else "Unknown"
        return (
            f"Response time: {self.response_time_seconds:.3f}s | "
            f"From cache: {cache_status} | "
            f"Model: {self.model_used or 'Unknown'}"
        )


class FullResponse(BaseModel, Generic[T]):
    """Complete response including request, meta and response data."""
    request: Request
    meta: Meta
    response: T
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def create(cls, request: Request, meta: Meta, response: T) -> 'FullResponse[T]':
        return cls(
            request=request,
            meta=meta,
            response=response
        )

    def save_to_log(self) -> str:
        """Save the full response to a JSON log file."""
        return save_response_log(
            response_data=self.model_dump(),
            model_name=self.meta.model_used or "unknown"
        )
