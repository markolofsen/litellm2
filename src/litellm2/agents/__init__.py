"""
AI Agent module with tool-based functionality

This module provides the building blocks for constructing AI agents with tools.
"""

# Export main components
from .simple_agent import SimpleAgent
# Экспортируем инструменты напрямую из smolagents для удобства
from smolagents import tool

__all__ = [
    'SimpleAgent',
    'tool'
]
