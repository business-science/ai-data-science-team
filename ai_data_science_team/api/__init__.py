"""
REST API server for AI Data Science Team.

This module provides a FastAPI-based REST API for accessing
AI Data Science Team functionality via HTTP endpoints.
"""

from ai_data_science_team.api.app import create_app, get_app
from ai_data_science_team.api.models import (
    AgentRequest,
    AgentResponse,
    DataRequest,
    HealthResponse,
    TaskStatus,
)
from ai_data_science_team.api.routes import router

__all__ = [
    # App factory
    "create_app",
    "get_app",
    # Models
    "AgentRequest",
    "AgentResponse",
    "DataRequest",
    "HealthResponse",
    "TaskStatus",
    # Router
    "router",
]
