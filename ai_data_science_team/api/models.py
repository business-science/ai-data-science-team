"""
Pydantic models for the REST API.

This module defines request and response models for the API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class TaskStatus(str, Enum):
    """Status of an async task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DataRequest(BaseModel):
    """Request model for data input."""
    model_config = ConfigDict(extra="allow")

    data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    data_url: Optional[str] = None
    data_format: str = "json"
    columns: Optional[List[str]] = None


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    model_config = ConfigDict(extra="allow")

    model_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """Request model for agent invocation."""
    model_config = ConfigDict(extra="allow")

    agent_type: str = Field(..., description="Type of agent to invoke")
    instructions: str = Field(..., description="Instructions for the agent")
    data: Optional[DataRequest] = None
    config: Optional[AgentConfig] = None
    async_mode: bool = Field(default=False, description="Run asynchronously")
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to call when async task completes"
    )


class AgentResponse(BaseModel):
    """Response model for agent invocation."""
    model_config = ConfigDict(extra="allow")

    task_id: str
    status: TaskStatus
    agent_type: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class DataCleaningRequest(BaseModel):
    """Request for data cleaning agent."""
    model_config = ConfigDict(extra="allow")

    data: DataRequest
    instructions: Optional[str] = "Clean the data"
    operations: Optional[List[str]] = None
    config: Optional[AgentConfig] = None


class DataCleaningResponse(BaseModel):
    """Response from data cleaning agent."""
    task_id: str
    status: TaskStatus
    cleaned_data: Optional[List[Dict[str, Any]]] = None
    operations_applied: List[str] = Field(default_factory=list)
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    code_generated: Optional[str] = None


class EDARequest(BaseModel):
    """Request for EDA agent."""
    model_config = ConfigDict(extra="allow")

    data: DataRequest
    analysis_type: str = "comprehensive"
    target_column: Optional[str] = None
    config: Optional[AgentConfig] = None


class EDAResponse(BaseModel):
    """Response from EDA agent."""
    task_id: str
    status: TaskStatus
    summary: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    correlations: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[str]] = None
    insights: Optional[List[str]] = None


class SQLQueryRequest(BaseModel):
    """Request for SQL query agent."""
    model_config = ConfigDict(extra="allow")

    question: str = Field(..., description="Natural language question")
    database_url: Optional[str] = None
    schema_info: Optional[Dict[str, Any]] = None
    config: Optional[AgentConfig] = None


class SQLQueryResponse(BaseModel):
    """Response from SQL query agent."""
    task_id: str
    status: TaskStatus
    sql_query: Optional[str] = None
    result: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None
    rows_returned: Optional[int] = None


class VisualizationRequest(BaseModel):
    """Request for visualization agent."""
    model_config = ConfigDict(extra="allow")

    data: DataRequest
    chart_type: Optional[str] = None
    instructions: str = Field(..., description="What to visualize")
    format: str = "png"
    config: Optional[AgentConfig] = None


class VisualizationResponse(BaseModel):
    """Response from visualization agent."""
    task_id: str
    status: TaskStatus
    chart_url: Optional[str] = None
    chart_base64: Optional[str] = None
    code_generated: Optional[str] = None


class PipelineStep(BaseModel):
    """A step in a data pipeline."""
    name: str
    agent_type: str
    instructions: str
    depends_on: List[str] = Field(default_factory=list)
    config: Optional[AgentConfig] = None


class PipelineRequest(BaseModel):
    """Request for running a data pipeline."""
    model_config = ConfigDict(extra="allow")

    name: str
    steps: List[PipelineStep]
    data: DataRequest
    parallel: bool = True
    config: Optional[AgentConfig] = None


class PipelineResponse(BaseModel):
    """Response from pipeline execution."""
    pipeline_id: str
    status: TaskStatus
    steps_completed: int = 0
    steps_total: int = 0
    results: Dict[str, Any] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    execution_time: Optional[float] = None


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    provider: str
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)


class AgentInfo(BaseModel):
    """Information about an available agent."""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    capabilities: List[str] = Field(default_factory=list)


class ListAgentsResponse(BaseModel):
    """Response listing available agents."""
    agents: List[AgentInfo]
    count: int
