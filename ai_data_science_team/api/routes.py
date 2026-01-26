"""
API routes for AI Data Science Team.

This module defines all REST API endpoints.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ai_data_science_team.api.models import (
    AgentInfo,
    AgentRequest,
    AgentResponse,
    DataCleaningRequest,
    DataCleaningResponse,
    EDARequest,
    EDAResponse,
    ErrorResponse,
    HealthResponse,
    ListAgentsResponse,
    PipelineRequest,
    PipelineResponse,
    SQLQueryRequest,
    SQLQueryResponse,
    TaskResponse,
    TaskStatus,
    VisualizationRequest,
    VisualizationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Task storage (in production, use Redis or database)
_tasks: Dict[str, Dict[str, Any]] = {}
_results: Dict[str, Any] = {}


def get_task_store(request: Request) -> Dict[str, Dict[str, Any]]:
    """Get task store from app state."""
    return getattr(request.app.state, "tasks", _tasks)


def get_result_store(request: Request) -> Dict[str, Any]:
    """Get result store from app state."""
    return getattr(request.app.state, "results", _results)


# =============================================================================
# Health and Info Endpoints
# =============================================================================


@router.get("/", tags=["info"])
async def root():
    """API root endpoint."""
    return {
        "name": "AI Data Science Team API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow(),
        components={
            "api": "healthy",
            "agents": "available",
        },
    )


@router.get("/agents", response_model=ListAgentsResponse, tags=["info"])
async def list_agents():
    """List available agents."""
    agents = [
        AgentInfo(
            name="data_cleaning",
            description="Cleans and preprocesses data",
            capabilities=["missing_values", "duplicates", "outliers", "type_conversion"],
        ),
        AgentInfo(
            name="data_wrangling",
            description="Transforms and reshapes data",
            capabilities=["filtering", "aggregation", "joining", "pivoting"],
        ),
        AgentInfo(
            name="eda",
            description="Performs exploratory data analysis",
            capabilities=["statistics", "correlations", "distributions", "visualizations"],
        ),
        AgentInfo(
            name="sql",
            description="Generates and executes SQL queries",
            capabilities=["query_generation", "schema_analysis", "optimization"],
        ),
        AgentInfo(
            name="visualization",
            description="Creates data visualizations",
            capabilities=["charts", "plots", "dashboards"],
        ),
        AgentInfo(
            name="feature_engineering",
            description="Creates and transforms features",
            capabilities=["encoding", "scaling", "feature_creation"],
        ),
    ]
    return ListAgentsResponse(agents=agents, count=len(agents))


# =============================================================================
# Task Management Endpoints
# =============================================================================


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: str, request: Request):
    """Get task status and result."""
    tasks = get_task_store(request)

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]
    return TaskResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error"),
        created_at=task["created_at"],
        updated_at=task.get("updated_at", task["created_at"]),
        completed_at=task.get("completed_at"),
    )


@router.get("/tasks", response_model=List[TaskResponse], tags=["tasks"])
async def list_tasks(
    request: Request,
    status: Optional[TaskStatus] = None,
    limit: int = 100,
):
    """List all tasks."""
    tasks = get_task_store(request)

    result = []
    for task_id, task in list(tasks.items())[:limit]:
        if status and task["status"] != status:
            continue
        result.append(TaskResponse(
            task_id=task_id,
            status=task["status"],
            progress=task.get("progress"),
            result=task.get("result"),
            error=task.get("error"),
            created_at=task["created_at"],
            updated_at=task.get("updated_at", task["created_at"]),
            completed_at=task.get("completed_at"),
        ))

    return result


@router.delete("/tasks/{task_id}", tags=["tasks"])
async def cancel_task(task_id: str, request: Request):
    """Cancel a running task."""
    tasks = get_task_store(request)

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]
    if task["status"] in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in {task['status']} status"
        )

    task["status"] = TaskStatus.CANCELLED
    task["updated_at"] = datetime.utcnow()

    return {"message": f"Task {task_id} cancelled"}


# =============================================================================
# Agent Invocation Endpoints
# =============================================================================


async def run_agent_task(
    task_id: str,
    agent_type: str,
    instructions: str,
    data: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
    tasks: Dict[str, Dict[str, Any]],
):
    """Background task to run an agent."""
    import pandas as pd

    tasks[task_id]["status"] = TaskStatus.RUNNING
    tasks[task_id]["updated_at"] = datetime.utcnow()

    try:
        # Prepare data
        df = None
        if data and data.get("data"):
            df = pd.DataFrame(data["data"])

        # Get agent based on type
        result = await execute_agent(agent_type, instructions, df, config)

        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["result"] = result
        tasks[task_id]["completed_at"] = datetime.utcnow()

    except Exception as e:
        logger.error(f"Agent task {task_id} failed: {e}", exc_info=True)
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.utcnow()

    tasks[task_id]["updated_at"] = datetime.utcnow()


async def execute_agent(
    agent_type: str,
    instructions: str,
    data: Optional[Any],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute an agent and return results."""
    # This is a placeholder implementation
    # In production, this would invoke actual agents

    # Simulate processing time
    await asyncio.sleep(0.1)

    if agent_type == "data_cleaning":
        return {
            "operations_applied": ["remove_duplicates", "fill_missing"],
            "rows_before": len(data) if data is not None else 0,
            "rows_after": len(data) if data is not None else 0,
            "code_generated": "# Data cleaning code\ndf = df.drop_duplicates()",
        }
    elif agent_type == "eda":
        return {
            "summary": {"rows": 100, "columns": 5},
            "statistics": {"mean": 50, "std": 10},
            "insights": ["Data is normally distributed"],
        }
    elif agent_type == "sql":
        return {
            "sql_query": f"-- Generated for: {instructions}\nSELECT * FROM table",
            "explanation": "This query retrieves all records",
        }
    elif agent_type == "visualization":
        return {
            "chart_type": "bar",
            "code_generated": "import matplotlib.pyplot as plt\nplt.bar(x, y)",
        }
    else:
        return {
            "message": f"Agent {agent_type} executed successfully",
            "instructions": instructions,
        }


@router.post("/agents/invoke", response_model=AgentResponse, tags=["agents"])
async def invoke_agent(
    request_data: AgentRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Invoke an agent to perform a task.

    This endpoint can run synchronously or asynchronously based on the
    async_mode parameter.
    """
    tasks = get_task_store(request)

    task_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Create task record
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "agent_type": request_data.agent_type,
        "instructions": request_data.instructions,
        "created_at": now,
        "updated_at": now,
    }

    data_dict = request_data.data.model_dump() if request_data.data else None
    config_dict = request_data.config.model_dump() if request_data.config else None

    if request_data.async_mode:
        # Run in background
        background_tasks.add_task(
            run_agent_task,
            task_id,
            request_data.agent_type,
            request_data.instructions,
            data_dict,
            config_dict,
            tasks,
        )

        return AgentResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            agent_type=request_data.agent_type,
            created_at=now,
        )
    else:
        # Run synchronously
        await run_agent_task(
            task_id,
            request_data.agent_type,
            request_data.instructions,
            data_dict,
            config_dict,
            tasks,
        )

        task = tasks[task_id]
        return AgentResponse(
            task_id=task_id,
            status=task["status"],
            agent_type=request_data.agent_type,
            result=task.get("result"),
            error=task.get("error"),
            created_at=task["created_at"],
            completed_at=task.get("completed_at"),
        )


# =============================================================================
# Specialized Agent Endpoints
# =============================================================================


@router.post("/agents/clean", response_model=DataCleaningResponse, tags=["agents"])
async def clean_data(
    request_data: DataCleaningRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Clean data using the data cleaning agent."""
    tasks = get_task_store(request)

    task_id = str(uuid.uuid4())
    now = datetime.utcnow()

    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "agent_type": "data_cleaning",
        "created_at": now,
        "updated_at": now,
    }

    data_dict = request_data.data.model_dump() if request_data.data else None
    config_dict = request_data.config.model_dump() if request_data.config else None

    await run_agent_task(
        task_id,
        "data_cleaning",
        request_data.instructions or "Clean the data",
        data_dict,
        config_dict,
        tasks,
    )

    task = tasks[task_id]
    result = task.get("result", {})

    return DataCleaningResponse(
        task_id=task_id,
        status=task["status"],
        operations_applied=result.get("operations_applied", []),
        rows_before=result.get("rows_before"),
        rows_after=result.get("rows_after"),
        code_generated=result.get("code_generated"),
    )


@router.post("/agents/eda", response_model=EDAResponse, tags=["agents"])
async def analyze_data(
    request_data: EDARequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Perform exploratory data analysis."""
    tasks = get_task_store(request)

    task_id = str(uuid.uuid4())
    now = datetime.utcnow()

    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "agent_type": "eda",
        "created_at": now,
        "updated_at": now,
    }

    data_dict = request_data.data.model_dump() if request_data.data else None
    config_dict = request_data.config.model_dump() if request_data.config else None

    instructions = f"Perform {request_data.analysis_type} analysis"
    if request_data.target_column:
        instructions += f" focusing on {request_data.target_column}"

    await run_agent_task(
        task_id,
        "eda",
        instructions,
        data_dict,
        config_dict,
        tasks,
    )

    task = tasks[task_id]
    result = task.get("result", {})

    return EDAResponse(
        task_id=task_id,
        status=task["status"],
        summary=result.get("summary"),
        statistics=result.get("statistics"),
        correlations=result.get("correlations"),
        insights=result.get("insights"),
    )


@router.post("/agents/sql", response_model=SQLQueryResponse, tags=["agents"])
async def generate_sql(
    request_data: SQLQueryRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Generate SQL from natural language."""
    tasks = get_task_store(request)

    task_id = str(uuid.uuid4())
    now = datetime.utcnow()

    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "agent_type": "sql",
        "created_at": now,
        "updated_at": now,
    }

    config_dict = request_data.config.model_dump() if request_data.config else None

    await run_agent_task(
        task_id,
        "sql",
        request_data.question,
        {"schema": request_data.schema_info},
        config_dict,
        tasks,
    )

    task = tasks[task_id]
    result = task.get("result", {})

    return SQLQueryResponse(
        task_id=task_id,
        status=task["status"],
        sql_query=result.get("sql_query"),
        explanation=result.get("explanation"),
    )


@router.post("/agents/visualize", response_model=VisualizationResponse, tags=["agents"])
async def create_visualization(
    request_data: VisualizationRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Create a data visualization."""
    tasks = get_task_store(request)

    task_id = str(uuid.uuid4())
    now = datetime.utcnow()

    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "agent_type": "visualization",
        "created_at": now,
        "updated_at": now,
    }

    data_dict = request_data.data.model_dump() if request_data.data else None
    config_dict = request_data.config.model_dump() if request_data.config else None

    await run_agent_task(
        task_id,
        "visualization",
        request_data.instructions,
        data_dict,
        config_dict,
        tasks,
    )

    task = tasks[task_id]
    result = task.get("result", {})

    return VisualizationResponse(
        task_id=task_id,
        status=task["status"],
        code_generated=result.get("code_generated"),
    )


# =============================================================================
# Pipeline Endpoints
# =============================================================================


@router.post("/pipelines/run", response_model=PipelineResponse, tags=["pipelines"])
async def run_pipeline(
    request_data: PipelineRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Run a data processing pipeline."""
    tasks = get_task_store(request)

    pipeline_id = str(uuid.uuid4())
    now = datetime.utcnow()

    tasks[pipeline_id] = {
        "status": TaskStatus.PENDING,
        "type": "pipeline",
        "name": request_data.name,
        "steps_total": len(request_data.steps),
        "steps_completed": 0,
        "created_at": now,
        "updated_at": now,
    }

    # Execute pipeline steps
    tasks[pipeline_id]["status"] = TaskStatus.RUNNING
    results = {}
    errors = {}

    for i, step in enumerate(request_data.steps):
        try:
            data_dict = request_data.data.model_dump() if request_data.data else None
            config_dict = step.config.model_dump() if step.config else None

            step_result = await execute_agent(
                step.agent_type,
                step.instructions,
                data_dict.get("data") if data_dict else None,
                config_dict,
            )
            results[step.name] = step_result
            tasks[pipeline_id]["steps_completed"] = i + 1

        except Exception as e:
            errors[step.name] = str(e)
            logger.error(f"Pipeline step {step.name} failed: {e}")

    # Update final status
    if errors:
        tasks[pipeline_id]["status"] = TaskStatus.FAILED
    else:
        tasks[pipeline_id]["status"] = TaskStatus.COMPLETED

    tasks[pipeline_id]["completed_at"] = datetime.utcnow()
    tasks[pipeline_id]["updated_at"] = datetime.utcnow()

    return PipelineResponse(
        pipeline_id=pipeline_id,
        status=tasks[pipeline_id]["status"],
        steps_completed=tasks[pipeline_id]["steps_completed"],
        steps_total=len(request_data.steps),
        results=results,
        errors=errors,
    )


# =============================================================================
# Data Endpoints
# =============================================================================


@router.post("/data/upload", tags=["data"])
async def upload_data(request: Request):
    """Upload data for processing."""
    # Get raw body
    body = await request.body()

    # Generate data ID
    data_id = str(uuid.uuid4())

    # Store data (in production, use proper storage)
    results = get_result_store(request)
    results[data_id] = {
        "data": body,
        "created_at": datetime.utcnow(),
    }

    return {
        "data_id": data_id,
        "message": "Data uploaded successfully",
        "size": len(body),
    }


@router.get("/data/{data_id}", tags=["data"])
async def get_data(data_id: str, request: Request):
    """Get uploaded data."""
    results = get_result_store(request)

    if data_id not in results:
        raise HTTPException(status_code=404, detail=f"Data {data_id} not found")

    return results[data_id]
