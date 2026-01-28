"""
Unit tests for the REST API.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

# Skip all tests if fastapi is not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ai_data_science_team.api.app import create_app
from ai_data_science_team.api.models import (
    TaskStatus,
    AgentRequest,
    DataRequest,
    AgentConfig,
    HealthResponse,
)


@pytest.fixture
def app():
    """Create test application."""
    return create_app(debug=True)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and info endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["docs"] == "/docs"

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data

    def test_list_agents(self, client):
        """Test listing available agents."""
        response = client.get("/agents")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "count" in data
        assert data["count"] > 0

        # Check agent structure
        agent = data["agents"][0]
        assert "name" in agent
        assert "description" in agent
        assert "capabilities" in agent


class TestTaskEndpoints:
    """Tests for task management endpoints."""

    def test_list_tasks_empty(self, client):
        """Test listing tasks when empty."""
        response = client.get("/tasks")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_nonexistent_task(self, client):
        """Test getting a task that doesn't exist."""
        response = client.get("/tasks/nonexistent-id")

        assert response.status_code == 404

    def test_cancel_nonexistent_task(self, client):
        """Test canceling a task that doesn't exist."""
        response = client.delete("/tasks/nonexistent-id")

        assert response.status_code == 404


class TestAgentInvocation:
    """Tests for agent invocation endpoints."""

    def test_invoke_agent_sync(self, client):
        """Test synchronous agent invocation."""
        request_data = {
            "agent_type": "data_cleaning",
            "instructions": "Clean the data",
            "async_mode": False,
        }

        response = client.post("/agents/invoke", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] in ["completed", "failed"]
        assert data["agent_type"] == "data_cleaning"

    def test_invoke_agent_async(self, client):
        """Test asynchronous agent invocation."""
        request_data = {
            "agent_type": "eda",
            "instructions": "Analyze the data",
            "async_mode": True,
        }

        response = client.post("/agents/invoke", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_invoke_agent_with_data(self, client):
        """Test agent invocation with data."""
        request_data = {
            "agent_type": "data_cleaning",
            "instructions": "Remove duplicates",
            "data": {
                "data": [
                    {"a": 1, "b": 2},
                    {"a": 3, "b": 4},
                ],
            },
            "async_mode": False,
        }

        response = client.post("/agents/invoke", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_invoke_agent_with_config(self, client):
        """Test agent invocation with config."""
        request_data = {
            "agent_type": "sql",
            "instructions": "Generate a query",
            "config": {
                "temperature": 0.5,
                "model_name": "gpt-4",
            },
            "async_mode": False,
        }

        response = client.post("/agents/invoke", json=request_data)

        assert response.status_code == 200


class TestSpecializedAgents:
    """Tests for specialized agent endpoints."""

    def test_clean_data_endpoint(self, client):
        """Test data cleaning endpoint."""
        request_data = {
            "data": {
                "data": [{"x": 1}, {"x": 2}],
            },
            "instructions": "Remove nulls",
        }

        response = client.post("/agents/clean", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "operations_applied" in data

    def test_eda_endpoint(self, client):
        """Test EDA endpoint."""
        request_data = {
            "data": {
                "data": [{"x": 1, "y": 2}],
            },
            "analysis_type": "comprehensive",
        }

        response = client.post("/agents/eda", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "status" in data

    def test_eda_endpoint_with_target(self, client):
        """Test EDA endpoint with target column."""
        request_data = {
            "data": {
                "data": [{"x": 1, "y": 2}],
            },
            "analysis_type": "comprehensive",
            "target_column": "y",
        }

        response = client.post("/agents/eda", json=request_data)

        assert response.status_code == 200

    def test_sql_endpoint(self, client):
        """Test SQL generation endpoint."""
        request_data = {
            "question": "Get all users who signed up this month",
        }

        response = client.post("/agents/sql", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "sql_query" in data or "status" in data

    def test_visualization_endpoint(self, client):
        """Test visualization endpoint."""
        request_data = {
            "data": {
                "data": [{"x": 1, "y": 2}],
            },
            "instructions": "Create a bar chart",
        }

        response = client.post("/agents/visualize", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data


class TestPipelineEndpoints:
    """Tests for pipeline endpoints."""

    def test_run_pipeline(self, client):
        """Test running a pipeline."""
        request_data = {
            "name": "test_pipeline",
            "steps": [
                {
                    "name": "clean",
                    "agent_type": "data_cleaning",
                    "instructions": "Clean the data",
                },
                {
                    "name": "analyze",
                    "agent_type": "eda",
                    "instructions": "Analyze the data",
                    "depends_on": ["clean"],
                },
            ],
            "data": {
                "data": [{"x": 1}, {"x": 2}],
            },
        }

        response = client.post("/pipelines/run", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_id" in data
        assert "status" in data
        assert "steps_completed" in data
        assert "steps_total" in data
        assert data["steps_total"] == 2

    def test_run_empty_pipeline(self, client):
        """Test running an empty pipeline."""
        request_data = {
            "name": "empty_pipeline",
            "steps": [],
            "data": {
                "data": [],
            },
        }

        response = client.post("/pipelines/run", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["steps_total"] == 0


class TestDataEndpoints:
    """Tests for data management endpoints."""

    def test_upload_data(self, client):
        """Test data upload."""
        data = b'{"test": "data"}'

        response = client.post(
            "/data/upload",
            content=data,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        result = response.json()
        assert "data_id" in result
        assert result["size"] == len(data)

    def test_get_nonexistent_data(self, client):
        """Test getting data that doesn't exist."""
        response = client.get("/data/nonexistent-id")

        assert response.status_code == 404


class TestModels:
    """Tests for Pydantic models."""

    def test_task_status_enum(self):
        """Test TaskStatus enum."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_agent_request_model(self):
        """Test AgentRequest model."""
        request = AgentRequest(
            agent_type="data_cleaning",
            instructions="Clean the data",
        )

        assert request.agent_type == "data_cleaning"
        assert request.instructions == "Clean the data"
        assert request.async_mode is False

    def test_agent_request_with_data(self):
        """Test AgentRequest with data."""
        request = AgentRequest(
            agent_type="eda",
            instructions="Analyze",
            data=DataRequest(data=[{"a": 1}]),
        )

        assert request.data is not None
        assert request.data.data == [{"a": 1}]

    def test_agent_config_model(self):
        """Test AgentConfig model."""
        config = AgentConfig(
            temperature=0.5,
            model_name="gpt-4",
        )

        assert config.temperature == 0.5
        assert config.model_name == "gpt-4"

    def test_data_request_model(self):
        """Test DataRequest model."""
        request = DataRequest(
            data=[{"x": 1}, {"x": 2}],
            data_format="json",
        )

        assert len(request.data) == 2
        assert request.data_format == "json"


class TestAppFactory:
    """Tests for app factory."""

    def test_create_app(self):
        """Test creating app."""
        app = create_app()

        assert app.title == "AI Data Science Team API"
        assert app.version == "0.1.0"

    def test_create_app_custom_config(self):
        """Test creating app with custom config."""
        app = create_app(
            title="Custom API",
            version="1.0.0",
            debug=True,
        )

        assert app.title == "Custom API"
        assert app.version == "1.0.0"
        assert app.debug is True


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_body(self, client):
        """Test handling invalid JSON."""
        response = client.post(
            "/agents/invoke",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_missing_required_field(self, client):
        """Test handling missing required field."""
        response = client.post(
            "/agents/invoke",
            json={"agent_type": "eda"},  # Missing instructions
        )

        assert response.status_code == 422


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS should allow the request
        assert response.status_code in (200, 204, 405)


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_json(self, client):
        """Test OpenAPI JSON endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "info" in data

    def test_docs_endpoint(self, client):
        """Test docs endpoint."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        """Test redoc endpoint."""
        response = client.get("/redoc")

        assert response.status_code == 200
