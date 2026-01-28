# AI Data Science Team - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Running the API Server](#running-the-api-server)
3. [Running the Streamlit App](#running-the-streamlit-app)
4. [Running Tests](#running-tests)
5. [Using Agents](#using-agents)
6. [Using the Cache System](#using-the-cache-system)
7. [Using Async/Parallel Execution](#using-asyncparallel-execution)
8. [Using Cloud Connectors](#using-cloud-connectors)
9. [Using the Plugin System](#using-the-plugin-system)

---

## Installation

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/business-science/ai-data-science-team.git
cd ai-data-science-team

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Install with Optional Dependencies
```bash
# Install with API server support
pip install -e ".[api]"

# Install with cloud connectors
pip install -e ".[cloud]"

# Install with machine learning tools
pip install -e ".[machine_learning]"

# Install everything
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

---

## Running the API Server

### Start the Server
```bash
# Using the CLI
ai-ds-team-api --host 127.0.0.1 --port 8000

# Or using Python module
python -m ai_data_science_team.api.cli --port 8000

# With auto-reload for development
ai-ds-team-api --port 8000 --reload
```

### API Endpoints
Once running, access:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/health

### Example API Calls
```bash
# List available agents
curl http://127.0.0.1:8000/agents

# Invoke data cleaning agent
curl -X POST http://127.0.0.1:8000/agents/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "data_cleaning",
    "instructions": "Remove duplicates and fill missing values",
    "data": {
      "data": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": null}]
    }
  }'

# Generate SQL from natural language
curl -X POST http://127.0.0.1:8000/agents/sql \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Find all users who signed up in the last 30 days"
  }'

# Run a pipeline
curl -X POST http://127.0.0.1:8000/pipelines/run \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_pipeline",
    "steps": [
      {"name": "clean", "agent_type": "data_cleaning", "instructions": "Clean data"},
      {"name": "analyze", "agent_type": "eda", "instructions": "Analyze", "depends_on": ["clean"]}
    ],
    "data": {"data": [{"x": 1}, {"x": 2}]}
  }'
```

---

## Running the Streamlit App

```bash
# Navigate to the apps directory
cd apps/ai-data-science-team

# Run the Streamlit app
streamlit run app.py

# Or from project root
streamlit run apps/ai-data-science-team/app.py
```

Access at: http://localhost:8501

---

## Running Tests

### Run All Tests
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ai_data_science_team --cov-report=html
```

### Run Specific Test Categories
```bash
# Run only unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_api.py

# Run specific test class
pytest tests/unit/test_api.py::TestHealthEndpoints

# Run specific test
pytest tests/unit/test_api.py::TestHealthEndpoints::test_health_check

# Run tests matching a pattern
pytest tests/ -k "cache"

# Run async tests
pytest tests/unit/test_async_ops.py
```

### Test Markers
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m integration
```

---

## Using Agents

### Using Ollama (Local LLM - No API Key Required)
```python
from langchain_ollama import ChatOllama
import pandas as pd

# Create local LLM with Ollama (make sure Ollama is running)
# Install Ollama: https://ollama.ai
# Pull a model: ollama pull llama3.2 or ollama pull mistral
llm = ChatOllama(model="llama3.2", temperature=0)

# Or use other local models
# llm = ChatOllama(model="mistral", temperature=0)
# llm = ChatOllama(model="codellama", temperature=0)
# llm = ChatOllama(model="deepseek-coder", temperature=0)
```

### Using OpenAI (Cloud LLM)
```python
import os
from langchain_openai import ChatOpenAI
import pandas as pd

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Import agents
from ai_data_science_team import (
    DataCleaningAgent,
    DataWranglingAgent,
    DataVisualizationAgent,
    SQLDatabaseAgent,
)

# Create sample data
df = pd.DataFrame({
    "name": ["Alice", "Bob", None, "Alice"],
    "age": [25, 30, 35, 25],
    "salary": [50000, None, 70000, 50000]
})

# Use Data Cleaning Agent
cleaning_agent = DataCleaningAgent(model=llm)
result = cleaning_agent.invoke({
    "data": df,
    "instructions": "Remove duplicates and fill missing values with appropriate defaults"
})
print(result)
```

### Available Agents
```python
from ai_data_science_team import (
    DataCleaningAgent,      # Clean and preprocess data
    DataWranglingAgent,     # Transform and reshape data
    DataVisualizationAgent, # Create visualizations
    SQLDatabaseAgent,       # Generate and run SQL queries
    FeatureEngineeringAgent,# Create features for ML
)

from ai_data_science_team.ds_agents import (
    EDAToolsAgent,          # Exploratory data analysis
)

from ai_data_science_team.ml_agents import (
    H2OMLAgent,             # AutoML with H2O
    MLflowToolsAgent,       # MLflow integration
)

from ai_data_science_team.multiagents import (
    SQLDataAnalyst,         # Multi-agent SQL analysis
    PandasDataAnalyst,      # Multi-agent pandas analysis
)
```

---

## Using the Cache System

### Basic Caching
```python
from ai_data_science_team.cache import (
    Cache,
    MemoryBackend,
    DiskBackend,
    cached,
)

# Create a memory cache
cache = Cache(backend=MemoryBackend(max_size=1000))

# Set and get values
cache.set("my_key", {"data": [1, 2, 3]}, ttl=3600)  # TTL in seconds
value = cache.get("my_key")

# Check existence
if cache.exists("my_key"):
    print("Key exists!")

# Delete
cache.delete("my_key")
```

### Using the @cached Decorator
```python
from ai_data_science_team.cache import cached, Cache, MemoryBackend
import pandas as pd

cache = Cache(backend=MemoryBackend())

@cached(cache=cache, ttl=3600)
def expensive_computation(df):
    # This result will be cached
    return df.describe()

df = pd.DataFrame({"a": range(1000)})
result = expensive_computation(df)  # Computed
result = expensive_computation(df)  # Returned from cache
```

### Disk-based Cache (Persistent)
```python
from ai_data_science_team.cache import Cache, DiskBackend

# Cache persists across restarts
cache = Cache(backend=DiskBackend(cache_dir=".cache/my_cache"))
cache.set("persistent_key", "persistent_value")
```

---

## Using Async/Parallel Execution

### Parallel Map
```python
from ai_data_science_team.async_ops import parallel_map

def process_item(x):
    return x ** 2

# Process items in parallel
results = parallel_map(process_item, [1, 2, 3, 4, 5], max_workers=4)
values = [r.result for r in results if r.succeeded]
print(values)  # [1, 4, 9, 16, 25]
```

### Parallel DataFrame Processing
```python
from ai_data_science_team.async_ops import parallel_apply
import pandas as pd

df = pd.DataFrame({"value": range(10000)})

def process_partition(partition):
    return partition * 2

# Process DataFrame in parallel partitions
result = parallel_apply(df, process_partition, n_partitions=4)
```

### Async Execution
```python
import asyncio
from ai_data_science_team.async_ops import AsyncExecutor, gather_results

async def fetch_data(url):
    # Simulate async operation
    await asyncio.sleep(0.1)
    return f"Data from {url}"

async def main():
    executor = AsyncExecutor(max_concurrency=5)

    # Run multiple tasks
    results = await executor.map(fetch_data, [
        "http://api1.com",
        "http://api2.com",
        "http://api3.com",
    ])

    for r in results:
        if r.succeeded:
            print(r.result)

asyncio.run(main())
```

### Rate Limiting and Retry
```python
from ai_data_science_team.async_ops import async_retry, rate_limit

@rate_limit(calls_per_second=10)
@async_retry(max_retries=3, delay=1.0)
async def call_api(data):
    # API call with rate limiting and automatic retry
    return await api.post(data)
```

---

## Using Cloud Connectors

### Snowflake
```python
from ai_data_science_team.connectors import SnowflakeConnector

# Connect to Snowflake
conn = SnowflakeConnector(
    account="your_account",
    user="your_user",
    password="your_password",
    database="your_db",
    warehouse="your_warehouse",
)

# Fetch data
df = conn.fetch("SELECT * FROM my_table LIMIT 100")

# Write data
conn.write(df, "new_table", if_exists="replace")

# Close connection
conn.disconnect()
```

### BigQuery
```python
from ai_data_science_team.connectors import BigQueryConnector

conn = BigQueryConnector(
    project="your-project-id",
    credentials_path="/path/to/credentials.json",
)

df = conn.fetch("SELECT * FROM `project.dataset.table`")
```

### PostgreSQL
```python
from ai_data_science_team.connectors import PostgresConnector

conn = PostgresConnector(
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="password",
)

df = conn.fetch("SELECT * FROM users")
```

### S3
```python
from ai_data_science_team.connectors import S3Connector

conn = S3Connector(bucket="my-bucket", region="us-east-1")

# Read files
df = conn.read_csv("path/to/file.csv")
df = conn.read_parquet("path/to/file.parquet")

# Write files
conn.write_csv(df, "output/data.csv")
conn.write_parquet(df, "output/data.parquet")
```

### Using the Factory
```python
from ai_data_science_team.connectors import get_connector

# Get connector by type
conn = get_connector("postgres", host="localhost", database="mydb")

# Or from URL
from ai_data_science_team.connectors.factory import get_connector_from_url
conn = get_connector_from_url("postgres://user:pass@localhost:5432/mydb")
```

---

## Using the Plugin System

### Creating a Custom Agent Plugin
```python
from ai_data_science_team.plugins import AgentPlugin, PluginMetadata, register_agent

@register_agent("my_custom_agent")
class MyCustomAgent(AgentPlugin):
    """My custom agent for specific tasks."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_custom_agent",
            version="1.0.0",
            description="Custom agent for XYZ tasks",
            author="Your Name",
        )

    def initialize(self, config: dict) -> None:
        self.model = config.get("model")

    def execute(self, input_data: dict) -> dict:
        # Your agent logic here
        return {"result": "processed"}
```

### Loading Plugins
```python
from ai_data_science_team.plugins import PluginLoader, get_registry

# Load plugins from a directory
loader = PluginLoader()
loader.load_from_directory("/path/to/plugins")

# Get registry
registry = get_registry()

# List available plugins
print(registry.list_agents())

# Get a plugin
agent_cls = registry.get_agent("my_custom_agent")
agent = agent_cls()
agent.initialize({"model": "gpt-4"})
result = agent.execute({"data": "input"})
```

---

## Environment Variables

Create a `.env` file (automatically ignored by git):

```bash
# =============================================================================
# LOCAL LLM (Ollama - Recommended for privacy)
# =============================================================================
# No API key needed! Just install Ollama and pull a model:
#   brew install ollama
#   ollama pull llama3.2
#   ollama pull mistral
#   ollama pull codellama
# Ollama runs locally at http://localhost:11434

OLLAMA_MODEL=llama3.2
# OLLAMA_MODEL=mistral
# OLLAMA_MODEL=codellama

# =============================================================================
# CLOUD LLMs (require API keys)
# =============================================================================

# OpenAI
OPENAI_API_KEY=sk-...

# Snowflake
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_DATABASE=...
SNOWFLAKE_WAREHOUSE=...

# BigQuery
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=mydb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=...
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Quick Reference

| Component | Command/Import |
|-----------|----------------|
| Run API | `ai-ds-team-api --port 8000` |
| Run Streamlit | `streamlit run apps/ai-data-science-team/app.py` |
| Run Tests | `pytest tests/` |
| Import Agents | `from ai_data_science_team import DataCleaningAgent` |
| Import Cache | `from ai_data_science_team.cache import Cache, cached` |
| Import Async | `from ai_data_science_team.async_ops import parallel_map` |
| Import Connectors | `from ai_data_science_team.connectors import get_connector` |
| Import Plugins | `from ai_data_science_team.plugins import register_agent` |
