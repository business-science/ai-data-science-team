# Custom Extensions

This directory contains your custom extensions to the AI Data Science Team project.

## Directory Structure

```
custom/
├── agents/         # Your custom agent implementations
├── tools/          # Custom tools and utilities
├── apps/           # Custom Streamlit applications
├── examples/       # Your notebooks and example scripts
├── experiments/    # Work-in-progress / experimental code
├── private/        # Private work (gitignored - never committed)
└── docs/           # Custom documentation
```

## Purpose

This structure keeps your custom work **completely separated** from upstream code, making it easy to:

1. **Merge upstream updates** without conflicts
2. **Maintain your innovations** independently
3. **Control what gets committed** to your fork

## Directories Explained

### `agents/`
Your custom agent implementations that extend or replace core functionality.

**Example:**
```python
# custom/agents/my_special_agent.py
from ai_data_science_team.agents import BaseAgent

class MySpecialAgent(BaseAgent):
    # Your custom agent logic
    pass
```

### `tools/`
Custom tools, integrations, or utility functions.

**Example:**
```python
# custom/tools/my_integration.py
def integrate_with_custom_api():
    # Your integration code
    pass
```

### `apps/`
Your custom Streamlit or other web applications.

### `examples/`
Notebooks and scripts demonstrating your custom features.

### `experiments/`
Sandbox for trying new ideas - can be messy, partially committed to git.

### `private/`
**NEVER committed to git** - API keys, proprietary code, client work, etc.
This is in `.gitignore` for safety.

### `docs/`
Documentation for your custom features and workflows.

## Workflow Guidelines

### When building custom features:

1. **Start in `experiments/`** - Try things out
2. **Move to proper directory** - Once stable, move to `agents/`, `tools/`, etc.
3. **Keep sensitive in `private/`** - Anything proprietary or with credentials

### When pulling upstream updates:

```bash
# Fetch latest from original project
git fetch upstream main

# Review what changed
git log HEAD..upstream/main --oneline

# Merge if desired (your custom/ directory won't conflict)
git merge upstream/main
```

### Git Safety

- **Upstream push is DISABLED** - You cannot accidentally push to the original repo
- **private/ is gitignored** - Your sensitive work stays local
- **Your fork is independent** - All commits go to your repository only

## Integration with Core Package

To use your custom code alongside the core package:

```python
# Import core functionality
from ai_data_science_team.agents import DataWranglingAgent

# Import your custom extensions
import sys
sys.path.append('/path/to/project/custom')
from agents.my_special_agent import MySpecialAgent

# Use both together
core_agent = DataWranglingAgent(...)
custom_agent = MySpecialAgent(...)
```

## Best Practices

1. **Mirror core structure** - Use similar patterns as the main package
2. **Document your additions** - Future you will thank present you
3. **Test independently** - Don't assume upstream tests cover your code
4. **Version control wisely** - Commit custom work, but not secrets
5. **Branch for experiments** - Use feature branches for big custom changes

## Questions?

See `CUSTOM_WORKFLOW.md` in the root directory for detailed workflow instructions.
