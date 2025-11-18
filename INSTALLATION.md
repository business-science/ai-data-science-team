# Installation Guide - Enhanced Fork

This guide shows how to install and set up the enhanced AI Data Science Team with custom extensions.

## Quick Install

```bash
# Clone this fork
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team

# Install the package
pip install -e .

# Install custom extensions dependencies
pip install scipy statsmodels shap flask flask-cors streamlit
```

---

## Detailed Installation

### Prerequisites

**Required:**
- Python 3.9 or higher
- pip (Python package manager)

**Optional:**
- Virtual environment (recommended)
- OpenAI API key (for using agents)

### Step 1: Clone the Repository

```bash
# Clone this enhanced fork
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Core Package

```bash
# Install the package in editable mode
pip install -e .
```

This installs:
- Original AI Data Science Team agents
- Core dependencies (LangChain, pandas, numpy, etc.)

### Step 4: Install Custom Extensions Dependencies

```bash
# For custom agents functionality
pip install scipy statsmodels shap

# For web apps and APIs (optional)
pip install streamlit flask flask-cors

# For machine learning extensions (optional)
pip install h2o mlflow pytimetk missingno sweetviz
```

### Step 5: Set Up API Keys

```bash
# Set OpenAI API key
export OPENAI_API_KEY='sk-your-key-here'

# Or create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

---

## Installation Options

### Option 1: Full Installation (Recommended)

Everything including web apps, APIs, and ML tools:

```bash
pip install -e .
pip install scipy statsmodels shap streamlit flask flask-cors h2o mlflow
```

### Option 2: Core + Custom Agents Only

Just the agents without web interfaces:

```bash
pip install -e .
pip install scipy statsmodels shap
```

### Option 3: Minimal Installation

Just the original package without custom extensions:

```bash
pip install -e .
```

---

## Verify Installation

### Test Core Package

```python
# test_core.py
from ai_data_science_team.agents import DataWranglingAgent
from langchain_openai import ChatOpenAI
import os

os.environ['OPENAI_API_KEY'] = 'your-key-here'
llm = ChatOpenAI(model="gpt-4")

agent = DataWranglingAgent(model=llm)
print("✅ Core package working!")
```

### Test Custom Extensions

```python
# test_custom.py
from custom.agents import DataQualityAgent
from langchain_openai import ChatOpenAI
import pandas as pd
import os

os.environ['OPENAI_API_KEY'] = 'your-key-here'
llm = ChatOpenAI(model="gpt-4")

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 70000, 80000]
})

# Test agent
agent = DataQualityAgent(model=llm)
result = agent.quick_check(df)
print("✅ Custom extensions working!")
print(result)
```

Run tests:
```bash
python test_core.py
python test_custom.py
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'custom'`

**Solution:**
```python
# Add project to Python path
import sys
sys.path.insert(0, '/full/path/to/ai-data-science-team')

from custom.agents import DataQualityAgent
```

Or install in editable mode:
```bash
pip install -e .
```

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'scipy'` (or statsmodels, shap, etc.)

**Solution:**
```bash
# Install missing dependencies
pip install scipy statsmodels shap

# Or install all at once
pip install scipy statsmodels shap streamlit flask flask-cors
```

### SHAP Installation Issues

**Problem:** SHAP fails to install on Windows/Mac

**Solution:**
```bash
# Try installing with conda
conda install -c conda-forge shap

# Or use pre-built wheels
pip install --upgrade pip
pip install shap
```

### H2O Installation Issues

**Problem:** H2O installation fails

**Solution:**
```bash
# Install specific version
pip install h2o==3.42.0.3

# Or skip H2O if you don't need it
# (Custom agents don't require H2O)
```

### API Key Not Found

**Problem:** `OpenAI API key not found`

**Solution:**
```bash
# Set environment variable (Linux/Mac)
export OPENAI_API_KEY='sk-...'

# Set environment variable (Windows)
set OPENAI_API_KEY=sk-...

# Or use .env file
pip install python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env

# In Python:
from dotenv import load_dotenv
load_dotenv()
```

---

## Platform-Specific Instructions

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Clone and install
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install scipy statsmodels shap streamlit flask flask-cors
```

### Windows

```powershell
# Install Python from python.org (3.9+)

# Clone and install
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team
python -m venv venv
venv\Scripts\activate
pip install -e .
pip install scipy statsmodels shap streamlit flask flask-cors
```

### Linux (Ubuntu/Debian)

```bash
# Install Python and dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git

# Clone and install
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install scipy statsmodels shap streamlit flask flask-cors
```

---

## Docker Installation (Optional)

For a containerized setup:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy project
COPY . /app

# Install dependencies
RUN pip install -e . && \
    pip install scipy statsmodels shap streamlit flask flask-cors

# Expose ports for Streamlit and Flask
EXPOSE 8501 5000

CMD ["bash"]
```

Build and run:
```bash
docker build -t ai-data-science-team .
docker run -it -e OPENAI_API_KEY='sk-...' ai-data-science-team
```

---

## Google Colab Installation

For use in Google Colab:

```python
# Install from GitHub
!git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
%cd ai-data-science-team
!pip install -q -e .
!pip install -q scipy statsmodels shap

# Set API key
import os
os.environ['OPENAI_API_KEY'] = 'sk-your-key-here'

# Import and use
from custom.agents import DataQualityAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = DataQualityAgent(model=llm)
```

---

## Updating the Package

### Update from This Fork

```bash
# Pull latest changes
git pull origin master

# Reinstall dependencies
pip install -e . --upgrade
pip install scipy statsmodels shap --upgrade
```

### Sync with Upstream (Original Package)

```bash
# Fetch from original repository
git fetch upstream main

# Review changes
git log HEAD..upstream/main --oneline

# Merge if desired
git merge upstream/main

# Reinstall
pip install -e . --upgrade
```

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove cloned repository
cd ..
rm -rf ai-data-science-team

# Or just uninstall package
pip uninstall ai-data-science-team
```

---

## Next Steps

After installation:

1. **Read the documentation:**
   - [Fork Features](FORK_FEATURES.md) - What's new in this fork
   - [Custom Extensions Guide](custom/docs/CUSTOM_EXTENSIONS_GUIDE.md) - Detailed agent docs
   - [Usage Examples](custom/examples/README.md) - How to use the agents

2. **Try examples:**
   ```bash
   # Streamlit web app
   streamlit run custom/examples/streamlit_quality_checker.py

   # CLI tool
   python custom/examples/cli_quality_check.py data.csv

   # REST API
   python custom/examples/api_server.py
   ```

3. **Build something:**
   - See templates in `custom/agents/TEMPLATE_agent.py`
   - Check workflow guide in `CUSTOM_WORKFLOW.md`

---

## Support

**Installation Issues:**
- Check [Troubleshooting](#troubleshooting) section above
- Open issue: [GitHub Issues](https://github.com/NicoLeeVaz/ai-data-science-team/issues)

**Documentation:**
- [Fork Features](FORK_FEATURES.md)
- [Custom Extensions Guide](custom/docs/CUSTOM_EXTENSIONS_GUIDE.md)
- [Usage Examples](custom/examples/README.md)

---

**Quick Reference:**

```bash
# Complete installation
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
pip install scipy statsmodels shap streamlit flask flask-cors
export OPENAI_API_KEY='sk-your-key-here'

# Test it
python -c "from custom.agents import DataQualityAgent; print('✅ Ready!')"
```
