# Quick Start Guide

Get started building custom features in 5 minutes!

## ⚡ Quick Setup Check

```bash
# Verify git protection is enabled
git remote -v
# Should show: upstream no_push (push)

# Check you're on the right branch
git branch
# Should show your current branch

# Verify gitignore is protecting private/
git status
# Should NOT show custom/private/ files
```

## 🚀 Your First Custom Agent

### Step 1: Copy the Template

```bash
cd custom/agents
cp TEMPLATE_agent.py my_first_agent.py
```

### Step 2: Edit the Template

Open `custom/agents/my_first_agent.py` and:

1. Rename `MyCustomAgent` to your agent name
2. Update the docstring
3. Modify the tools or add new ones
4. Customize the prompt

### Step 3: Test It

```python
# custom/examples/test_my_agent.py
import sys
sys.path.insert(0, '/path/to/ai-data-science-team')

from langchain_openai import ChatOpenAI
from custom.agents.my_first_agent import MyFirstAgent
import os

os.environ['OPENAI_API_KEY'] = 'your-key-here'  # Or use .env file

llm = ChatOpenAI(model="gpt-4")
agent = MyFirstAgent(model=llm)

result = agent.invoke("Test my agent")
print(result)
```

### Step 4: Commit Your Work

```bash
git add custom/
git commit -m "Add my first custom agent"
git push origin main  # Or your feature branch
```

## 🛠️ Common Tasks

### Create a Custom Tool

```python
# custom/tools/my_helpers.py

def preprocess_data(df):
    """Your custom preprocessing logic"""
    # ... your code ...
    return df

def custom_visualization(data, **kwargs):
    """Your custom viz logic"""
    # ... your code ...
    return fig
```

### Create a Custom App

```bash
# custom/apps/my_app.py
import streamlit as st
from ai_data_science_team.agents import DataWranglingAgent

st.title("My Custom App")

# Your Streamlit app code here
```

Run it:
```bash
streamlit run custom/apps/my_app.py
```

### Work with Sensitive Code

```python
# custom/private/secret_sauce.py
# This file will NEVER be committed

API_KEY = "your-secret-key"
PROPRIETARY_ALGORITHM = "..."
```

```python
# custom/agents/my_agent.py
# Import your secrets safely
import sys
sys.path.insert(0, 'custom')
from private.secret_sauce import API_KEY
```

## 📂 Where Things Go

| What | Where | Committed? |
|------|-------|-----------|
| Stable custom agents | `custom/agents/` | ✅ Yes |
| Custom tools | `custom/tools/` | ✅ Yes |
| Custom apps | `custom/apps/` | ✅ Yes |
| Example notebooks | `custom/examples/` | ✅ Yes |
| Work-in-progress | `custom/experiments/` | ✅ Yes (optional) |
| Sensitive code | `custom/private/` | ❌ NO - Auto-ignored |
| API keys, secrets | `custom/private/` | ❌ NO - Auto-ignored |
| Large data files | `custom/data/` | ❌ NO - Auto-ignored |

## 🎓 Learning by Example

### Extend an Existing Agent

```python
# custom/agents/enhanced_wrangling.py
from ai_data_science_team.agents import DataWranglingAgent

class EnhancedDataWranglingAgent(DataWranglingAgent):
    """Extended version with custom features"""

    def custom_merge_strategy(self, df1, df2):
        """My special merge logic"""
        # Your custom logic here
        pass
```

### Combine Multiple Agents

```python
# custom/agents/super_analyst.py
from ai_data_science_team.agents import DataWranglingAgent
from ai_data_science_team.agents import DataVisualizationAgent

class SuperAnalyst:
    def __init__(self, model):
        self.wrangler = DataWranglingAgent(model=model)
        self.visualizer = DataVisualizationAgent(model=model)

    def analyze(self, data):
        # Wrangle then visualize
        cleaned = self.wrangler.invoke(data)
        viz = self.visualizer.invoke(cleaned)
        return viz
```

## 🔄 Staying Updated

Pull latest from upstream regularly:

```bash
# Every week or so
git fetch upstream main
git merge upstream/main
git push origin main
```

Your custom code in `custom/` won't conflict!

## 🆘 Common Issues

### "Import Error: No module named 'custom'"

Fix:
```python
import sys
sys.path.insert(0, '/full/path/to/ai-data-science-team')
from custom.agents.my_agent import MyAgent
```

### "Git won't let me push to upstream"

Good! That's the safety feature working. Push to YOUR fork instead:
```bash
git push origin main  # Not upstream!
```

### "My private files are showing in git status"

Check the file name matches .gitignore patterns:
- Must be in `custom/private/`
- Or match: `*.key`, `*.secret`, `*_secret.py`, etc.

Run: `git status --ignored` to verify

## 📚 Next Steps

1. Read `CUSTOM_WORKFLOW.md` for detailed workflows
2. Check `custom/README.md` for directory structure
3. Look at core examples in `examples/` for inspiration
4. Start building!

## 💡 Pro Tips

1. **Start simple** - Copy template, make small changes, test
2. **Use experiments/** - Try things out before committing
3. **Document as you go** - Add docstrings and comments
4. **Test with core package** - Make sure integration works
5. **Version control often** - Commit working states

Happy building! 🚀
