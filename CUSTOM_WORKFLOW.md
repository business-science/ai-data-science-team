# Custom Workflow Guide

This document explains how to safely work on your fork while maintaining the ability to pull updates from the upstream project.

## 🛡️ Safety Guardrails (Already Configured)

### 1. Git Protection

Your git is configured to **prevent accidental pushes to upstream**:

```bash
git remote -v
# Shows:
# upstream ... (fetch)
# upstream no_push (push)  ← Cannot push!
```

**What this means:**
- ✅ You can PULL from upstream (get their updates)
- ✅ You can PUSH to origin (your fork)
- ❌ You CANNOT push to upstream (safety!)

### 2. Directory Separation

```
ai-data-science-team/
├── ai_data_science_team/    # Core package (upstream code)
├── apps/                     # Core apps (upstream code)
├── examples/                 # Core examples (upstream code)
├── custom/                   # YOUR CODE (isolated)
│   ├── agents/              # Your custom agents
│   ├── tools/               # Your custom tools
│   ├── apps/                # Your custom apps
│   ├── examples/            # Your examples
│   ├── experiments/         # WIP code
│   ├── private/             # NEVER committed (gitignored)
│   └── docs/                # Your documentation
└── ... other core files
```

### 3. Gitignore Protection

The following are **automatically ignored** (won't be committed):
- `custom/private/` - Everything in here
- `*.key`, `*.secret` files in custom/
- `*_secret.py`, `*_private.py` files
- `.env*` files in custom/
- `credentials*.json` in custom/
- Large data files (CSV, Excel, Parquet, etc.)

## 📋 Common Workflows

### Starting a New Custom Feature

```bash
# 1. Create a feature branch
git checkout -b feature/my-awesome-agent

# 2. Work in the custom/ directory
# Create your files in custom/agents/, custom/tools/, etc.

# 3. Test your code
python custom/examples/test_my_agent.py

# 4. Commit to YOUR fork
git add custom/
git commit -m "Add my awesome agent"
git push origin feature/my-awesome-agent
```

**Key Points:**
- Work only in `custom/` directory
- All commits go to YOUR fork
- Impossible to accidentally push to upstream

### Pulling Updates from Upstream

```bash
# 1. Fetch latest changes from original project
git fetch upstream main

# 2. See what changed
git log HEAD..upstream/main --oneline

# 3. (Optional) View detailed changes
git diff HEAD..upstream/main

# 4. If you want to merge the updates
git checkout main
git merge upstream/main

# 5. Push to your fork
git push origin main
```

**Why this is safe:**
- Your `custom/` directory won't conflict
- You review changes before merging
- Updates only affect core package, not your custom code

### Working with Sensitive Code

For proprietary work, API keys, or client projects:

```bash
# Put everything in custom/private/
custom/private/
├── client_project/
│   └── proprietary_agent.py
├── api_keys.py
└── secret_sauce.py

# These files are AUTOMATICALLY ignored by git
# They will NEVER be committed
```

**Verify protection:**
```bash
git status
# Should NOT show custom/private/ files
```

### Experiment Safely

```bash
# Use custom/experiments/ for trying new ideas
custom/experiments/
├── crazy_idea_v1.py
├── test_new_approach.ipynb
└── benchmark_comparison.py

# These CAN be committed (unlike private/)
# But you can uncomment the experiments line in .gitignore if you want to keep them local
```

## 🔄 Branch Strategy

### Recommended Setup

```
main                    ← Tracks upstream + your stable custom work
├── feature/agent-x     ← Your custom agent development
├── feature/tool-y      ← Your custom tool development
└── experiment/idea-z   ← Experimental work
```

### Branch Naming Convention

```bash
# Features
git checkout -b feature/custom-nlp-agent
git checkout -b feature/custom-viz-tool

# Experiments
git checkout -b experiment/gpt4-integration
git checkout -b experiment/performance-tests

# Fixes to core (if contributing back)
git checkout -b fix/data-cleaning-bug

# Documentation
git checkout -b docs/custom-setup-guide
```

## 🚨 Emergency: "I Accidentally Modified Core Files"

If you modified files in `ai_data_science_team/`, `apps/`, or other core directories:

### Option 1: Stash and Move to Custom
```bash
# Save your changes
git stash

# Create the file in custom/ instead
# ... create custom/agents/my_version.py ...

# Restore original core file
git checkout HEAD -- ai_data_science_team/agents/some_agent.py

# Apply your stashed changes to new location if needed
git stash pop
# ... manually move relevant code to custom/ ...
```

### Option 2: Create a Custom Wrapper
Instead of modifying core files, wrap them:

```python
# custom/agents/enhanced_wrangling.py
from ai_data_science_team.agents import DataWranglingAgent

class EnhancedDataWranglingAgent(DataWranglingAgent):
    """My enhanced version with custom features"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Your custom initialization

    def custom_method(self):
        # Your new functionality
        pass
```

## 🎯 Best Practices

### DO ✅

1. **Always work in `custom/` directory**
2. **Use feature branches for new work**
3. **Commit and push regularly to YOUR fork**
4. **Pull from upstream periodically** (stay up to date)
5. **Document your custom features**
6. **Test before committing**
7. **Use `custom/private/` for sensitive work**

### DON'T ❌

1. **Don't modify core package files directly**
2. **Don't commit API keys or secrets**
3. **Don't push large data files**
4. **Don't try to push to upstream** (it's blocked anyway)
5. **Don't forget to test after merging upstream changes**

## 🧪 Testing Custom + Core Integration

```python
# custom/examples/test_integration.py
import sys
sys.path.insert(0, '/path/to/project')

# Import core functionality
from ai_data_science_team.agents import DataWranglingAgent

# Import your custom code
from custom.agents.my_agent import MyCustomAgent
from custom.tools.my_tools import my_helper_function

# Test integration
def test_custom_with_core():
    core_agent = DataWranglingAgent(...)
    custom_agent = MyCustomAgent(...)

    result = my_helper_function(core_agent, custom_agent)
    assert result is not None
```

## 📊 Tracking What's Custom vs Core

```bash
# See all custom work
git log --all --oneline -- custom/

# See what changed in core since last upstream merge
git log main..upstream/main -- ai_data_science_team/

# See your modifications to core (if any - ideally none!)
git diff upstream/main -- ai_data_science_team/
```

## 🔍 Quick Reference

| Task | Command |
|------|---------|
| Create custom feature | `git checkout -b feature/NAME` |
| Commit custom work | `git add custom/ && git commit -m "..."` |
| Push to your fork | `git push origin BRANCH_NAME` |
| Get upstream updates | `git fetch upstream main` |
| Merge upstream | `git merge upstream/main` |
| Check what's ignored | `git status --ignored` |
| View remotes | `git remote -v` |

## 🆘 Getting Help

If something goes wrong:

1. **Check git status**: `git status`
2. **Check what branch you're on**: `git branch`
3. **Check recent commits**: `git log --oneline -5`
4. **If in doubt, create a backup**: `git stash` or make a backup branch

## 📚 Additional Resources

- Git branching: https://learngitbranching.js.org/
- Forking workflow: https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow
- Custom directory: `custom/README.md`

---

**Remember:** Your `custom/` directory is your safe space. Everything else should remain as upstream code that you can update cleanly!
