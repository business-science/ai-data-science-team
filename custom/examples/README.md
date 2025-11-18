# Usage Examples

This directory contains examples showing different ways to use the custom agents.

## 📁 Files Overview

| File | Type | Purpose |
|------|------|---------|
| `streamlit_quality_checker.py` | Web App | Browser-based UI for data quality checking |
| `cli_quality_check.py` | CLI | Command-line tool for quick analysis |
| `api_server.py` | REST API | HTTP API for JavaScript/web integration |
| `web_client.html` | Web Page | JavaScript example using the API |

---

## 🚀 Quick Start

### 1. Python Script (Simplest)

**Create: `my_analysis.py`**
```python
from langchain_openai import ChatOpenAI
from custom.agents import DataQualityAgent
import pandas as pd
import os

# Setup
os.environ['OPENAI_API_KEY'] = 'sk-...'
llm = ChatOpenAI(model="gpt-4")

# Load data (Excel or CSV)
df = pd.read_excel("my_data.xlsx")
# or: df = pd.read_csv("my_data.csv")

# Analyze
agent = DataQualityAgent(model=llm)
report = agent.generate_report(df, "my_data")

print(report)

# Save to file
with open("quality_report.txt", "w") as f:
    f.write(report)
```

**Run:**
```bash
python my_analysis.py
```

---

### 2. Jupyter Notebook (Interactive)

**Create new notebook: `analysis.ipynb`**

```python
# Cell 1: Setup
import pandas as pd
from langchain_openai import ChatOpenAI
from custom.agents import DataQualityAgent, OutlierDetectionAgent
import os

os.environ['OPENAI_API_KEY'] = 'sk-...'
llm = ChatOpenAI(model="gpt-4")

# Cell 2: Load data
df = pd.read_csv("data.csv")
df.head()

# Cell 3: Quality check
dq = DataQualityAgent(model=llm)
result = dq.quick_check(df)
print(result)

# Cell 4: Outlier detection
outlier_agent = OutlierDetectionAgent(model=llm)
outliers = outlier_agent.quick_detect(
    data=df,
    columns=['revenue', 'quantity'],
    method='iqr'
)
print(outliers['treatment_recommendations'])
```

**Run in:**
- Jupyter Notebook
- VS Code
- Google Colab
- JupyterLab

---

### 3. Streamlit Web App (Share with Team)

**Run the included example:**

```bash
# Install Streamlit
pip install streamlit

# Set API key
export OPENAI_API_KEY='sk-...'

# Run app
streamlit run custom/examples/streamlit_quality_checker.py
```

**Access:** Opens in browser at `http://localhost:8501`

**Features:**
- 📤 Drag & drop Excel/CSV files
- ✅ Quality analysis reports
- 🎯 Outlier detection
- 📥 Download results
- 🎨 Beautiful UI

**Perfect for:**
- Non-technical users
- Quick data checks
- Sharing with team members
- Demo presentations

---

### 4. Command-Line Tool (Fast & Scriptable)

**Run the CLI:**

```bash
# Set API key
export OPENAI_API_KEY='sk-...'

# Make executable
chmod +x custom/examples/cli_quality_check.py

# Basic usage
python custom/examples/cli_quality_check.py data.csv

# With outlier detection
python custom/examples/cli_quality_check.py sales.xlsx --outliers

# Save to file
python custom/examples/cli_quality_check.py data.csv --outliers -o report.txt

# Custom method
python custom/examples/cli_quality_check.py data.csv --outliers --method zscore
```

**Use in shell scripts:**
```bash
#!/bin/bash
for file in data/*.csv; do
    python custom/examples/cli_quality_check.py "$file" -o "reports/$(basename $file .csv)_report.txt"
done
```

---

### 5. REST API + JavaScript (Web Integration)

**Step 1: Start the API server**

```bash
# Install Flask
pip install flask flask-cors

# Set API key
export OPENAI_API_KEY='sk-...'

# Run server
python custom/examples/api_server.py
```

Server runs at: `http://localhost:5000`

**Step 2: Use from JavaScript**

**Option A: Open the HTML client**
```bash
# Open in browser
open custom/examples/web_client.html
# or
firefox custom/examples/web_client.html
```

**Option B: Use from your own JavaScript**
```javascript
// Upload file and check quality
async function checkQuality(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:5000/api/quality-check', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    console.log(data.report);
}

// Detect outliers
async function detectOutliers(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('method', 'iqr');

    const response = await fetch('http://localhost:5000/api/detect-outliers', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    console.log(data.outlier_info);
}

// From file input
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    checkQuality(file);
});
```

**Option C: Use with fetch/axios**
```javascript
// Using axios
import axios from 'axios';

const formData = new FormData();
formData.append('file', fileObject);

axios.post('http://localhost:5000/api/quality-check', formData)
    .then(response => {
        console.log(response.data.report);
    });
```

**Option D: Use from React**
```jsx
import React, { useState } from 'react';

function DataQualityChecker() {
    const [report, setReport] = useState('');

    const handleUpload = async (event) => {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:5000/api/quality-check', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        setReport(data.report);
    };

    return (
        <div>
            <input type="file" onChange={handleUpload} accept=".csv,.xlsx" />
            <pre>{report}</pre>
        </div>
    );
}
```

---

## 📊 What Can You Analyze?

### Excel Files
```python
df = pd.read_excel("data.xlsx")
agent.quick_check(df)
```

### CSV Files
```python
df = pd.read_csv("data.csv")
agent.quick_check(df)
```

### Database Queries
```python
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)
agent.quick_check(df)
```

### API Data
```python
import requests
data = requests.get('https://api.example.com/data').json()
df = pd.DataFrame(data)
agent.quick_check(df)
```

---

## 🎯 Common Use Cases

### 1. Daily Data Quality Checks
```bash
# Cron job (runs every day at 9 AM)
0 9 * * * cd /path/to/project && python custom/examples/cli_quality_check.py /data/daily_export.csv --outliers -o /reports/$(date +\%Y\%m\%d)_report.txt
```

### 2. ML Pipeline Integration
```python
from custom.agents import DataQualityAgent, OutlierDetectionAgent
from sklearn.model_selection import train_test_split

# 1. Check data quality
dq_agent = DataQualityAgent(model=llm)
quality = dq_agent.quick_check(df)

# 2. Handle outliers
outlier_agent = OutlierDetectionAgent(model=llm)
outliers = outlier_agent.get_consensus_outliers(df, columns, methods=['iqr', 'zscore'])
df_clean = df.drop(outliers)

# 3. Continue with ML
X_train, X_test, y_train, y_test = train_test_split(...)
```

### 3. Data Validation Service
```python
# Flask app
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['data']
    df = pd.read_csv(file)

    # Validate
    agent = DataQualityAgent(model=llm)
    report = agent.quick_check(df)

    # Return validation results
    return jsonify({'valid': 'ERROR' not in report, 'report': report})
```

### 4. Scheduled Reports
```python
import schedule
import time

def weekly_report():
    df = pd.read_csv("weekly_data.csv")
    agent = DataQualityAgent(model=llm)
    report = agent.generate_report(df, "weekly_data")

    # Email report
    send_email(to="team@company.com", body=report)

schedule.every().monday.at("09:00").do(weekly_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 🔧 Configuration

### API Keys

**Option 1: Environment Variable**
```bash
export OPENAI_API_KEY='sk-...'
```

**Option 2: .env File**
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-..." > .env

# In Python
from dotenv import load_dotenv
load_dotenv()
```

**Option 3: In Code (not recommended)**
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
```

### Model Selection

```python
# GPT-4 (most accurate, slower, more expensive)
llm = ChatOpenAI(model="gpt-4")

# GPT-3.5-turbo (faster, cheaper)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Other providers
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")
```

---

## 🐛 Troubleshooting

### Import Errors
```python
# Add project to path
import sys
sys.path.insert(0, '/path/to/ai-data-science-team')
from custom.agents import DataQualityAgent
```

### API Server Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
python api_server.py --port 8080
```

### CORS Errors (JavaScript)
```python
# Already handled in api_server.py with flask-cors
# If using different server, add CORS headers:
from flask_cors import CORS
CORS(app)
```

### File Upload Size Limits
```python
# Flask config
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
```

---

## 📚 Next Steps

1. **Read the full documentation:** `custom/docs/CUSTOM_EXTENSIONS_GUIDE.md`
2. **Try other agents:**
   - `FeatureImportanceAgent`
   - `ModelComparisonAgent`
   - `OutlierDetectionAgent`
   - `TimeSeriesAgent`
3. **Build your own agent:** See `custom/agents/TEMPLATE_agent.py`

---

## 💡 Tips

- **Start with quick methods** for known tasks, use full agent for exploration
- **Batch process files** with the CLI tool
- **Cache results** - agents can be slow/expensive
- **Use appropriate models** - GPT-3.5 for simple tasks, GPT-4 for complex
- **Monitor costs** - check OpenAI usage dashboard
