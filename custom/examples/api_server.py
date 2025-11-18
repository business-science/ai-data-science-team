"""
REST API Server for Custom Agents

Allows JavaScript/web apps to use the agents via HTTP requests.

Run: python api_server.py
Then: curl -X POST -F "file=@data.csv" http://localhost:5000/api/quality-check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import io
import os
from langchain_openai import ChatOpenAI
import sys

# Add custom modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from custom.agents import DataQualityAgent, OutlierDetectionAgent, FeatureImportanceAgent

app = Flask(__name__)
CORS(app)  # Allow JavaScript from browsers

# Initialize agents (in production, use connection pooling)
llm = ChatOpenAI(model="gpt-4", api_key=os.environ.get('OPENAI_API_KEY'))
dq_agent = DataQualityAgent(model=llm, verbose=False)
outlier_agent = OutlierDetectionAgent(model=llm, verbose=False)


@app.route('/api/quality-check', methods=['POST'])
def quality_check():
    """
    Check data quality

    Request:
        POST /api/quality-check
        Form-data: file (CSV or Excel)

    Response:
        JSON with quality report
    """
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file.read()))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file.read()))
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Run quality check
        result = dq_agent.quick_check(df)

        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'report': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-outliers', methods=['POST'])
def detect_outliers():
    """
    Detect outliers in data

    Request:
        POST /api/detect-outliers
        Form-data:
            - file (CSV or Excel)
            - method (optional): iqr, zscore, isolation_forest

    Response:
        JSON with outlier information
    """
    try:
        # Get file and parameters
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        method = request.form.get('method', 'iqr')

        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file.read()))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file.read()))
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            return jsonify({'error': 'No numeric columns found'}), 400

        # Detect outliers
        result = outlier_agent.quick_detect(
            data=df,
            columns=numeric_cols,
            method=method
        )

        return jsonify({
            'status': 'success',
            'method': method,
            'outlier_info': result['outlier_info'],
            'recommendations': result['treatment_recommendations']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Custom AI Agents API'})


if __name__ == '__main__':
    # Check API key
    if 'OPENAI_API_KEY' not in os.environ:
        print("WARNING: OPENAI_API_KEY not set")

    print("Starting API server...")
    print("Endpoints:")
    print("  - POST /api/quality-check")
    print("  - POST /api/detect-outliers")
    print("  - GET  /api/health")
    print("\nAccess at: http://localhost:5000")

    app.run(debug=True, port=5000)
