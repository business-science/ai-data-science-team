#!/usr/bin/env python3
"""
CLI Data Quality Checker

Usage:
    python cli_quality_check.py data.csv
    python cli_quality_check.py data.xlsx --method iqr --outliers
"""

import argparse
import pandas as pd
import sys
import os
from langchain_openai import ChatOpenAI

# Add custom modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from custom.agents import DataQualityAgent, OutlierDetectionAgent


def main():
    parser = argparse.ArgumentParser(description='Check data quality using AI agents')
    parser.add_argument('file', help='CSV or Excel file to analyze')
    parser.add_argument('--outliers', action='store_true', help='Run outlier detection')
    parser.add_argument('--method', default='iqr', choices=['iqr', 'zscore', 'isolation_forest'],
                        help='Outlier detection method')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--model', default='gpt-4', help='OpenAI model to use')

    args = parser.parse_args()

    # Check API key
    if 'OPENAI_API_KEY' not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Load data
    print(f"Loading {args.file}...")
    try:
        if args.file.endswith('.csv'):
            df = pd.read_csv(args.file)
        elif args.file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.file)
        else:
            print(f"ERROR: Unsupported file format. Use .csv, .xlsx, or .xls")
            sys.exit(1)

        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)

    # Initialize LLM
    print(f"Initializing {args.model}...")
    llm = ChatOpenAI(model=args.model)

    # Quality check
    print("\n" + "="*80)
    print("QUALITY ANALYSIS")
    print("="*80 + "\n")

    agent = DataQualityAgent(model=llm, verbose=False)
    report = agent.generate_report(df, report_name=os.path.basename(args.file))
    print(report)

    # Outlier detection
    if args.outliers:
        print("\n" + "="*80)
        print("OUTLIER DETECTION")
        print("="*80 + "\n")

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            print("⚠ No numeric columns found for outlier detection")
        else:
            print(f"Checking columns: {', '.join(numeric_cols)}")
            print(f"Method: {args.method}\n")

            outlier_agent = OutlierDetectionAgent(model=llm, verbose=False)
            result = outlier_agent.quick_detect(
                data=df,
                columns=numeric_cols,
                method=args.method
            )

            print("OUTLIERS DETECTED:")
            print("-" * 80)
            for col, info in result['outlier_info'].items():
                if isinstance(info, dict) and 'outlier_count' in info:
                    print(f"{col}: {info['outlier_count']} outliers ({info['outlier_pct']:.2f}%)")

            print("\n" + result['treatment_recommendations'])

    # Save output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
            if args.outliers:
                f.write("\n\n" + result['treatment_recommendations'])
        print(f"\n✓ Report saved to {args.output}")


if __name__ == '__main__':
    main()
