"""
Streamlit Data Quality Checker App

A web interface for the Data Quality Agent that allows users to upload
Excel/CSV files and get instant quality reports.

Run with: streamlit run streamlit_quality_checker.py
"""

import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
import sys
import os

# Add custom modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from custom.agents import DataQualityAgent, OutlierDetectionAgent

st.set_page_config(page_title="Data Quality Checker", page_icon="✅", layout="wide")

st.title("🔍 Data Quality Checker")
st.markdown("Upload your Excel or CSV file to get instant quality analysis")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    model_name = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])

    st.markdown("---")
    st.markdown("### About")
    st.info("This app uses AI agents to analyze your data quality")

# File upload
uploaded_file = st.file_uploader(
    "Upload your data file",
    type=['csv', 'xlsx', 'xls'],
    help="Supported formats: CSV, Excel"
)

if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File loaded: {uploaded_file.name}")

        # Show data preview
        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(df.head(100))
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Analysis options
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            run_quality = st.checkbox("✅ Quality Analysis", value=True)
        with col2:
            run_outliers = st.checkbox("🎯 Outlier Detection", value=False)

        if st.button("🚀 Run Analysis", type="primary"):
            if not api_key:
                st.error("❌ Please enter your OpenAI API Key in the sidebar")
            else:
                os.environ['OPENAI_API_KEY'] = api_key

                # Initialize LLM
                with st.spinner("Initializing AI agents..."):
                    llm = ChatOpenAI(model=model_name)

                # Quality Analysis
                if run_quality:
                    st.markdown("---")
                    st.header("📋 Quality Analysis Report")

                    with st.spinner("Running quality checks..."):
                        agent = DataQualityAgent(model=llm, verbose=False)

                        # Quick check
                        quick_result = agent.quick_check(df)

                        # Display in tabs
                        tab1, tab2 = st.tabs(["Summary", "Full Report"])

                        with tab1:
                            st.text(quick_result)

                        with tab2:
                            full_report = agent.generate_report(
                                df,
                                report_name=uploaded_file.name
                            )
                            st.text(full_report)

                            # Download button
                            st.download_button(
                                label="📥 Download Report",
                                data=full_report,
                                file_name="quality_report.txt",
                                mime="text/plain"
                            )

                # Outlier Detection
                if run_outliers:
                    st.markdown("---")
                    st.header("🎯 Outlier Detection")

                    # Select numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                    if not numeric_cols:
                        st.warning("⚠️ No numeric columns found for outlier detection")
                    else:
                        selected_cols = st.multiselect(
                            "Select columns to check",
                            numeric_cols,
                            default=numeric_cols[:3]  # Default to first 3
                        )

                        method = st.selectbox(
                            "Detection method",
                            ['iqr', 'zscore', 'isolation_forest'],
                            help="IQR: Non-parametric, good for skewed data\n"
                                 "Z-score: Parametric, assumes normal distribution\n"
                                 "Isolation Forest: ML-based, multivariate"
                        )

                        if selected_cols:
                            with st.spinner(f"Detecting outliers using {method}..."):
                                outlier_agent = OutlierDetectionAgent(model=llm, verbose=False)

                                result = outlier_agent.quick_detect(
                                    data=df,
                                    columns=selected_cols,
                                    method=method
                                )

                                # Display results
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.subheader("Outlier Summary")
                                    st.json(result['outlier_info'])

                                with col2:
                                    st.subheader("Treatment Recommendations")
                                    st.text(result['treatment_recommendations'])

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    # Instructions when no file uploaded
    st.info("👆 Upload a CSV or Excel file to get started")

    st.markdown("### Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Quality Analysis:**
        - Schema validation
        - Missing value detection
        - Duplicate detection
        - Data type checking
        - Quality scoring
        """)

    with col2:
        st.markdown("""
        **Outlier Detection:**
        - Multiple detection methods
        - Automatic recommendations
        - Treatment strategies
        - Interactive results
        """)

# Footer
st.markdown("---")
st.caption("Powered by Custom AI Data Science Agents")
