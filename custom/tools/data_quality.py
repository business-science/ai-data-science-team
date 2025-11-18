"""
Data Quality Validation Tools

Tools for comprehensive data quality assessment, validation, and reporting.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from langchain.tools import tool


@tool
def check_schema_compliance(data: dict, expected_schema: dict) -> str:
    """
    Validate DataFrame schema against expected schema definition.

    Args:
        data: Dictionary representation of DataFrame
        expected_schema: Dict mapping column names to expected dtypes

    Returns:
        String report of schema compliance issues

    Example:
        >>> schema = {"age": "int64", "name": "object", "salary": "float64"}
        >>> result = check_schema_compliance(df.to_dict(), schema)
    """
    df = pd.DataFrame(data)
    issues = []

    # Check missing columns
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {', '.join(missing_cols)}")

    # Check extra columns
    extra_cols = set(df.columns) - set(expected_schema.keys())
    if extra_cols:
        issues.append(f"Unexpected columns: {', '.join(extra_cols)}")

    # Check data types
    for col, expected_dtype in expected_schema.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if actual_dtype != expected_dtype:
                issues.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")

    if not issues:
        return "✓ Schema validation passed - all columns present with correct types"

    return "Schema validation issues found:\n" + "\n".join(f"  - {issue}" for issue in issues)


@tool
def detect_data_anomalies(data: dict, numeric_std_threshold: float = 3.0) -> str:
    """
    Detect various data anomalies including extreme values, invalid formats, and inconsistencies.

    Args:
        data: Dictionary representation of DataFrame
        numeric_std_threshold: Number of standard deviations for outlier detection

    Returns:
        Detailed report of detected anomalies
    """
    df = pd.DataFrame(data)
    anomalies = []

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        anomalies.append(f"Duplicate rows: {dup_count} ({dup_count/len(df)*100:.2f}%)")

    # Check each column
    for col in df.columns:
        col_anomalies = []

        # Missing values
        missing = df[col].isna().sum()
        if missing > 0:
            col_anomalies.append(f"{missing} missing ({missing/len(df)*100:.2f}%)")

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                col_anomalies.append(f"{inf_count} infinite values")

            # Check for extreme outliers
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                mean = clean_data.mean()
                std = clean_data.std()
                if std > 0:
                    outliers = np.abs((clean_data - mean) / std) > numeric_std_threshold
                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        col_anomalies.append(f"{outlier_count} extreme outliers (>{numeric_std_threshold}σ)")

            # Check for negative values where inappropriate
            if clean_data.min() < 0 and col.lower() in ['age', 'price', 'quantity', 'count']:
                col_anomalies.append(f"Contains negative values (min: {clean_data.min():.2f})")

        # String columns
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Check for whitespace issues
            if df[col].dtype == object:
                str_series = df[col].astype(str)
                leading_space = (str_series.str.startswith(' ')).sum()
                trailing_space = (str_series.str.endswith(' ')).sum()
                if leading_space > 0:
                    col_anomalies.append(f"{leading_space} values with leading whitespace")
                if trailing_space > 0:
                    col_anomalies.append(f"{trailing_space} values with trailing whitespace")

        # Date columns
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Check for future dates where inappropriate
            if col.lower() in ['birth_date', 'start_date', 'created_at']:
                future_dates = (df[col] > pd.Timestamp.now()).sum()
                if future_dates > 0:
                    col_anomalies.append(f"{future_dates} future dates")

        if col_anomalies:
            anomalies.append(f"{col}: {', '.join(col_anomalies)}")

    if not anomalies:
        return "✓ No data anomalies detected"

    return "Data anomalies detected:\n" + "\n".join(f"  - {anomaly}" for anomaly in anomalies)


@tool
def validate_business_rules(data: dict, rules: dict) -> str:
    """
    Validate custom business rules on the dataset.

    Args:
        data: Dictionary representation of DataFrame
        rules: Dict of rule_name -> rule_expression (lambda or string)

    Returns:
        Report of business rule violations

    Example:
        >>> rules = {
        ...     "age_range": lambda df: (df['age'] >= 0) & (df['age'] <= 120),
        ...     "salary_positive": lambda df: df['salary'] > 0
        ... }
        >>> result = validate_business_rules(df.to_dict(), rules)
    """
    df = pd.DataFrame(data)
    violations = []

    for rule_name, rule_func in rules.items():
        try:
            if callable(rule_func):
                valid = rule_func(df)
                invalid_count = (~valid).sum()
                if invalid_count > 0:
                    violations.append(
                        f"{rule_name}: {invalid_count} violations ({invalid_count/len(df)*100:.2f}%)"
                    )
        except Exception as e:
            violations.append(f"{rule_name}: Error evaluating rule - {str(e)}")

    if not violations:
        return "✓ All business rules passed"

    return "Business rule violations:\n" + "\n".join(f"  - {v}" for v in violations)


@tool
def calculate_data_quality_score(data: dict) -> str:
    """
    Calculate overall data quality score based on multiple metrics.

    Args:
        data: Dictionary representation of DataFrame

    Returns:
        Comprehensive quality score report with breakdown
    """
    df = pd.DataFrame(data)

    scores = {}

    # Completeness (100 - % missing)
    missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    completeness = max(0, 100 - missing_pct)
    scores['Completeness'] = completeness

    # Uniqueness (% non-duplicate rows)
    dup_pct = (df.duplicated().sum() / len(df)) * 100
    uniqueness = max(0, 100 - dup_pct)
    scores['Uniqueness'] = uniqueness

    # Validity (check for invalid values)
    validity_issues = 0
    total_checks = 0

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            total_checks += 2
            # Check for inf
            if np.isinf(df[col]).sum() == 0:
                validity_issues += 1
            # Check for extreme outliers
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                std = clean_data.std()
                if std > 0:
                    outlier_pct = (np.abs((clean_data - clean_data.mean()) / std) > 4).sum() / len(clean_data)
                    if outlier_pct < 0.05:  # Less than 5% outliers
                        validity_issues += 1

    validity = (validity_issues / total_checks * 100) if total_checks > 0 else 100
    scores['Validity'] = validity

    # Consistency (check data types consistency)
    consistency_score = 100  # Start optimistic
    for col in df.columns:
        if df[col].dtype == object:
            # Check if numeric values are stored as strings
            try:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    numeric_count = pd.to_numeric(non_null, errors='coerce').notna().sum()
                    if numeric_count / len(non_null) > 0.9:  # >90% numeric but stored as object
                        consistency_score -= 10
            except:
                pass

    scores['Consistency'] = max(0, consistency_score)

    # Overall score
    overall_score = np.mean(list(scores.values()))

    # Generate report
    report = "DATA QUALITY SCORECARD\n"
    report += "=" * 50 + "\n"
    report += f"Overall Quality Score: {overall_score:.1f}/100\n\n"
    report += "Breakdown:\n"
    for metric, score in scores.items():
        stars = "★" * int(score / 20) + "☆" * (5 - int(score / 20))
        report += f"  {metric:15s}: {score:5.1f}/100 {stars}\n"

    report += "\nInterpretation:\n"
    if overall_score >= 90:
        report += "  ✓ Excellent - Data is high quality and ready for analysis"
    elif overall_score >= 75:
        report += "  ⚠ Good - Minor issues present, review recommended"
    elif overall_score >= 60:
        report += "  ⚠ Fair - Significant issues detected, cleaning needed"
    else:
        report += "  ✗ Poor - Major quality issues, extensive cleaning required"

    return report


@tool
def generate_data_quality_report(data: dict, report_name: str = "data_quality_report") -> str:
    """
    Generate a comprehensive data quality report with all checks.

    Args:
        data: Dictionary representation of DataFrame
        report_name: Name for the report

    Returns:
        Full quality assessment report
    """
    df = pd.DataFrame(data)

    report = f"DATA QUALITY REPORT: {report_name}\n"
    report += "=" * 80 + "\n\n"

    # Basic statistics
    report += f"Dataset Overview:\n"
    report += f"  Rows: {len(df):,}\n"
    report += f"  Columns: {len(df.columns)}\n"
    report += f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"

    # Column-by-column analysis
    report += "Column Analysis:\n"
    report += "-" * 80 + "\n"

    for col in df.columns:
        report += f"\n{col} ({df[col].dtype}):\n"

        # Basic stats
        total = len(df)
        missing = df[col].isna().sum()
        unique = df[col].nunique()

        report += f"  Missing: {missing:,} ({missing/total*100:.1f}%)\n"
        report += f"  Unique: {unique:,} ({unique/total*100:.1f}%)\n"

        # Type-specific stats
        if pd.api.types.is_numeric_dtype(df[col]):
            clean = df[col].dropna()
            if len(clean) > 0:
                report += f"  Range: [{clean.min():.2f}, {clean.max():.2f}]\n"
                report += f"  Mean: {clean.mean():.2f}, Std: {clean.std():.2f}\n"

                # Outliers
                if clean.std() > 0:
                    outliers = (np.abs((clean - clean.mean()) / clean.std()) > 3).sum()
                    if outliers > 0:
                        report += f"  ⚠ Outliers (>3σ): {outliers}\n"

        elif pd.api.types.is_object_dtype(df[col]):
            # Top values
            top_values = df[col].value_counts().head(3)
            report += f"  Top values: {', '.join([f'{v}({c})' for v, c in top_values.items()])}\n"

    report += "\n" + "=" * 80 + "\n"

    # Quality score
    score_result = calculate_data_quality_score(data)
    report += "\n" + score_result

    return report
