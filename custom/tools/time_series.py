"""
Time Series Analysis Tools

Tools for analyzing time series data including trend detection, seasonality,
decomposition, and basic forecasting.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from langchain.tools import tool
import json


@tool
def detect_seasonality(
    data: dict,
    value_column: str,
    date_column: str,
    max_lag: int = 365
) -> str:
    """
    Detect seasonality in time series data using autocorrelation.

    Args:
        data: Dictionary representation of DataFrame with time series
        value_column: Column containing the values
        date_column: Column containing dates
        max_lag: Maximum lag to check for seasonality

    Returns:
        JSON string with seasonality detection results
    """
    from scipy import stats

    df = pd.DataFrame(data)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    series = df[value_column].values

    # Calculate autocorrelation
    autocorr = []
    for lag in range(1, min(max_lag, len(series) // 2)):
        if lag < len(series):
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            autocorr.append({'lag': lag, 'autocorrelation': float(corr)})

    # Find significant peaks in autocorrelation
    autocorr_df = pd.DataFrame(autocorr)
    threshold = 0.3  # Significant correlation threshold

    significant_lags = autocorr_df[autocorr_df['autocorrelation'] > threshold]

    # Detect common seasonality patterns
    seasonality_detected = []

    for _, row in significant_lags.iterrows():
        lag = row['lag']
        corr = row['autocorrelation']

        # Weekly (7 days)
        if 6 <= lag <= 8:
            seasonality_detected.append({
                'type': 'weekly',
                'period': lag,
                'strength': corr
            })
        # Monthly (28-31 days)
        elif 28 <= lag <= 31:
            seasonality_detected.append({
                'type': 'monthly',
                'period': lag,
                'strength': corr
            })
        # Quarterly (~90 days)
        elif 88 <= lag <= 93:
            seasonality_detected.append({
                'type': 'quarterly',
                'period': lag,
                'strength': corr
            })
        # Yearly (365 days)
        elif 360 <= lag <= 370:
            seasonality_detected.append({
                'type': 'yearly',
                'period': lag,
                'strength': corr
            })

    result = {
        'seasonality_found': len(seasonality_detected) > 0,
        'patterns': seasonality_detected,
        'top_autocorrelations': autocorr_df.nlargest(10, 'autocorrelation').to_dict(orient='records')
    }

    return json.dumps(result, indent=2)


@tool
def decompose_time_series(
    data: dict,
    value_column: str,
    date_column: str,
    model: str = "additive",
    period: Optional[int] = None
) -> str:
    """
    Decompose time series into trend, seasonal, and residual components.

    Args:
        data: Dictionary representation of DataFrame
        value_column: Column with values
        date_column: Column with dates
        model: 'additive' or 'multiplicative'
        period: Seasonal period (auto-detected if None)

    Returns:
        JSON string with decomposition results
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    df = pd.DataFrame(data)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).set_index(date_column)

    series = df[value_column]

    # Auto-detect period if not provided
    if period is None:
        # Try to infer frequency
        freq = pd.infer_freq(series.index)
        if freq:
            if 'D' in freq:
                period = 7  # Weekly for daily data
            elif 'H' in freq:
                period = 24  # Daily for hourly data
            elif 'M' in freq:
                period = 12  # Yearly for monthly data
        else:
            period = min(len(series) // 2, 365)  # Default

    # Perform decomposition
    try:
        decomposition = seasonal_decompose(
            series,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )

        result = {
            'model': model,
            'period': period,
            'trend': decomposition.trend.dropna().to_dict(),
            'seasonal': decomposition.seasonal.dropna().to_dict(),
            'residual': decomposition.resid.dropna().to_dict(),
            'trend_strength': float(1 - (decomposition.resid.var() / (decomposition.trend + decomposition.resid).var())),
            'seasonal_strength': float(1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var()))
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "message": "Decomposition failed. Try adjusting the period parameter."})


@tool
def check_stationarity(
    data: dict,
    value_column: str,
    date_column: str
) -> str:
    """
    Check if time series is stationary using Augmented Dickey-Fuller test.

    Args:
        data: Dictionary representation of DataFrame
        value_column: Column with values
        date_column: Column with dates

    Returns:
        JSON string with stationarity test results
    """
    from statsmodels.tsa.stattools import adfuller

    df = pd.DataFrame(data)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    series = df[value_column].dropna()

    # Perform ADF test
    adf_result = adfuller(series, autolag='AIC')

    result = {
        'test': 'Augmented Dickey-Fuller',
        'adf_statistic': float(adf_result[0]),
        'p_value': float(adf_result[1]),
        'critical_values': {k: float(v) for k, v in adf_result[4].items()},
        'is_stationary': adf_result[1] < 0.05,
        'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
    }

    if not result['is_stationary']:
        result['recommendation'] = "Series is non-stationary. Consider differencing or detrending."

    return json.dumps(result, indent=2)


@tool
def forecast_baseline(
    data: dict,
    value_column: str,
    date_column: str,
    forecast_periods: int = 30,
    method: str = "naive"
) -> str:
    """
    Create baseline forecasts using simple methods.

    Args:
        data: Dictionary representation of DataFrame
        value_column: Column with values
        date_column: Column with dates
        forecast_periods: Number of periods to forecast
        method: 'naive', 'mean', 'seasonal_naive', 'drift'

    Returns:
        JSON string with forecast results
    """
    df = pd.DataFrame(data)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    series = df[value_column].values
    dates = df[date_column].values

    # Generate future dates
    last_date = pd.to_datetime(dates[-1])
    freq = pd.infer_freq(pd.to_datetime(dates))
    if freq is None:
        freq = 'D'  # Default to daily

    future_dates = pd.date_range(
        start=last_date,
        periods=forecast_periods + 1,
        freq=freq
    )[1:]  # Exclude the start date

    # Generate forecasts based on method
    if method == "naive":
        # Last value forecast
        forecast = [series[-1]] * forecast_periods

    elif method == "mean":
        # Historical mean
        forecast = [np.mean(series)] * forecast_periods

    elif method == "seasonal_naive":
        # Last season's values
        season_length = min(7, len(series))  # Assume weekly seasonality
        forecast = []
        for i in range(forecast_periods):
            forecast.append(series[-(season_length - (i % season_length))])

    elif method == "drift":
        # Linear trend from first to last value
        drift = (series[-1] - series[0]) / (len(series) - 1)
        forecast = [series[-1] + drift * (i + 1) for i in range(forecast_periods)]

    else:
        return json.dumps({"error": f"Unknown method: {method}"})

    result = {
        'method': method,
        'forecast_periods': forecast_periods,
        'historical_last_value': float(series[-1]),
        'forecast_values': [float(v) for v in forecast],
        'forecast_dates': [str(d) for d in future_dates],
        'historical_mean': float(np.mean(series)),
        'historical_std': float(np.std(series))
    }

    return json.dumps(result, indent=2)


@tool
def visualize_time_series(
    data: dict,
    value_column: str,
    date_column: str,
    forecast_data: Optional[str] = None,
    title: str = "Time Series Analysis"
) -> str:
    """
    Create visualizations for time series data.

    Args:
        data: Dictionary representation of DataFrame
        value_column: Column with values
        date_column: Column with dates
        forecast_data: Optional JSON string with forecast data
        title: Plot title

    Returns:
        JSON string of Plotly figure
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    df = pd.DataFrame(data)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df[value_column],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))

    # Add forecast if provided
    if forecast_data:
        forecast = json.loads(forecast_data)
        if 'forecast_dates' in forecast and 'forecast_values' in forecast:
            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(d) for d in forecast['forecast_dates']],
                y=forecast['forecast_values'],
                mode='lines',
                name=f"Forecast ({forecast['method']})",
                line=dict(color='red', width=2, dash='dash')
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=value_column,
        height=500,
        hovermode='x unified'
    )

    return pio.to_json(fig)


@tool
def calculate_time_series_metrics(
    data: dict,
    value_column: str,
    date_column: str
) -> str:
    """
    Calculate various time series metrics and statistics.

    Args:
        data: Dictionary representation of DataFrame
        value_column: Column with values
        date_column: Column with dates

    Returns:
        JSON string with metrics
    """
    df = pd.DataFrame(data)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    series = df[value_column].dropna()

    # Calculate metrics
    metrics = {
        'count': len(series),
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'range': float(series.max() - series.min()),
        'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else None,
        'start_date': str(df[date_column].min()),
        'end_date': str(df[date_column].max()),
        'duration_days': (df[date_column].max() - df[date_column].min()).days,
    }

    # Calculate growth metrics
    if len(series) > 1:
        first_value = series.iloc[0]
        last_value = series.iloc[-1]

        metrics['first_value'] = float(first_value)
        metrics['last_value'] = float(last_value)
        metrics['absolute_change'] = float(last_value - first_value)
        metrics['percent_change'] = float((last_value - first_value) / first_value * 100) if first_value != 0 else None

        # Average daily change
        total_days = (df[date_column].max() - df[date_column].min()).days
        if total_days > 0:
            metrics['avg_daily_change'] = float((last_value - first_value) / total_days)

    # Volatility (rolling std)
    if len(series) > 7:
        rolling_std = series.rolling(window=7).std()
        metrics['volatility_7day_avg'] = float(rolling_std.mean())

    # Trend direction
    if len(series) > 2:
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series.values, 1)
        metrics['trend_slope'] = float(slope)
        metrics['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'

    return json.dumps(metrics, indent=2)
