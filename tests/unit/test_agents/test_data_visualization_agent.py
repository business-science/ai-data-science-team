"""
Unit tests for DataVisualizationAgent.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

try:
    from ai_data_science_team.agents.data_visualization_agent import (
        DataVisualizationAgent,
        make_data_visualization_agent,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestDataVisualizationAgent:
    """Tests for DataVisualizationAgent class."""

    def test_agent_initialization(self, mock_llm):
        """Test that agent initializes correctly."""
        agent = DataVisualizationAgent(model=mock_llm)
        assert agent is not None

    def test_agent_has_required_methods(self, mock_llm):
        """Test that agent has required interface methods."""
        agent = DataVisualizationAgent(model=mock_llm)

        assert hasattr(agent, "invoke_agent")
        assert hasattr(agent, "get_plotly_graph")
        assert hasattr(agent, "get_data_visualization_function")

    def test_agent_params_structure(self, mock_llm):
        """Test that agent params have expected structure."""
        agent = DataVisualizationAgent(
            model=mock_llm,
            human_in_the_loop=True,
            log=True,
        )
        assert agent._params["human_in_the_loop"] is True
        assert agent._params["log"] is True


class TestPlotlyVisualizationLogic:
    """Tests for visualization logic using Plotly."""

    def test_create_scatter_plot(self, sample_df):
        """Test scatter plot creation."""
        import plotly.express as px

        fig = px.scatter(
            sample_df,
            x="age",
            y="salary",
            color="department",
        )

        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0

    def test_create_bar_chart(self, sample_df):
        """Test bar chart creation."""
        import plotly.express as px

        # Group data for bar chart
        grouped = sample_df.groupby("department")["salary"].mean().reset_index()

        fig = px.bar(grouped, x="department", y="salary")

        assert fig is not None
        assert fig.data[0].type == "bar"

    def test_create_line_chart(self, sample_df_timeseries):
        """Test line chart creation."""
        import plotly.express as px

        fig = px.line(
            sample_df_timeseries,
            x="date",
            y="value",
        )

        assert fig is not None
        assert fig.data[0].type == "scatter"
        assert fig.data[0].mode == "lines"

    def test_create_histogram(self, sample_df):
        """Test histogram creation."""
        import plotly.express as px

        fig = px.histogram(sample_df, x="age", nbins=10)

        assert fig is not None
        assert fig.data[0].type == "histogram"

    def test_create_box_plot(self, sample_df):
        """Test box plot creation."""
        import plotly.express as px

        fig = px.box(
            sample_df,
            x="department",
            y="salary",
        )

        assert fig is not None
        assert fig.data[0].type == "box"

    def test_create_pie_chart(self, sample_df):
        """Test pie chart creation."""
        import plotly.express as px

        dept_counts = sample_df["department"].value_counts().reset_index()
        dept_counts.columns = ["department", "count"]

        fig = px.pie(dept_counts, values="count", names="department")

        assert fig is not None
        assert fig.data[0].type == "pie"

    def test_create_heatmap(self, sample_df_for_ml):
        """Test heatmap creation."""
        import plotly.express as px
        import plotly.graph_objects as go

        # Create correlation matrix
        numeric_df = sample_df_for_ml.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
        ))

        assert fig is not None
        assert fig.data[0].type == "heatmap"

    def test_figure_to_dict(self, sample_df):
        """Test converting figure to dictionary."""
        import plotly.express as px

        fig = px.scatter(sample_df, x="age", y="salary")
        fig_dict = fig.to_dict()

        assert isinstance(fig_dict, dict)
        assert "data" in fig_dict
        assert "layout" in fig_dict

    def test_figure_to_json(self, sample_df):
        """Test converting figure to JSON."""
        import plotly.express as px
        import json

        fig = px.scatter(sample_df, x="age", y="salary")
        fig_json = fig.to_json()

        # Should be valid JSON
        parsed = json.loads(fig_json)
        assert isinstance(parsed, dict)

    def test_figure_to_html(self, sample_df):
        """Test converting figure to HTML."""
        import plotly.express as px

        fig = px.scatter(sample_df, x="age", y="salary")
        html = fig.to_html()

        assert isinstance(html, str)
        assert "<div" in html or "<script" in html


class TestVisualizationEdgeCases:
    """Tests for edge cases in visualization."""

    def test_empty_dataframe_visualization(self):
        """Test visualization with empty DataFrame."""
        import plotly.express as px

        empty_df = pd.DataFrame({"x": [], "y": []})

        # Should not raise, but create empty plot
        fig = px.scatter(empty_df, x="x", y="y")
        assert fig is not None

    def test_single_point_visualization(self):
        """Test visualization with single data point."""
        import plotly.express as px

        single_df = pd.DataFrame({"x": [1], "y": [2]})

        fig = px.scatter(single_df, x="x", y="y")
        assert fig is not None
        assert len(fig.data[0].x) == 1

    def test_visualization_with_all_null_column(self):
        """Test visualization when a column is all null."""
        import plotly.express as px

        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [None, None, None],
        })

        # This should handle gracefully
        fig = px.scatter(df, x="x", y="y")
        assert fig is not None

    def test_large_dataset_visualization(self):
        """Test visualization with large dataset."""
        import plotly.express as px
        import numpy as np

        np.random.seed(42)
        large_df = pd.DataFrame({
            "x": np.random.randn(10000),
            "y": np.random.randn(10000),
        })

        fig = px.scatter(large_df, x="x", y="y")
        assert fig is not None
        assert len(fig.data[0].x) == 10000

    def test_categorical_color_scale(self, sample_df):
        """Test visualization with categorical color encoding."""
        import plotly.express as px

        fig = px.scatter(
            sample_df,
            x="age",
            y="salary",
            color="department",
        )

        # Should have multiple traces (one per department)
        assert len(fig.data) >= 2

    def test_custom_layout(self, sample_df):
        """Test customizing plot layout."""
        import plotly.express as px

        fig = px.scatter(sample_df, x="age", y="salary")

        fig.update_layout(
            title="Custom Title",
            xaxis_title="Age (years)",
            yaxis_title="Salary ($)",
            template="plotly_dark",
        )

        assert fig.layout.title.text == "Custom Title"
        assert fig.layout.xaxis.title.text == "Age (years)"
