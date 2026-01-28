"""
CLI tool for AI Data Science Team.

Usage:
    ai-ds-team --help
    ai-ds-team version
    ai-ds-team agent list
    ai-ds-team pipeline export <pipeline_file>
"""

import argparse
import sys

from ai_data_science_team._version import __version__


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="ai-ds-team",
        description="AI Data Science Team - Build and run AI-powered data science workflows",
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"ai-data-science-team {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    # Agent commands
    agent_parser = subparsers.add_parser("agent", help="Agent management commands")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command")

    agent_list_parser = agent_subparsers.add_parser("list", help="List available agents")

    # Pipeline commands
    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline management commands")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_command")

    pipeline_export_parser = pipeline_subparsers.add_parser("export", help="Export pipeline")
    pipeline_export_parser.add_argument("file", help="Pipeline file to export")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "version":
        print(f"ai-data-science-team version {__version__}")

    elif args.command == "agent":
        if args.agent_command == "list":
            list_agents()
        else:
            agent_parser.print_help()

    elif args.command == "pipeline":
        if args.pipeline_command == "export":
            export_pipeline(args.file)
        else:
            pipeline_parser.print_help()

    else:
        parser.print_help()


def list_agents():
    """List all available agents."""
    agents = [
        ("DataCleaningAgent", "Clean and preprocess data"),
        ("DataWranglingAgent", "Transform and reshape data"),
        ("DataVisualizationAgent", "Create visualizations with Plotly"),
        ("FeatureEngineeringAgent", "Engineer features for ML"),
        ("SQLDatabaseAgent", "Query SQL databases"),
        ("DataLoaderToolsAgent", "Load data from files"),
        ("EDAToolsAgent", "Exploratory data analysis"),
        ("H2OMLAgent", "Train ML models with H2O AutoML"),
        ("MLflowToolsAgent", "Track experiments with MLflow"),
        ("WorkflowPlannerAgent", "Plan data science workflows"),
    ]

    print("\nAvailable Agents:")
    print("-" * 60)
    for name, description in agents:
        print(f"  {name:<25} {description}")
    print()


def export_pipeline(filepath: str):
    """Export a pipeline to a standalone script."""
    print(f"Exporting pipeline from: {filepath}")
    print("Pipeline export feature coming soon!")


if __name__ == "__main__":
    main()
