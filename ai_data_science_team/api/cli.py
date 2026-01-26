"""
CLI for running the AI Data Science Team API server.

Usage:
    python -m ai_data_science_team.api.cli [OPTIONS]

Or if installed:
    ai-ds-team-api [OPTIONS]
"""

import argparse
import logging
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Data Science Team REST API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level",
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="+",
        default=["*"],
        help="Allowed CORS origins",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    from ai_data_science_team.api.app import run_server

    print(f"Starting AI Data Science Team API server on {args.host}:{args.port}")
    print(f"API docs available at http://{args.host}:{args.port}/docs")

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
