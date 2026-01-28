"""
FastAPI application factory.

This module provides the FastAPI application setup and configuration.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Global app instance
_app: Optional[FastAPI] = None

# Version
__version__ = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting AI Data Science Team API server")

    # Initialize task store
    app.state.tasks = {}
    app.state.results = {}

    yield

    # Shutdown
    logger.info("Shutting down AI Data Science Team API server")


def create_app(
    title: str = "AI Data Science Team API",
    description: str = "REST API for AI-powered data science workflows",
    version: str = __version__,
    cors_origins: Optional[list] = None,
    debug: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    title : str
        API title.
    description : str
        API description.
    version : str
        API version.
    cors_origins : list, optional
        Allowed CORS origins. Defaults to ["*"].
    debug : bool
        Enable debug mode.

    Returns
    -------
    FastAPI
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        debug=debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    cors_origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if debug else None,
            },
        )

    # Include routers
    from ai_data_science_team.api.routes import router
    app.include_router(router)

    return app


def get_app() -> FastAPI:
    """
    Get or create the global FastAPI application instance.

    Returns
    -------
    FastAPI
        The application instance.
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
):
    """
    Run the API server.

    Parameters
    ----------
    host : str
        Host to bind to.
    port : int
        Port to bind to.
    reload : bool
        Enable auto-reload for development.
    workers : int
        Number of worker processes.
    log_level : str
        Logging level.
    """
    import uvicorn

    uvicorn.run(
        "ai_data_science_team.api.app:get_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        factory=True,
    )
