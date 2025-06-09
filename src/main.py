"""
Main FastAPI application for AI Legal & Policy Research Assistant.

This module provides the entry point for the application, setting up
the FastAPI server with all necessary routers and middleware.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import Settings
from src.core.logging import setup_logging
from src.ingestion.router import router as ingestion_router
from src.pipeline.router import router as pipeline_router
from src.querying.router import router as querying_router
from src.outputs.router import router as outputs_router

# Setup logging
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting AI Legal & Policy Research Assistant")

    # Initialize any required services here
    # await initialize_vector_store()
    # await initialize_llm_clients()

    yield

    # Shutdown
    logger.info("Shutting down AI Legal & Policy Research Assistant")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    settings = Settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="AI Legal & Policy Research Assistant",
        description="AI-powered research assistant for legal and policy documents",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        ingestion_router, prefix="/api/v1/ingestion", tags=["Document Ingestion"]
    )
    app.include_router(pipeline_router, prefix="/api/v1/pipeline", tags=["AI Pipeline"])
    app.include_router(
        querying_router, prefix="/api/v1/query", tags=["Interactive Querying"]
    )
    app.include_router(
        outputs_router, prefix="/api/v1/outputs", tags=["Output Generation"]
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "legal-policy-assistant"}

    return app


# Create the app instance
app = create_app()


def main() -> None:
    """Main entry point for the application."""
    settings = Settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
