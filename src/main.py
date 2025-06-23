"""
Main FastAPI application for AI Legal & Policy Research Assistant.

This module provides the entry point for the application, setting up
the FastAPI server with all necessary routers and middleware.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import Settings
from src.core.database import init_database
from src.core.logging import setup_logging, log_endpoint_start, log_endpoint_complete
from src.ingestion.router import router as ingestion_router
from src.pipeline.router import router as pipeline_router
from src.querying.router import router as querying_router
from src.outputs.router import router as outputs_router
from src.ai.embeddings import router as embeddings_router

# Setup logging
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting AI Legal & Policy Research Assistant")

    # Initialize database
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e), exc_info=True)
        raise

    # Initialize any other required services here
    # await initialize_vector_store()
    # await initialize_llm_clients()

    yield

    # Shutdown
    logger.info("Shutting down AI Legal & Policy Research Assistant")


async def logging_middleware(request: Request, call_next):
    """Middleware to log all endpoint requests with timing."""
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    # Record start time
    start_time = time.time()

    # Log endpoint start
    log_endpoint_start(
        endpoint=f"{request.method}_{request.url.path.replace('/', '_').strip('_')}",
        method=request.method,
        path=str(request.url.path),
        request_id=request_id,
        query_params=dict(request.query_params) if request.query_params else None,
    )

    try:
        # Process the request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log endpoint completion
        log_endpoint_complete(
            endpoint=f"{request.method}_{request.url.path.replace('/', '_').strip('_')}",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )

        return response

    except Exception as e:
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log endpoint error
        log_endpoint_complete(
            endpoint=f"{request.method}_{request.url.path.replace('/', '_').strip('_')}",
            method=request.method,
            path=str(request.url.path),
            status_code=500,
            duration_ms=duration_ms,
            request_id=request_id,
            error=str(e),
        )

        # Re-raise the exception
        raise


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

    # Add logging middleware first
    app.middleware("http")(logging_middleware)

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
    app.include_router(
        embeddings_router, prefix="/api/v1/embeddings", tags=["Embeddings"]
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
