"""
Logging configuration for AI Legal & Policy Research Assistant.

This module sets up structured logging using structlog with appropriate
processors for development and production environments.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import Processor


def setup_logging(log_level: str = "INFO") -> None:
    """Set up structured logging for the application.

    Args:
        log_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=get_processors(),
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_processors() -> list[Processor]:
    """Get the list of structlog processors based on environment.

    Returns:
        List of structlog processors for log formatting.
    """
    # In development, use human-readable format
    # In production, use JSON format for better log aggregation
    import os

    if os.getenv("DEBUG", "false").lower() == "true":
        return [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        return [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: The name of the logger, typically __name__.

    Returns:
        A configured structlog BoundLogger instance.
    """
    return structlog.get_logger(name)


def log_request_response(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: str = None,
    **kwargs: Any,
) -> None:
    """Log HTTP request/response information.

    Args:
        method: HTTP method (GET, POST, etc.).
        path: Request path.
        status_code: HTTP status code.
        duration_ms: Request duration in milliseconds.
        user_id: Optional user ID for the request.
        **kwargs: Additional fields to log.
    """
    logger = get_logger("http")

    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        **kwargs,
    }

    if user_id:
        log_data["user_id"] = user_id

    if status_code >= 400:
        logger.warning("HTTP error", **log_data)
    else:
        logger.info("HTTP request", **log_data)


def log_ai_operation(
    operation: str,
    model: str,
    tokens_used: int = None,
    confidence: float = None,
    duration_ms: float = None,
    **kwargs: Any,
) -> None:
    """Log AI/LLM operation information.

    Args:
        operation: Type of AI operation (e.g., "summarize", "extract_entities").
        model: Model used for the operation.
        tokens_used: Number of tokens consumed.
        confidence: Confidence score of the operation.
        duration_ms: Operation duration in milliseconds.
        **kwargs: Additional fields to log.
    """
    logger = get_logger("ai")

    log_data = {
        "operation": operation,
        "model": model,
        **kwargs,
    }

    if tokens_used is not None:
        log_data["tokens_used"] = tokens_used

    if confidence is not None:
        log_data["confidence"] = confidence

    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    logger.info("AI operation completed", **log_data)


def log_document_processing(
    document_id: str,
    file_type: str,
    file_size: int,
    pages: int = None,
    processing_time_ms: float = None,
    success: bool = True,
    error: str = None,
    **kwargs: Any,
) -> None:
    """Log document processing information.

    Args:
        document_id: Unique identifier for the document.
        file_type: Type of file processed (pdf, txt, etc.).
        file_size: Size of the file in bytes.
        pages: Number of pages in the document.
        processing_time_ms: Processing time in milliseconds.
        success: Whether processing was successful.
        error: Error message if processing failed.
        **kwargs: Additional fields to log.
    """
    logger = get_logger("document")

    log_data = {
        "document_id": document_id,
        "file_type": file_type,
        "file_size": file_size,
        "success": success,
        **kwargs,
    }

    if pages is not None:
        log_data["pages"] = pages

    if processing_time_ms is not None:
        log_data["processing_time_ms"] = processing_time_ms

    if error:
        log_data["error"] = error
        logger.error("Document processing failed", **log_data)
    else:
        logger.info("Document processing completed", **log_data)


class LoggingMixin:
    """Mixin class to add logging capabilities to other classes."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get a logger instance bound to this class."""
        return get_logger(self.__class__.__name__)


def debug_log(*args, **kwargs) -> None:
    """Simple debug logger that always shows up in server output.

    Use this for debugging when you need immediate visibility.

    Args:
        *args: Arguments to log (like print)
        **kwargs: Additional key-value pairs to log

    Example:
        debug_log("LLM Config:", config_dict)
        debug_log("Processing document", doc_id="123", status="active")
    """
    import sys
    import time

    # Create a simple timestamp
    timestamp = time.strftime("%H:%M:%S")

    # Format the message
    if args:
        message = " ".join(str(arg) for arg in args)
    else:
        message = "DEBUG"

    # Add kwargs if provided
    if kwargs:
        kwargs_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
        message = f"{message} | {kwargs_str}"

    # Force output to stderr (which uvicorn doesn't capture)
    print(f"🐛 [{timestamp}] {message}", file=sys.stderr, flush=True)
