"""
Logging configuration for AI Legal & Policy Research Assistant.

This module sets up structured logging using structlog with appropriate
processors for development and production environments.
"""

import logging
import sys
from typing import Any, Dict
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import structlog
from structlog.types import Processor


class ImmediateFlushRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler that flushes after each write."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record and flush immediately."""
        super().emit(record)
        if self.stream is not None:
            self.stream.flush()


def setup_logging(log_level: str = "INFO") -> None:
    """Set up structured logging for the application.

    Args:
        log_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"app_{timestamp}.log"

    # Set log level to DEBUG if DEBUG environment variable is true
    import os

    if os.getenv("DEBUG", "false").lower() == "true":
        log_level = "DEBUG"

    # Configure standard library logging with rotating file handler
    file_handler = ImmediateFlushRotatingFileHandler(
        filename=log_file,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,  # Keep 5 backup files
        encoding="utf-8",
        delay=False,  # Create file immediately
    )

    # Set handler level to match root logger
    file_handler.setLevel(getattr(logging, log_level.upper()))

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)

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
    """Simple debug logger that writes to the application log file.

    Use this for debugging when you need immediate visibility.
    Logs will be written to the same rotating log file as the rest of the application.

    Args:
        *args: Arguments to log (like print)
        **kwargs: Additional key-value pairs to log

    Example:
        debug_log("LLM Config:", config_dict)
        debug_log("Processing document", doc_id="123", status="active")
    """
    logger = get_logger("debug")

    # Format the message
    if args:
        message = " ".join(str(arg) for arg in args)
    else:
        message = "DEBUG"

    # Log with both the message and any additional kwargs
    logger.debug(message, **kwargs)


def log_endpoint_start(
    endpoint: str,
    method: str,
    path: str,
    request_id: str = None,
    user_id: str = None,
    **kwargs: Any,
) -> None:
    """Log when an endpoint starts processing.

    Args:
        endpoint: Name of the endpoint (e.g., "analyze_document").
        method: HTTP method (GET, POST, etc.).
        path: Request path.
        request_id: Optional request ID for tracking.
        user_id: Optional user ID for the request.
        **kwargs: Additional fields to log.
    """
    logger = get_logger("endpoint")

    log_data = {
        "endpoint": endpoint,
        "method": method,
        "path": path,
        "event_type": "start",
        **kwargs,
    }

    if request_id:
        log_data["request_id"] = request_id

    if user_id:
        log_data["user_id"] = user_id

    logger.info("Endpoint started", **log_data)


def log_endpoint_complete(
    endpoint: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str = None,
    user_id: str = None,
    **kwargs: Any,
) -> None:
    """Log when an endpoint completes processing.

    Args:
        endpoint: Name of the endpoint (e.g., "analyze_document").
        method: HTTP method (GET, POST, etc.).
        path: Request path.
        status_code: HTTP status code.
        duration_ms: Request duration in milliseconds.
        request_id: Optional request ID for tracking.
        user_id: Optional user ID for the request.
        **kwargs: Additional fields to log.
    """
    logger = get_logger("endpoint")

    log_data = {
        "endpoint": endpoint,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "event_type": "complete",
        **kwargs,
    }

    if request_id:
        log_data["request_id"] = request_id

    if user_id:
        log_data["user_id"] = user_id

    if status_code >= 400:
        logger.warning("Endpoint completed with error", **log_data)
    else:
        logger.info("Endpoint completed successfully", **log_data)


def log_async_operation_start(
    operation: str,
    operation_id: str = None,
    **kwargs: Any,
) -> None:
    """Log when an async operation starts.

    Args:
        operation: Type of async operation (e.g., "document_analysis").
        operation_id: Optional operation ID for tracking.
        **kwargs: Additional fields to log.
    """
    logger = get_logger("async")

    log_data = {
        "operation": operation,
        "event_type": "start",
        **kwargs,
    }

    if operation_id:
        log_data["operation_id"] = operation_id

    logger.info("Async operation started", **log_data)


def log_async_operation_complete(
    operation: str,
    duration_ms: float,
    success: bool = True,
    operation_id: str = None,
    error: str = None,
    **kwargs: Any,
) -> None:
    """Log when an async operation completes.

    Args:
        operation: Type of async operation (e.g., "document_analysis").
        duration_ms: Operation duration in milliseconds.
        success: Whether the operation was successful.
        operation_id: Optional operation ID for tracking.
        error: Error message if operation failed.
        **kwargs: Additional fields to log.
    """
    logger = get_logger("async")

    log_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
        "event_type": "complete",
        **kwargs,
    }

    if operation_id:
        log_data["operation_id"] = operation_id

    if error:
        log_data["error"] = error

    if success:
        logger.info("Async operation completed successfully", **log_data)
    else:
        logger.error("Async operation failed", **log_data)
