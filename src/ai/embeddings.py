"""
FastAPI router for embeddings endpoints.

This module provides REST API endpoints for managing document embeddings,
including status checking, manual triggering, and similarity search.
"""

import time
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, status

from src.core.models import ErrorResponse, ProcessingStatus
from src.core.repository import document_repository, embedding_repository
from src.core.logging import (
    get_logger,
    log_async_operation_start,
    log_async_operation_complete,
)
from src.ingestion.embedding import process_document_embeddings

router = APIRouter()
logger = get_logger(__name__)


@router.get(
    "/status/{document_id}",
    summary="Get embedding status for a document",
    description="Check the embedding status and statistics for a specific document.",
    responses={
        404: {"model": ErrorResponse, "description": "Document Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_embedding_status(document_id: UUID):
    """Get embedding status for a document.

    Args:
        document_id: The UUID of the document to check.

    Returns:
        Embedding status information.

    Raises:
        HTTPException: If document is not found or retrieval fails.
    """
    try:
        # Check if document exists
        document = document_repository.get_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        # Get embeddings for the document
        embeddings = embedding_repository.get_by_document_id(str(document_id))

        # Calculate statistics
        total_chunks = len(embeddings)
        avg_chunk_length = (
            sum(len(emb.chunk_text) for emb in embeddings) / total_chunks
            if total_chunks > 0
            else 0
        )

        return {
            "document_id": str(document_id),
            "document_filename": document.filename,
            "document_status": document.processing_status.value,
            "embeddings": {
                "total_chunks": total_chunks,
                "average_chunk_length": round(avg_chunk_length, 2),
                "created_at": (
                    min(emb.created_at for emb in embeddings).isoformat()
                    if embeddings
                    else None
                ),
                "last_updated": (
                    max(emb.created_at for emb in embeddings).isoformat()
                    if embeddings
                    else None
                ),
            },
            "status": "completed" if total_chunks > 0 else "pending",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get embedding status",
            document_id=str(document_id),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving embedding status: {str(e)}",
        )


@router.post(
    "/process/{document_id}",
    summary="Manually trigger embeddings for a document",
    description="Manually trigger the embedding process for a document that hasn't been embedded yet.",
    responses={
        404: {"model": ErrorResponse, "description": "Document Not Found"},
        409: {"model": ErrorResponse, "description": "Embeddings Already Exist"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def trigger_embeddings(document_id: UUID):
    """Manually trigger embeddings for a document.

    Args:
        document_id: The UUID of the document to embed.

    Returns:
        Embedding trigger confirmation.

    Raises:
        HTTPException: If embedding cannot be triggered.
    """
    # Generate operation ID for tracking
    operation_id = str(uuid4())

    logger.info(
        "Manual embedding trigger requested",
        document_id=str(document_id),
        operation_id=operation_id,
    )

    try:
        # Check if document exists
        document = document_repository.get_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        # Check if embeddings already exist
        existing_embeddings = embedding_repository.get_by_document_id(str(document_id))
        if existing_embeddings:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Embeddings already exist for document {document_id}",
            )

        # Log async operation start
        log_async_operation_start(
            operation="manual_embedding_trigger",
            operation_id=operation_id,
            document_id=str(document_id),
            filename=document.filename,
        )

        # Start embedding process asynchronously
        import asyncio

        asyncio.create_task(
            process_document_embeddings(str(document_id), document.content)
        )

        logger.info(
            "Embedding process started",
            document_id=str(document_id),
            operation_id=operation_id,
        )

        return {
            "message": f"Embedding process started for document {document_id}",
            "document_id": str(document_id),
            "operation_id": operation_id,
            "status": "processing",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log async operation failure
        log_async_operation_complete(
            operation="manual_embedding_trigger",
            duration_ms=0,
            success=False,
            operation_id=operation_id,
            error=str(e),
        )

        logger.error(
            "Failed to trigger embeddings",
            document_id=str(document_id),
            error=str(e),
            operation_id=operation_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error triggering embeddings: {str(e)}",
        )


@router.get(
    "/search",
    summary="Search documents using embeddings",
    description="Search for documents using vector similarity with a query text.",
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def search_embeddings(
    query: str = Query(..., description="Search query text"),
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of results to return"
    ),
):
    """Search documents using embeddings.

    Args:
        query: The search query text.
        limit: Maximum number of results to return.

    Returns:
        Search results with similarity scores.

    Raises:
        HTTPException: If search fails.
    """
    try:
        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Embed the query text
        # 2. Search the FAISS index for similar vectors
        # 3. Return the most similar document chunks with scores

        # For now, return a placeholder response
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "message": "Embedding search not yet implemented",
        }

    except Exception as e:
        logger.error(
            "Embedding search failed",
            query=query,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during search: {str(e)}",
        )


@router.delete(
    "/{document_id}",
    summary="Delete embeddings for a document",
    description="Delete all embeddings associated with a specific document.",
    responses={
        404: {"model": ErrorResponse, "description": "Document Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def delete_embeddings(document_id: UUID):
    """Delete embeddings for a document.

    Args:
        document_id: The UUID of the document whose embeddings to delete.

    Returns:
        Deletion confirmation.

    Raises:
        HTTPException: If deletion fails.
    """
    try:
        # Check if document exists
        document = document_repository.get_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        # Delete embeddings
        deleted_count = embedding_repository.delete_by_document_id(str(document_id))

        logger.info(
            "Embeddings deleted",
            document_id=str(document_id),
            deleted_count=deleted_count,
        )

        return {
            "message": f"Deleted {deleted_count} embeddings for document {document_id}",
            "document_id": str(document_id),
            "deleted_count": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete embeddings",
            document_id=str(document_id),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error deleting embeddings: {str(e)}",
        )


@router.get(
    "/stats",
    summary="Get embedding statistics",
    description="Get overall statistics about the embedding system.",
)
async def get_embedding_stats():
    """Get embedding statistics.

    Returns:
        Embedding system statistics.
    """
    try:
        # Get basic statistics
        total_documents = document_repository.count()
        all_documents = document_repository.list_documents(
            limit=1000
        )  # Get all documents
        total_embeddings = sum(
            len(embedding_repository.get_by_document_id(str(doc.id)))
            for doc in all_documents
        )

        # Count documents with embeddings
        documents_with_embeddings = sum(
            1
            for doc in all_documents
            if embedding_repository.get_by_document_id(str(doc.id))
        )

        return {
            "total_documents": total_documents,
            "documents_with_embeddings": documents_with_embeddings,
            "total_embeddings": total_embeddings,
            "embedding_coverage": (
                round(documents_with_embeddings / total_documents * 100, 2)
                if total_documents > 0
                else 0
            ),
            "average_embeddings_per_document": (
                round(total_embeddings / documents_with_embeddings, 2)
                if documents_with_embeddings > 0
                else 0
            ),
        }

    except Exception as e:
        logger.error(
            "Failed to get embedding statistics",
            error=str(e),
            exc_info=True,
        )
        return {
            "error": str(e),
            "status": "degraded",
        }
