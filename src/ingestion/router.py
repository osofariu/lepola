"""
FastAPI router for document ingestion endpoints.

This module provides REST API endpoints for uploading and ingesting
various types of legal and policy documents.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from src.core.models import DocumentUploadResponse, ErrorResponse, ProcessingStatus
from src.core.repository import document_repository
from src.ingestion.service import DocumentIngestionError, DocumentIngestionService

router = APIRouter()

# Initialize the ingestion service
ingestion_service = DocumentIngestionService()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        413: {"model": ErrorResponse, "description": "File Too Large"},
        422: {"model": ErrorResponse, "description": "Unsupported File Type"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Upload a document for ingestion",
    description="Upload a legal or policy document (PDF, TXT, DOCX, HTML) for processing and analysis.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    metadata: Optional[str] = Form(
        None, description="Optional metadata as JSON string"
    ),
    async_embedding: bool = Query(
        True, description="Run embedding and indexing asynchronously after ingestion"
    ),
) -> DocumentUploadResponse:
    """Upload and ingest a document file.

    Args:
        file: The uploaded file.
        metadata: Optional metadata as JSON string.
        async_embedding: Whether to run embedding/indexing asynchronously after ingestion.

    Returns:
        DocumentUploadResponse: Upload status and document ID.

    Raises:
        HTTPException: If upload or processing fails.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided"
            )

        # Get file size
        file_size = 0
        content = await file.read()
        file_size = len(content)

        # Reset file pointer and create a file-like object
        from io import BytesIO

        file_data = BytesIO(content)

        # Ingest the document
        document = await ingestion_service.ingest_file(
            file_data=file_data,
            filename=file.filename,
            file_size=file_size,
            run_embedding=async_embedding,
        )

        return DocumentUploadResponse(
            document_id=document.id,
            status=document.processing_status,
            message=f"Document '{file.filename}' uploaded and processed successfully",
        )

    except DocumentIngestionError as e:
        if "exceeds maximum" in str(e):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e)
            )
        elif "Unsupported file type" in str(e):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
            )
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during document upload: {str(e)}",
        )


@router.post(
    "/url",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Invalid URL or Content Type"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Ingest content from a URL",
    description="Fetch and ingest content from a web URL (HTML pages, PDFs, etc.).",
)
async def ingest_url(
    url: str = Form(..., description="URL to fetch content from"),
    metadata: Optional[str] = Form(
        None, description="Optional metadata as JSON string"
    ),
    async_embedding: bool = Query(
        True, description="Run embedding and indexing asynchronously after ingestion"
    ),
) -> DocumentUploadResponse:
    """Ingest content from a web URL.

    Args:
        url: The URL to fetch content from.
        metadata: Optional metadata as JSON string.
        async_embedding: Whether to run embedding/indexing asynchronously after ingestion.

    Returns:
        DocumentUploadResponse: Ingestion status and document ID.

    Raises:
        HTTPException: If URL ingestion fails.
    """
    try:
        # Ingest content from URL
        document = await ingestion_service.ingest_url(
            url, run_embedding=async_embedding
        )

        return DocumentUploadResponse(
            document_id=document.id,
            status=document.processing_status,
            message=f"Content from '{url}' ingested successfully",
        )

    except DocumentIngestionError as e:
        if "Invalid URL" in str(e) or "Failed to fetch" in str(e):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
            )
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during URL ingestion: {str(e)}",
        )


@router.get(
    "/document/{document_id}",
    summary="Get document information",
    description="Retrieve information about an ingested document by its ID.",
    responses={
        404: {"model": ErrorResponse, "description": "Document Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_document(document_id: UUID):
    """Get document information by ID.

    Args:
        document_id: The UUID of the document to retrieve.

    Returns:
        Document information.

    Raises:
        HTTPException: If document is not found or retrieval fails.
    """
    try:
        document = await ingestion_service.get_document_by_id(document_id)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        return document

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving document: {str(e)}",
        )


@router.get(
    "/documents",
    summary="List documents",
    description="List ingested documents with optional filtering and pagination.",
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def list_documents(
    limit: int = Query(50, ge=1, le=100, description="Number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    status: Optional[ProcessingStatus] = Query(
        None, description="Filter by processing status"
    ),
):
    """List documents with pagination and filtering.

    Args:
        limit: Maximum number of documents to return (1-100).
        offset: Number of documents to skip.
        file_type: Filter by file type.
        status: Filter by processing status.

    Returns:
        List of documents with metadata.

    Raises:
        HTTPException: If retrieval fails.
    """
    try:
        documents = document_repository.list_documents(
            limit=limit, offset=offset, file_type=file_type, status=status
        )

        total_count = document_repository.count(status=status)

        return {
            "documents": documents,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total_count,
                "has_prev": offset > 0,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving documents: {str(e)}",
        )


@router.get(
    "/status",
    summary="Get ingestion service status",
    description="Check the status and health of the document ingestion service.",
)
async def get_ingestion_status():
    """Get the status of the ingestion service.

    Returns:
        Service status information.
    """
    try:
        total_documents = document_repository.count()
        completed_documents = document_repository.count(ProcessingStatus.COMPLETED)
        failed_documents = document_repository.count(ProcessingStatus.FAILED)

        return {
            "status": "healthy",
            "service": "document-ingestion",
            "supported_types": ingestion_service.supported_types,
            "max_file_size": ingestion_service.max_file_size,
            "documents": {
                "total": total_documents,
                "completed": completed_documents,
                "failed": failed_documents,
            },
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "document-ingestion",
            "supported_types": ingestion_service.supported_types,
            "max_file_size": ingestion_service.max_file_size,
            "error": str(e),
        }
