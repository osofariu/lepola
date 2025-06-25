"""
FastAPI router for AI pipeline endpoints.

This module provides REST API endpoints for triggering AI analysis
of ingested documents.
"""

import time
from uuid import UUID, uuid4
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends

from src.core.models import AnalysisStartResponse, ErrorResponse
from src.core.repository import DocumentRepository, AnalysisRepository
from src.core.logging import (
    get_logger,
    log_async_operation_start,
    log_async_operation_complete,
)
from src.pipeline.service import AIAnalysisError, AIAnalysisPipeline

router = APIRouter()
logger = get_logger(__name__)

# Simple in-memory store for analysis job status tracking (for async operations)
# In production, use Redis/database with job queue
analysis_job_status = {}


def get_document_repository() -> DocumentRepository:
    """Dependency injection for document repository - can be mocked in tests."""
    from src.core.repository import document_repository

    return document_repository


def get_analysis_repository() -> AnalysisRepository:
    """Dependency injection for analysis repository - can be mocked in tests."""
    from src.core.repository import analysis_repository

    return analysis_repository


def get_ai_pipeline(
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
) -> AIAnalysisPipeline:
    """Dependency injection for AI pipeline - can be mocked in tests."""
    return AIAnalysisPipeline(analysis_repository=analysis_repo)


@router.post(
    "/analyze/{document_id}",
    response_model=AnalysisStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Document Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Start document analysis",
    description="Trigger AI analysis of an ingested document.",
)
async def analyze_document(
    document_id: UUID,
    force_regenerate_entities: bool = False,
    ai_pipeline: AIAnalysisPipeline = Depends(get_ai_pipeline),
    document_repo: DocumentRepository = Depends(get_document_repository),
):
    """Start AI analysis of a document.

    Args:
        document_id: ID of the document to analyze.
        force_regenerate_entities: If True, regenerate entities even if they exist.
        ai_pipeline: AI pipeline service (injected).
        document_repo: Document repository (injected).

    Returns:
        AnalysisStartResponse: Analysis startup confirmation.

    Raises:
        HTTPException: If analysis cannot be started.
    """
    # Generate operation ID for tracking
    operation_id = str(uuid4())

    logger.info(
        "Document analysis requested",
        document_id=str(document_id),
        operation_id=operation_id,
    )

    try:
        # 1. Retrieve the document from the database
        logger.info(
            "Retrieving document from database",
            document_id=str(document_id),
            operation_id=operation_id,
        )

        document = document_repo.get_by_id(document_id)
        if not document:
            logger.warning(
                "Document not found",
                document_id=str(document_id),
                operation_id=operation_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        logger.info(
            "Document retrieved successfully",
            document_id=str(document_id),
            filename=document.filename,
            file_type=document.file_type.value,
            file_size=document.file_size,
            operation_id=operation_id,
        )

        # 2. Check if document is in a valid state for analysis
        if document.processing_status.value != "completed":
            logger.warning(
                "Document not ready for analysis",
                document_id=str(document_id),
                status=document.processing_status.value,
                operation_id=operation_id,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document {document_id} is not ready for analysis. Status: {document.processing_status.value}",
            )

        # 3. Start the analysis job
        logger.info(
            "Starting AI analysis",
            document_id=str(document_id),
            operation_id=operation_id,
        )

        # Log async operation start
        log_async_operation_start(
            operation="document_analysis",
            operation_id=operation_id,
            document_id=str(document_id),
            filename=document.filename,
            file_size=document.file_size,
        )

        analysis_start_time = time.time()
        analysis_id = None
        analysis_status = "failed"  # Default status

        try:
            # Perform the analysis
            analysis_result = await ai_pipeline.analyze_document(
                document, force_regenerate_entities
            )

            # Use the actual analysis result ID
            analysis_id = analysis_result.id
            analysis_status = "completed"
            analysis_duration_ms = (time.time() - analysis_start_time) * 1000

            # Log async operation completion
            log_async_operation_complete(
                operation="document_analysis",
                duration_ms=analysis_duration_ms,
                success=True,
                operation_id=operation_id,
                analysis_id=str(analysis_id),
                confidence_level=analysis_result.confidence_level.value,
                processing_time_ms=analysis_result.processing_time_ms,
            )

            # Store job status for tracking
            analysis_job_status[str(analysis_id)] = {
                "document_id": str(document_id),
                "status": analysis_status,
                "created_at": document.created_at.isoformat(),
                "document_filename": document.filename,
                "document_type": document.file_type.value,
                "result": analysis_result.model_dump(),
                "completed_at": analysis_result.created_at.isoformat(),
                "confidence_level": analysis_result.confidence_level.value,
                "processing_time_ms": analysis_result.processing_time_ms,
            }

            logger.info(
                "Analysis completed successfully",
                document_id=str(document_id),
                analysis_id=str(analysis_id),
                confidence_level=analysis_result.confidence_level.value,
                processing_time_ms=analysis_result.processing_time_ms,
                operation_id=operation_id,
            )

        except AIAnalysisError as e:
            # Generate temporary ID for failed analysis
            analysis_id = uuid4()
            analysis_status = "failed"
            analysis_duration_ms = (time.time() - analysis_start_time) * 1000

            # Log async operation failure
            log_async_operation_complete(
                operation="document_analysis",
                duration_ms=analysis_duration_ms,
                success=False,
                operation_id=operation_id,
                analysis_id=str(analysis_id),
                error=str(e),
            )

            logger.error(
                "Analysis failed with AIAnalysisError",
                document_id=str(document_id),
                analysis_id=str(analysis_id),
                error=str(e),
                operation_id=operation_id,
            )

            try:
                analysis_job_status[str(analysis_id)] = {
                    "document_id": str(document_id),
                    "status": analysis_status,
                    "created_at": document.created_at.isoformat(),
                    "document_filename": document.filename,
                    "document_type": document.file_type.value,
                    "error": str(e),
                }
            except Exception:
                # If we can't even store the status, we'll still have the variables set
                pass

        except Exception as e:
            # Generate temporary ID for failed analysis
            analysis_id = uuid4()
            analysis_status = "failed"
            analysis_duration_ms = (time.time() - analysis_start_time) * 1000

            # Log async operation failure
            log_async_operation_complete(
                operation="document_analysis",
                duration_ms=analysis_duration_ms,
                success=False,
                operation_id=operation_id,
                analysis_id=str(analysis_id),
                error=str(e),
            )

            logger.error(
                "Analysis failed with unexpected error",
                document_id=str(document_id),
                analysis_id=str(analysis_id),
                error=str(e),
                operation_id=operation_id,
                exc_info=True,
            )

            try:
                analysis_job_status[str(analysis_id)] = {
                    "document_id": str(document_id),
                    "status": analysis_status,
                    "created_at": document.created_at.isoformat(),
                    "document_filename": document.filename,
                    "document_type": document.file_type.value,
                    "error": f"Unexpected error: {str(e)}",
                }
            except Exception:
                # If we can't even store the status, we'll still have the variables set
                pass

        # Ensure we always have an analysis_id
        if analysis_id is None:
            analysis_id = uuid4()

        logger.info(
            "Analysis job completed",
            document_id=str(document_id),
            analysis_id=str(analysis_id),
            status=analysis_status,
            operation_id=operation_id,
        )

        return AnalysisStartResponse(
            analysis_id=analysis_id,
            status=analysis_status,
            estimated_completion_time=None,  # Could estimate based on document size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in analyze_document endpoint",
            document_id=str(document_id),
            operation_id=operation_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis: {str(e)}",
        )


@router.get(
    "/analysis/{analysis_id}",
    summary="Get analysis results",
    description="Retrieve the results of a completed document analysis.",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_analysis_results(
    analysis_id: UUID,
    document_repo: DocumentRepository = Depends(get_document_repository),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
):
    """Get analysis results by ID.

    Args:
        analysis_id: ID of the analysis to retrieve.

    Returns:
        Analysis results.

    Raises:
        HTTPException: If analysis results cannot be retrieved.
    """
    try:
        # First check if it's a currently running job
        analysis_job = analysis_job_status.get(str(analysis_id))
        if analysis_job and analysis_job["status"] in ["queued", "processing"]:
            return {
                "analysis_id": str(analysis_id),
                "status": analysis_job["status"],
                "document_id": analysis_job["document_id"],
                "document_filename": analysis_job["document_filename"],
                "created_at": analysis_job["created_at"],
            }

        # Try to get completed analysis from database
        analysis_result = analysis_repo.get_by_id(analysis_id)
        if not analysis_result:
            # Check if it's a failed job
            if analysis_job and analysis_job["status"] == "failed":
                return {
                    "analysis_id": str(analysis_id),
                    "status": "failed",
                    "document_id": analysis_job["document_id"],
                    "document_filename": analysis_job["document_filename"],
                    "created_at": analysis_job["created_at"],
                    "error": analysis_job["error"],
                }

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis {analysis_id} not found",
            )

        # Get document info for response
        document = document_repo.get_by_id(analysis_result.document_id)
        document_filename = document.filename if document else "unknown"

        return {
            "analysis_id": str(analysis_id),
            "status": "completed",
            "document_id": str(analysis_result.document_id),
            "document_filename": document_filename,
            "created_at": analysis_result.created_at.isoformat(),
            "completed_at": analysis_result.created_at.isoformat(),
            "confidence_level": analysis_result.confidence_level.value,
            "processing_time_ms": analysis_result.processing_time_ms,
            "result": analysis_result.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analysis: {str(e)}",
        )


@router.get(
    "/analyses",
    summary="List all analysis jobs",
    description="Get a list of all analysis jobs with their current status.",
)
async def list_analyses(
    limit: int = 10,
    offset: int = 0,
    status_filter: Optional[str] = None,
    requires_review: Optional[bool] = None,
    document_repo: DocumentRepository = Depends(get_document_repository),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
):
    """List all analysis jobs with pagination and filtering.

    Args:
        limit: Maximum number of results to return.
        offset: Number of results to skip.
        status_filter: Filter by status (queued, processing, completed, failed).
        requires_review: Filter by human review requirement (only applies to completed analyses).

    Returns:
        List of analysis jobs.
    """
    try:
        # Get completed analyses from database
        completed_analyses = []
        if not status_filter or status_filter == "completed":
            db_results = analysis_repo.list_all(
                limit=limit * 2,  # Get extra in case we need to filter
                offset=0,
                requires_review=(
                    requires_review if status_filter == "completed" else None
                ),
            )

            for result in db_results:
                document = document_repo.get_by_id(result.document_id)
                completed_analyses.append(
                    {
                        "analysis_id": str(result.id),
                        "document_id": str(result.document_id),
                        "document_filename": (
                            document.filename if document else "unknown"
                        ),
                        "status": "completed",
                        "created_at": result.created_at.isoformat(),
                        "completed_at": result.created_at.isoformat(),
                        "confidence_level": result.confidence_level.value,
                        "processing_time_ms": result.processing_time_ms,
                        "requires_human_review": result.requires_human_review,
                    }
                )

        # Get in-progress jobs from memory
        in_progress_jobs = []
        if not status_filter or status_filter in ["queued", "processing", "failed"]:
            jobs = analysis_job_status
            if status_filter:
                jobs = {
                    k: v
                    for k, v in analysis_job_status.items()
                    if v["status"] == status_filter
                }

            in_progress_jobs = [
                {"analysis_id": analysis_id, **job_data}
                for analysis_id, job_data in jobs.items()
            ]

        # Combine and sort all jobs
        all_jobs = completed_analyses + in_progress_jobs
        all_jobs.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        total_count = len(all_jobs)
        paginated_jobs = all_jobs[offset : offset + limit]

        return {
            "analyses": paginated_jobs,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "status_filter": status_filter,
            "requires_review": requires_review,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list analyses: {str(e)}",
        )


@router.get(
    "/status",
    summary="Get pipeline status",
    description="Check the status and health of the AI pipeline service.",
)
async def get_pipeline_status(
    ai_pipeline: AIAnalysisPipeline = Depends(get_ai_pipeline),
):
    """Get the status of the AI pipeline service.

    Returns:
        Service status information.
    """
    try:
        # Calculate job statistics
        total_jobs = len(analysis_job_status)
        completed_jobs = sum(
            1 for job in analysis_job_status.values() if job["status"] == "completed"
        )
        failed_jobs = sum(
            1 for job in analysis_job_status.values() if job["status"] == "failed"
        )
        processing_jobs = sum(
            1 for job in analysis_job_status.values() if job["status"] == "processing"
        )
        queued_jobs = sum(
            1 for job in analysis_job_status.values() if job["status"] == "queued"
        )

        return {
            "status": "healthy",
            "service": "ai-pipeline",
            "model_available": True,  # Could check LLM connectivity
            "confidence_threshold": ai_pipeline.confidence_threshold,
            "jobs": {
                "total": total_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "processing": processing_jobs,
                "queued": queued_jobs,
            },
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "ai-pipeline",
            "model_available": False,
            "error": str(e),
        }


@router.get(
    "/document/{document_id}/analyses",
    summary="Get analyses for a document",
    description="Get all analysis results for a specific document.",
    responses={
        404: {"model": ErrorResponse, "description": "Document Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_document_analyses(
    document_id: UUID,
    document_repo: DocumentRepository = Depends(get_document_repository),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
):
    """Get all analysis results for a specific document.

    Args:
        document_id: ID of the document to get analyses for.

    Returns:
        List of analysis results for the document.

    Raises:
        HTTPException: If document is not found or retrieval fails.
    """
    try:
        # Check if document exists
        document = document_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        # Get all analyses for this document
        analyses = analysis_repo.list_by_document_id(document_id)

        # Format response
        analysis_list = []
        for analysis in analyses:
            analysis_list.append(
                {
                    "analysis_id": str(analysis.id),
                    "document_id": str(analysis.document_id),
                    "document_filename": document.filename,
                    "status": "completed",
                    "created_at": analysis.created_at.isoformat(),
                    "confidence_level": analysis.confidence_level.value,
                    "processing_time_ms": analysis.processing_time_ms,
                    "requires_human_review": analysis.requires_human_review,
                    "model_used": analysis.model_used,
                    "entity_count": len(analysis.entities),
                    "warning_count": len(analysis.warnings),
                }
            )

        return {
            "document_id": str(document_id),
            "document_filename": document.filename,
            "total_analyses": len(analysis_list),
            "analyses": analysis_list,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document analyses: {str(e)}",
        )


@router.delete(
    "/analysis/{analysis_id}",
    summary="Delete an analysis result",
    description="Delete a completed analysis result from the database.",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def delete_analysis(
    analysis_id: UUID,
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
):
    """Delete an analysis result.

    Args:
        analysis_id: ID of the analysis to delete.

    Returns:
        Confirmation of deletion.

    Raises:
        HTTPException: If analysis is not found or deletion fails.
    """
    try:
        # Check if analysis exists in database
        analysis_result = analysis_repo.get_by_id(analysis_id)
        if not analysis_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis {analysis_id} not found",
            )

        # Delete from database
        success = analysis_repo.delete(analysis_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete analysis {analysis_id}",
            )

        # Also remove from in-memory status if present
        analysis_job_status.pop(str(analysis_id), None)

        return {
            "message": f"Analysis {analysis_id} deleted successfully",
            "analysis_id": str(analysis_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete analysis: {str(e)}",
        )
