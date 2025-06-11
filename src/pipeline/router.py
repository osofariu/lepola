"""
FastAPI router for AI pipeline endpoints.

This module provides REST API endpoints for triggering AI analysis
of ingested documents.
"""

from uuid import UUID, uuid4
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from src.core.models import AnalysisStartResponse, ErrorResponse
from src.core.repository import document_repository
from src.pipeline.service import AIAnalysisError, AIAnalysisPipeline

router = APIRouter()

# Initialize the AI pipeline
ai_pipeline = AIAnalysisPipeline()

# Simple in-memory store for analysis jobs (in production, use Redis/database)
analysis_jobs = {}


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
async def analyze_document(document_id: UUID):
    """Start AI analysis of a document.

    Args:
        document_id: ID of the document to analyze.

    Returns:
        AnalysisStartResponse: Analysis startup confirmation.

    Raises:
        HTTPException: If analysis cannot be started.
    """
    try:
        # 1. Retrieve the document from the database
        document = document_repository.get_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found",
            )

        # 2. Check if document is in a valid state for analysis
        if document.processing_status.value != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document {document_id} is not ready for analysis. Status: {document.processing_status.value}",
            )

        # 3. Generate unique analysis ID
        analysis_id = uuid4()

        # 4. Store analysis job info (in production, this would be in a proper job queue)
        analysis_jobs[str(analysis_id)] = {
            "document_id": str(document_id),
            "status": "queued",
            "created_at": document.created_at.isoformat(),
            "document_filename": document.filename,
            "document_type": document.file_type.value,
        }

        # 5. Queue the analysis job (for now, we'll start it immediately)
        # In production, this would be sent to a background task queue like Celery
        try:
            # Start analysis in background (simplified for now)
            analysis_jobs[str(analysis_id)]["status"] = "processing"

            # Perform the analysis
            analysis_result = await ai_pipeline.analyze_document(document)

            # Store the results
            analysis_jobs[str(analysis_id)].update(
                {
                    "status": "completed",
                    "result": analysis_result.model_dump(),
                    "completed_at": analysis_result.created_at.isoformat(),
                    "confidence_level": analysis_result.confidence_level.value,
                    "processing_time_ms": analysis_result.processing_time_ms,
                }
            )

        except AIAnalysisError as e:
            analysis_jobs[str(analysis_id)].update(
                {
                    "status": "failed",
                    "error": str(e),
                }
            )
            # Don't raise here, we've queued the job and it will show as failed
        except Exception as e:
            analysis_jobs[str(analysis_id)].update(
                {
                    "status": "failed",
                    "error": f"Unexpected error: {str(e)}",
                }
            )

        return AnalysisStartResponse(
            analysis_id=analysis_id,
            status="queued",
            estimated_completion_time=None,  # Could estimate based on document size
        )

    except HTTPException:
        raise
    except Exception as e:
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
async def get_analysis_results(analysis_id: UUID):
    """Get analysis results by ID.

    Args:
        analysis_id: ID of the analysis to retrieve.

    Returns:
        Analysis results.

    Raises:
        HTTPException: If analysis results cannot be retrieved.
    """
    try:
        # Check if analysis exists
        analysis_job = analysis_jobs.get(str(analysis_id))
        if not analysis_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis {analysis_id} not found",
            )

        # Return different responses based on status
        if analysis_job["status"] == "queued":
            return {
                "analysis_id": str(analysis_id),
                "status": "queued",
                "document_id": analysis_job["document_id"],
                "document_filename": analysis_job["document_filename"],
                "created_at": analysis_job["created_at"],
            }

        elif analysis_job["status"] == "processing":
            return {
                "analysis_id": str(analysis_id),
                "status": "processing",
                "document_id": analysis_job["document_id"],
                "document_filename": analysis_job["document_filename"],
                "created_at": analysis_job["created_at"],
            }

        elif analysis_job["status"] == "completed":
            return {
                "analysis_id": str(analysis_id),
                "status": "completed",
                "document_id": analysis_job["document_id"],
                "document_filename": analysis_job["document_filename"],
                "created_at": analysis_job["created_at"],
                "completed_at": analysis_job["completed_at"],
                "confidence_level": analysis_job["confidence_level"],
                "processing_time_ms": analysis_job["processing_time_ms"],
                "result": analysis_job["result"],
            }

        elif analysis_job["status"] == "failed":
            return {
                "analysis_id": str(analysis_id),
                "status": "failed",
                "document_id": analysis_job["document_id"],
                "document_filename": analysis_job["document_filename"],
                "created_at": analysis_job["created_at"],
                "error": analysis_job["error"],
            }

        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unknown analysis status: {analysis_job['status']}",
            )

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
):
    """List all analysis jobs with pagination and filtering.

    Args:
        limit: Maximum number of results to return.
        offset: Number of results to skip.
        status_filter: Filter by status (queued, processing, completed, failed).

    Returns:
        List of analysis jobs.
    """
    try:
        # Filter jobs if status_filter is provided
        jobs = analysis_jobs
        if status_filter:
            jobs = {
                k: v for k, v in analysis_jobs.items() if v["status"] == status_filter
            }

        # Convert to list and apply pagination
        job_list = [
            {"analysis_id": analysis_id, **job_data}
            for analysis_id, job_data in jobs.items()
        ]

        # Sort by creation time (most recent first)
        job_list.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        paginated_jobs = job_list[offset : offset + limit]

        return {
            "analyses": paginated_jobs,
            "total": len(job_list),
            "limit": limit,
            "offset": offset,
            "status_filter": status_filter,
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
async def get_pipeline_status():
    """Get the status of the AI pipeline service.

    Returns:
        Service status information.
    """
    try:
        # Calculate job statistics
        total_jobs = len(analysis_jobs)
        completed_jobs = sum(
            1 for job in analysis_jobs.values() if job["status"] == "completed"
        )
        failed_jobs = sum(
            1 for job in analysis_jobs.values() if job["status"] == "failed"
        )
        processing_jobs = sum(
            1 for job in analysis_jobs.values() if job["status"] == "processing"
        )
        queued_jobs = sum(
            1 for job in analysis_jobs.values() if job["status"] == "queued"
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
