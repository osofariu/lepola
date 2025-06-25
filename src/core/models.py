"""
Core data models for AI Legal & Policy Research Assistant.

This module defines the main Pydantic models used throughout the application
for data validation and serialization.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return current UTC datetime using the new timezone-aware API."""
    return datetime.now(UTC)


class DocumentType(str, Enum):
    """Enumeration of supported document types."""

    PDF = "pdf"
    TEXT = "txt"
    DOCX = "docx"
    HTML = "html"
    URL = "url"


class ProcessingStatus(str, Enum):
    """Enumeration of document processing statuses."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Enumeration of AI confidence levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OutputFormat(str, Enum):
    """Enumeration of output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PDF = "pdf"


class BaseEntity(BaseModel):
    """Base model with common fields for all entities."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    created_at: datetime = Field(
        default_factory=utc_now, description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )


class DocumentMetadata(BaseModel):
    """Metadata for ingested documents."""

    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    subject: Optional[str] = Field(default=None, description="Document subject")
    keywords: list[str] = Field(default_factory=list, description="Document keywords")
    creation_date: Optional[datetime] = Field(
        default=None, description="Document creation date"
    )
    modification_date: Optional[datetime] = Field(
        default=None, description="Document modification date"
    )
    page_count: Optional[int] = Field(default=None, description="Number of pages")
    word_count: Optional[int] = Field(default=None, description="Number of words")
    language: Optional[str] = Field(default="en", description="Document language")


class Document(BaseEntity):
    """Document model for ingested legal/policy documents."""

    filename: str = Field(..., description="Original filename")
    file_type: DocumentType = Field(..., description="Type of document")
    file_size: int = Field(..., description="File size in bytes")
    content: str = Field(..., description="Extracted text content")
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Document metadata"
    )
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING, description="Processing status"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )
    source_url: Optional[str] = Field(
        default=None, description="Source URL if document was fetched from web"
    )
    checksum: Optional[str] = Field(
        default=None, description="File checksum for integrity verification"
    )


class ExtractedEntity(BaseModel):
    """Model for entities extracted from documents."""

    entity_type: str = Field(
        ..., description="Type of entity (law, agency, group, etc.)"
    )
    entity_value: str = Field(..., description="The extracted entity value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_text: str = Field(..., description="Original text where entity was found")
    start_position: int = Field(..., description="Start position in document")
    end_position: int = Field(..., description="End position in document")


class KeyProvision(BaseModel):
    """Model for key provisions identified in documents."""

    provision_type: str = Field(
        ..., description="Type of provision (requirement, prohibition, etc.)"
    )
    title: str = Field(..., description="Provision title or summary")
    description: str = Field(..., description="Detailed description")
    impact_assessment: str = Field(..., description="Assessment of potential impact")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_section: str = Field(
        ..., description="Document section where provision was found"
    )
    affected_groups: list[str] = Field(
        default_factory=list, description="Groups affected by this provision"
    )
    legal_references: list[str] = Field(
        default_factory=list, description="Referenced laws or precedents"
    )


class RiskAssessment(BaseModel):
    """Model for risk assessments of legal documents."""

    risk_type: str = Field(
        ..., description="Type of risk (civil rights, privacy, etc.)"
    )
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    description: str = Field(..., description="Risk description")
    affected_rights: list[str] = Field(
        default_factory=list, description="Affected rights or protections"
    )
    mitigation_suggestions: list[str] = Field(
        default_factory=list, description="Suggested mitigations"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class DocumentSummary(BaseModel):
    """Model for document summaries."""

    executive_summary: str = Field(..., description="High-level executive summary")
    key_points: list[str] = Field(..., description="Key points from the document")
    main_provisions: list[KeyProvision] = Field(
        ..., description="Main provisions identified"
    )
    risk_assessments: list[RiskAssessment] = Field(..., description="Risk assessments")
    affected_groups: list[str] = Field(
        ..., description="Groups affected by the document"
    )
    legal_precedents: list[str] = Field(..., description="Relevant legal precedents")
    implementation_timeline: Optional[str] = Field(
        default=None, description="Implementation timeline if specified"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence in summary"
    )


class AnalysisResult(BaseEntity):
    """Model for complete document analysis results."""

    document_id: UUID = Field(..., description="ID of the analyzed document")
    entities: list[ExtractedEntity] = Field(..., description="Extracted entities")
    summary: DocumentSummary = Field(..., description="Document summary")
    confidence_level: ConfidenceLevel = Field(
        ..., description="Overall confidence level"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    model_used: str = Field(..., description="AI model used for analysis")
    warnings: list[str] = Field(default_factory=list, description="Analysis warnings")
    requires_human_review: bool = Field(
        default=False, description="Whether human review is recommended"
    )
    entities_source_analysis_id: Optional[UUID] = Field(
        default=None,
        description="ID of the analysis that provided the entities (for caching)",
    )


class Query(BaseModel):
    """Model for user queries."""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="User's question"
    )
    document_ids: Optional[list[UUID]] = Field(
        default=None, description="Specific documents to query against"
    )
    max_results: int = Field(
        default=5, ge=1, le=20, description="Maximum number of results to return"
    )
    include_context: bool = Field(
        default=True, description="Whether to include source context"
    )


class QueryResult(BaseModel):
    """Model for query results."""

    answer: str = Field(..., description="Answer to the user's question")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the answer"
    )
    sources: list[dict[str, Any]] = Field(..., description="Source citations")
    related_documents: list[UUID] = Field(..., description="Related document IDs")
    suggestions: list[str] = Field(
        default_factory=list, description="Follow-up question suggestions"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about the answer"
    )


class OutputRequest(BaseModel):
    """Model for output generation requests."""

    analysis_id: UUID = Field(
        ..., description="ID of the analysis to generate output for"
    )
    format: OutputFormat = Field(..., description="Desired output format")
    include_sources: bool = Field(
        default=True, description="Whether to include source citations"
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include document metadata"
    )
    template: Optional[str] = Field(default=None, description="Custom template to use")


class GeneratedOutput(BaseEntity):
    """Model for generated outputs."""

    analysis_id: UUID = Field(..., description="ID of the source analysis")
    format: OutputFormat = Field(..., description="Output format")
    content: str = Field(..., description="Generated content")
    file_path: Optional[str] = Field(default=None, description="Path to generated file")
    size_bytes: int = Field(..., description="Output size in bytes")


class HealthCheck(BaseModel):
    """Model for health check responses."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(
        default_factory=utc_now, description="Health check timestamp"
    )
    dependencies: dict[str, str] = Field(
        default_factory=dict, description="Dependency status"
    )


# Request/Response models for API endpoints
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    document_id: UUID = Field(..., description="ID of the uploaded document")
    status: ProcessingStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class AnalysisStartResponse(BaseModel):
    """Response model for starting document analysis."""

    analysis_id: UUID = Field(..., description="ID of the started analysis")
    estimated_completion_time: Optional[datetime] = Field(
        default=None, description="Estimated completion time"
    )
    status: str = Field(..., description="Analysis status")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=utc_now, description="Error timestamp")


class Embedding(BaseModel):
    """Model for document chunk embeddings and their metadata.

    Attributes:
        id: Unique identifier for the embedding record.
        document_id: ID of the source document.
        chunk_id: Unique identifier for the chunk within the document.
        vector_id: Index of the embedding in the FAISS index.
        chunk_text: The text content of the chunk.
        start_pos: Start character position of the chunk in the document.
        end_pos: End character position of the chunk in the document.
        created_at: Timestamp when the embedding was created.
    """

    id: Optional[int] = Field(
        default=None, description="Unique identifier for the embedding record"
    )
    document_id: str = Field(..., description="ID of the source document")
    chunk_id: str = Field(
        ..., description="Unique identifier for the chunk within the document"
    )
    vector_id: int = Field(..., description="Index of the embedding in the FAISS index")
    chunk_text: str = Field(..., description="The text content of the chunk")
    start_pos: int = Field(
        ..., description="Start character position of the chunk in the document"
    )
    end_pos: int = Field(
        ..., description="End character position of the chunk in the document"
    )
    created_at: datetime = Field(
        default_factory=utc_now, description="Timestamp when the embedding was created"
    )
