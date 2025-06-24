"""
Simple repository pattern for document operations.

This module provides a clean interface for database operations using synchronous SQLite.
Can be easily upgraded to SQLAlchemy later without changing the service layer.
"""

import json
import sqlite3
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from src.core.config import settings
from src.core.logging import LoggingMixin
from src.core.models import (
    Document,
    DocumentMetadata,
    ProcessingStatus,
    AnalysisResult,
    ExtractedEntity,
    DocumentSummary,
    KeyProvision,
    RiskAssessment,
    ConfidenceLevel,
    Embedding,
)


class DocumentRepository(LoggingMixin):
    """Repository for document database operations."""

    def __init__(self):
        """Initialize repository with database path."""
        database_url = settings.database_url
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]  # Remove sqlite:///
        else:
            self.db_path = database_url

    def create(self, document: Document) -> Document:
        """Create a new document in the database.

        Args:
            document: Document to create

        Returns:
            Created document
        """
        with sqlite3.connect(self.db_path, timeout=settings.database_timeout) as db:
            # Enable WAL mode for better concurrency
            db.execute("PRAGMA journal_mode=WAL")

            # Insert document
            db.execute(
                """
                INSERT INTO documents (
                    id, filename, file_type, file_size, content, 
                    processing_status, error_message, source_url, checksum, 
                    file_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(document.id),
                    document.filename,
                    document.file_type.value,
                    document.file_size,
                    document.content,
                    document.processing_status.value,
                    document.error_message,
                    document.source_url,
                    document.checksum,
                    getattr(document, "file_path", None),  # Will add this field later
                    document.created_at.isoformat(),
                    document.updated_at.isoformat() if document.updated_at else None,
                ),
            )

            # Insert metadata
            db.execute(
                """
                INSERT INTO document_metadata (
                    document_id, title, author, subject, keywords,
                    creation_date, modification_date, page_count, 
                    word_count, language
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(document.id),
                    document.metadata.title,
                    document.metadata.author,
                    document.metadata.subject,
                    json.dumps(document.metadata.keywords),  # Store as JSON string
                    (
                        document.metadata.creation_date.isoformat()
                        if document.metadata.creation_date
                        else None
                    ),
                    (
                        document.metadata.modification_date.isoformat()
                        if document.metadata.modification_date
                        else None
                    ),
                    document.metadata.page_count,
                    document.metadata.word_count,
                    document.metadata.language,
                ),
            )

            db.commit()

        self.logger.info("Document created in database", document_id=str(document.id))
        return document

    def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as db:
            # Get document
            cursor = db.execute(
                """
                SELECT id, filename, file_type, file_size, content,
                       processing_status, error_message, source_url, checksum,
                       file_path, created_at, updated_at
                FROM documents WHERE id = ?
            """,
                (str(document_id),),
            )
            doc_row = cursor.fetchone()

            if not doc_row:
                return None

            # Get metadata
            cursor = db.execute(
                """
                SELECT title, author, subject, keywords, creation_date,
                       modification_date, page_count, word_count, language
                FROM document_metadata WHERE document_id = ?
            """,
                (str(document_id),),
            )
            meta_row = cursor.fetchone()

        # Convert back to Pydantic models
        metadata = DocumentMetadata()
        if meta_row:
            metadata = DocumentMetadata(
                title=meta_row[0],
                author=meta_row[1],
                subject=meta_row[2],
                keywords=json.loads(meta_row[3]) if meta_row[3] else [],
                creation_date=meta_row[4],
                modification_date=meta_row[5],
                page_count=meta_row[6],
                word_count=meta_row[7],
                language=meta_row[8] or "en",
            )

        document = Document(
            id=UUID(doc_row[0]),
            filename=doc_row[1],
            file_type=doc_row[2],
            file_size=doc_row[3],
            content=doc_row[4],
            processing_status=ProcessingStatus(doc_row[5]),
            error_message=doc_row[6],
            source_url=doc_row[7],
            checksum=doc_row[8],
            metadata=metadata,
            created_at=doc_row[10],
            updated_at=doc_row[11],
        )

        return document

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        file_type: Optional[str] = None,
        status: Optional[ProcessingStatus] = None,
    ) -> list[Document]:
        """List documents with optional filtering.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            file_type: Filter by file type
            status: Filter by processing status

        Returns:
            List of documents
        """
        query = """
            SELECT d.id, d.filename, d.file_type, d.file_size, d.content,
                   d.processing_status, d.error_message, d.source_url, d.checksum,
                   d.file_path, d.created_at, d.updated_at,
                   m.title, m.author, m.subject, m.keywords, m.creation_date,
                   m.modification_date, m.page_count, m.word_count, m.language
            FROM documents d
            LEFT JOIN document_metadata m ON d.id = m.document_id
            WHERE 1=1
        """
        params = []

        if file_type:
            query += " AND d.file_type = ?"
            params.append(file_type)

        if status:
            query += " AND d.processing_status = ?"
            params.append(status.value)

        query += " ORDER BY d.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        documents = []
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(query, params)
            for row in cursor.fetchall():
                # Create metadata
                metadata = DocumentMetadata(
                    title=row[12],
                    author=row[13],
                    subject=row[14],
                    keywords=json.loads(row[15]) if row[15] else [],
                    creation_date=row[16],
                    modification_date=row[17],
                    page_count=row[18],
                    word_count=row[19],
                    language=row[20] or "en",
                )

                # Create document
                document = Document(
                    id=UUID(row[0]),
                    filename=row[1],
                    file_type=row[2],
                    file_size=row[3],
                    content=row[4],
                    processing_status=ProcessingStatus(row[5]),
                    error_message=row[6],
                    source_url=row[7],
                    checksum=row[8],
                    metadata=metadata,
                    created_at=row[10],
                    updated_at=row[11],
                )
                documents.append(document)

        return documents

    def update_status(
        self,
        document_id: UUID,
        status: ProcessingStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update document processing status.

        Args:
            document_id: Document ID
            status: New processing status
            error_message: Error message if status is FAILED

        Returns:
            True if document was updated, False if not found
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                """
                UPDATE documents 
                SET processing_status = ?, error_message = ?, updated_at = datetime('now')
                WHERE id = ?
            """,
                (status.value, error_message, str(document_id)),
            )

            db.commit()

        updated = cursor.rowcount > 0
        if updated:
            self.logger.info(
                "Document status updated",
                document_id=str(document_id),
                status=status.value,
            )

        return updated

    def delete(self, document_id: UUID) -> bool:
        """Delete document from database.

        Args:
            document_id: Document ID

        Returns:
            True if document was deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as db:
            # Delete document (metadata will be cascade deleted)
            cursor = db.execute(
                "DELETE FROM documents WHERE id = ?", (str(document_id),)
            )
            db.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.info(
                "Document deleted from database", document_id=str(document_id)
            )

        return deleted

    def count(self, status: Optional[ProcessingStatus] = None) -> int:
        """Count documents.

        Args:
            status: Filter by processing status

        Returns:
            Number of documents
        """
        query = "SELECT COUNT(*) FROM documents"
        params = []

        if status:
            query += " WHERE processing_status = ?"
            params.append(status.value)

        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(query, params)
            row = cursor.fetchone()
            return row[0] if row else 0


# Global repository instance
document_repository = DocumentRepository()


class AnalysisRepository(LoggingMixin):
    """Repository for analysis results database operations."""

    def __init__(self):
        """Initialize repository with database path."""
        database_url = settings.database_url
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]  # Remove sqlite:///
        else:
            self.db_path = database_url

    def create(self, analysis_result: AnalysisResult) -> AnalysisResult:
        """Create a new analysis result in the database.

        Args:
            analysis_result: Analysis result to create

        Returns:
            Created analysis result
        """
        # Use configurable timeout for long-running operations
        # and enable WAL mode for better concurrency
        with sqlite3.connect(self.db_path, timeout=settings.database_timeout) as db:
            # Enable WAL mode for better concurrency
            db.execute("PRAGMA journal_mode=WAL")

            # Insert analysis result
            db.execute(
                """
                INSERT INTO analysis_results (
                    id, document_id, confidence_level, processing_time_ms,
                    model_used, warnings, requires_human_review,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(analysis_result.id),
                    str(analysis_result.document_id),
                    analysis_result.confidence_level.value,
                    analysis_result.processing_time_ms,
                    analysis_result.model_used,
                    json.dumps(analysis_result.warnings),
                    int(analysis_result.requires_human_review),
                    analysis_result.created_at.isoformat(),
                    (
                        analysis_result.updated_at.isoformat()
                        if analysis_result.updated_at
                        else None
                    ),
                ),
            )

            # Insert extracted entities
            for entity in analysis_result.entities:
                db.execute(
                    """
                    INSERT INTO extracted_entities (
                        analysis_id, entity_type, entity_value, confidence,
                        source_text, start_position, end_position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(analysis_result.id),
                        entity.entity_type,
                        entity.entity_value,
                        entity.confidence,
                        entity.source_text,
                        entity.start_position,
                        entity.end_position,
                    ),
                )

            # Insert document summary
            summary_cursor = db.execute(
                """
                INSERT INTO document_summaries (
                    analysis_id, executive_summary, key_points, affected_groups,
                    legal_precedents, implementation_timeline, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(analysis_result.id),
                    analysis_result.summary.executive_summary,
                    json.dumps(analysis_result.summary.key_points),
                    json.dumps(analysis_result.summary.affected_groups),
                    json.dumps(analysis_result.summary.legal_precedents),
                    analysis_result.summary.implementation_timeline,
                    analysis_result.summary.confidence_score,
                ),
            )
            summary_id = summary_cursor.lastrowid

            # Insert key provisions
            for provision in analysis_result.summary.main_provisions:
                db.execute(
                    """
                    INSERT INTO key_provisions (
                        summary_id, provision_type, title, description,
                        impact_assessment, confidence, source_section,
                        affected_groups, legal_references
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary_id,
                        provision.provision_type,
                        provision.title,
                        provision.description,
                        provision.impact_assessment,
                        provision.confidence,
                        provision.source_section,
                        json.dumps(provision.affected_groups),
                        json.dumps(provision.legal_references),
                    ),
                )

            # Insert risk assessments
            for risk in analysis_result.summary.risk_assessments:
                db.execute(
                    """
                    INSERT INTO risk_assessments (
                        summary_id, risk_type, risk_level, description,
                        affected_rights, mitigation_suggestions, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary_id,
                        risk.risk_type,
                        risk.risk_level,
                        risk.description,
                        json.dumps(risk.affected_rights),
                        json.dumps(risk.mitigation_suggestions),
                        risk.confidence,
                    ),
                )

            db.commit()

        self.logger.info(
            "Analysis result created in database",
            analysis_id=str(analysis_result.id),
            document_id=str(analysis_result.document_id),
        )
        return analysis_result

    def get_by_id(self, analysis_id: UUID) -> Optional[AnalysisResult]:
        """Get analysis result by ID.

        Args:
            analysis_id: Analysis ID

        Returns:
            Analysis result if found, None otherwise
        """
        with sqlite3.connect(self.db_path, timeout=settings.database_timeout) as db:
            # Enable WAL mode for better concurrency
            db.execute("PRAGMA journal_mode=WAL")

            # Get analysis result
            cursor = db.execute(
                """
                SELECT id, document_id, confidence_level, processing_time_ms,
                       model_used, warnings, requires_human_review,
                       created_at, updated_at
                FROM analysis_results WHERE id = ?
                """,
                (str(analysis_id),),
            )
            result_row = cursor.fetchone()

            if not result_row:
                return None

            # Get extracted entities
            cursor = db.execute(
                """
                SELECT entity_type, entity_value, confidence, source_text,
                       start_position, end_position
                FROM extracted_entities WHERE analysis_id = ?
                """,
                (str(analysis_id),),
            )
            entity_rows = cursor.fetchall()

            # Get document summary
            cursor = db.execute(
                """
                SELECT id, executive_summary, key_points, affected_groups,
                       legal_precedents, implementation_timeline, confidence_score
                FROM document_summaries WHERE analysis_id = ?
                """,
                (str(analysis_id),),
            )
            summary_row = cursor.fetchone()

            if not summary_row:
                return None

            summary_id = summary_row[0]

            # Get key provisions
            cursor = db.execute(
                """
                SELECT provision_type, title, description, impact_assessment,
                       confidence, source_section, affected_groups, legal_references
                FROM key_provisions WHERE summary_id = ?
                """,
                (summary_id,),
            )
            provision_rows = cursor.fetchall()

            # Get risk assessments
            cursor = db.execute(
                """
                SELECT risk_type, risk_level, description, affected_rights,
                       mitigation_suggestions, confidence
                FROM risk_assessments WHERE summary_id = ?
                """,
                (summary_id,),
            )
            risk_rows = cursor.fetchall()

        # Convert to Pydantic models
        entities = [
            ExtractedEntity(
                entity_type=row[0],
                entity_value=row[1],
                confidence=row[2],
                source_text=row[3],
                start_position=row[4],
                end_position=row[5],
            )
            for row in entity_rows
        ]

        provisions = [
            KeyProvision(
                provision_type=row[0],
                title=row[1],
                description=row[2],
                impact_assessment=row[3],
                confidence=row[4],
                source_section=row[5],
                affected_groups=json.loads(row[6]) if row[6] else [],
                legal_references=json.loads(row[7]) if row[7] else [],
            )
            for row in provision_rows
        ]

        risks = [
            RiskAssessment(
                risk_type=row[0],
                risk_level=row[1],
                description=row[2],
                affected_rights=json.loads(row[3]) if row[3] else [],
                mitigation_suggestions=json.loads(row[4]) if row[4] else [],
                confidence=row[5],
            )
            for row in risk_rows
        ]

        summary = DocumentSummary(
            executive_summary=summary_row[1],
            key_points=json.loads(summary_row[2]) if summary_row[2] else [],
            main_provisions=provisions,
            risk_assessments=risks,
            affected_groups=json.loads(summary_row[3]) if summary_row[3] else [],
            legal_precedents=json.loads(summary_row[4]) if summary_row[4] else [],
            implementation_timeline=summary_row[5],
            confidence_score=summary_row[6],
        )

        # Parse datetime strings
        created_at = datetime.fromisoformat(result_row[7])
        updated_at = datetime.fromisoformat(result_row[8]) if result_row[8] else None

        analysis_result = AnalysisResult(
            id=UUID(result_row[0]),
            document_id=UUID(result_row[1]),
            entities=entities,
            summary=summary,
            confidence_level=ConfidenceLevel(result_row[2]),
            processing_time_ms=result_row[3],
            model_used=result_row[4],
            warnings=json.loads(result_row[5]) if result_row[5] else [],
            requires_human_review=bool(result_row[6]),
            created_at=created_at,
            updated_at=updated_at,
        )

        return analysis_result

    def list_by_document_id(self, document_id: UUID) -> List[AnalysisResult]:
        """List all analysis results for a document.

        Args:
            document_id: Document ID

        Returns:
            List of analysis results
        """
        with sqlite3.connect(self.db_path, timeout=settings.database_timeout) as db:
            # Enable WAL mode for better concurrency
            db.execute("PRAGMA journal_mode=WAL")

            cursor = db.execute(
                """
                SELECT id FROM analysis_results 
                WHERE document_id = ? 
                ORDER BY created_at DESC
                """,
                (str(document_id),),
            )
            analysis_ids = [UUID(row[0]) for row in cursor.fetchall()]

        # Get full analysis results
        results = []
        for analysis_id in analysis_ids:
            result = self.get_by_id(analysis_id)
            if result:
                results.append(result)

        return results

    def list_all(
        self, limit: int = 50, offset: int = 0, requires_review: Optional[bool] = None
    ) -> List[AnalysisResult]:
        """List all analysis results with optional filtering.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip
            requires_review: Filter by human review requirement

        Returns:
            List of analysis results
        """
        query = "SELECT id FROM analysis_results WHERE 1=1"
        params = []

        if requires_review is not None:
            query += " AND requires_human_review = ?"
            params.append(int(requires_review))

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(query, params)
            analysis_ids = [UUID(row[0]) for row in cursor.fetchall()]

        # Get full analysis results
        results = []
        for analysis_id in analysis_ids:
            result = self.get_by_id(analysis_id)
            if result:
                results.append(result)

        return results

    def delete(self, analysis_id: UUID) -> bool:
        """Delete analysis result from database.

        Args:
            analysis_id: Analysis ID

        Returns:
            True if analysis was deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "DELETE FROM analysis_results WHERE id = ?", (str(analysis_id),)
            )
            db.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.info(
                "Analysis result deleted from database", analysis_id=str(analysis_id)
            )

        return deleted


# Global repository instances
analysis_repository = AnalysisRepository()


class EmbeddingRepository(LoggingMixin):
    """Repository for embedding database operations."""

    def __init__(self):
        database_url = settings.database_url
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]
        else:
            self.db_path = database_url

    def create(self, embedding: Embedding) -> Embedding:
        """Create a new embedding record in the database."""
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                """
                INSERT INTO embeddings (
                    document_id, chunk_id, vector_id, chunk_text, start_pos, end_pos, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding.document_id,
                    embedding.chunk_id,
                    embedding.vector_id,
                    embedding.chunk_text,
                    embedding.start_pos,
                    embedding.end_pos,
                    embedding.created_at.isoformat(),
                ),
            )
            db.commit()
            embedding.id = cursor.lastrowid
        self.logger.info("Embedding created in database", embedding_id=embedding.id)
        return embedding

    def get_by_document_id(self, document_id: str) -> list[Embedding]:
        """Get all embeddings for a document."""
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                """
                SELECT id, document_id, chunk_id, vector_id, chunk_text, start_pos, end_pos, created_at
                FROM embeddings WHERE document_id = ?
                ORDER BY vector_id ASC
                """,
                (document_id,),
            )
            rows = cursor.fetchall()
        return [
            Embedding(
                id=row[0],
                document_id=row[1],
                chunk_id=row[2],
                vector_id=row[3],
                chunk_text=row[4],
                start_pos=row[5],
                end_pos=row[6],
                created_at=datetime.fromisoformat(row[7]),
            )
            for row in rows
        ]

    def list_by_document_id(self, document_id: str) -> list[Embedding]:
        """Alias for get_by_document_id for consistency."""
        return self.get_by_document_id(document_id)

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all embeddings for a document. Returns number deleted."""
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "DELETE FROM embeddings WHERE document_id = ?", (document_id,)
            )
            db.commit()
            deleted = cursor.rowcount
        self.logger.info(
            "Embeddings deleted for document", document_id=document_id, count=deleted
        )
        return deleted


# Global repository instance
embedding_repository = EmbeddingRepository()
