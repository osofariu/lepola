"""
Simple repository pattern for document operations.

This module provides a clean interface for database operations using synchronous SQLite.
Can be easily upgraded to SQLAlchemy later without changing the service layer.
"""

import json
import sqlite3
from typing import Optional
from uuid import UUID

from src.core.config import settings
from src.core.logging import LoggingMixin
from src.core.models import Document, DocumentMetadata, ProcessingStatus


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
        with sqlite3.connect(self.db_path) as db:
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
