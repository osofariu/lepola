"""
Simple SQLite database setup for document storage.

This module provides a lightweight database layer using SQLite with async support.
Can be easily upgraded to SQLAlchemy later without changing the repository interface.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import aiosqlite

from src.core.config import settings
from src.core.logging import LoggingMixin


class Database(LoggingMixin):
    """Simple SQLite database manager with async support."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.

        Args:
            database_url: Database URL. If None, uses settings.database_url
        """
        if database_url is None:
            database_url = settings.database_url

        # Extract file path from sqlite:/// URL
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]  # Remove sqlite:///
        else:
            self.db_path = database_url

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def get_connection(self) -> aiosqlite.Connection:
        """Get async database connection."""
        return await aiosqlite.connect(self.db_path, check_same_thread=False)

    async def init_database(self) -> None:
        """Initialize database schema."""
        # Use synchronous sqlite3 for initial setup to avoid threading issues
        with sqlite3.connect(self.db_path) as db:
            self._create_tables_sync(db)
            db.commit()

        self.logger.info("Database initialized", db_path=self.db_path)

    async def _create_tables(self, db: aiosqlite.Connection) -> None:
        """Create database tables."""

        # Documents table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                content TEXT NOT NULL,
                processing_status TEXT NOT NULL,
                error_message TEXT,
                source_url TEXT,
                checksum TEXT,
                file_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """
        )

        # Document metadata table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                title TEXT,
                author TEXT,
                subject TEXT,
                keywords TEXT,  -- JSON array as string
                creation_date TEXT,
                modification_date TEXT,
                page_count INTEGER,
                word_count INTEGER,
                language TEXT DEFAULT 'en',
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """
        )

        # Processing logs table (for future use)
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                processing_time_ms REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """
        )

        # Analysis results table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                confidence_level TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                model_used TEXT NOT NULL,
                warnings TEXT,  -- JSON array as string
                requires_human_review BOOLEAN NOT NULL,
                entities_source_analysis_id TEXT,  -- ID of analysis that provided entities
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                FOREIGN KEY (entities_source_analysis_id) REFERENCES analysis_results (id) ON DELETE SET NULL
            )
        """
        )

        # Extracted entities table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS extracted_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_text TEXT NOT NULL,
                start_position INTEGER NOT NULL,
                end_position INTEGER NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id) ON DELETE CASCADE
            )
        """
        )

        # Document summaries table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                executive_summary TEXT NOT NULL,
                key_points TEXT,  -- JSON array as string
                affected_groups TEXT,  -- JSON array as string
                legal_precedents TEXT,  -- JSON array as string
                implementation_timeline TEXT,
                confidence_score REAL NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id) ON DELETE CASCADE
            )
        """
        )

        # Key provisions table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS key_provisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_id INTEGER NOT NULL,
                provision_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                impact_assessment TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_section TEXT NOT NULL,
                affected_groups TEXT,  -- JSON array as string
                legal_references TEXT,  -- JSON array as string
                FOREIGN KEY (summary_id) REFERENCES document_summaries (id) ON DELETE CASCADE
            )
        """
        )

        # Risk assessments table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_id INTEGER NOT NULL,
                risk_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                description TEXT NOT NULL,
                affected_rights TEXT,  -- JSON array as string
                mitigation_suggestions TEXT,  -- JSON array as string
                confidence REAL NOT NULL,
                FOREIGN KEY (summary_id) REFERENCES document_summaries (id) ON DELETE CASCADE
            )
        """
        )

        # Embeddings table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                vector_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """
        )

        # Create indexes for better performance
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_document_id ON document_metadata(document_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_results_document_id ON analysis_results(document_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_extracted_entities_analysis_id ON extracted_entities(analysis_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_summaries_analysis_id ON document_summaries(analysis_id)"
        )

    def _create_tables_sync(self, db: sqlite3.Connection) -> None:
        """Create database tables synchronously."""

        # Documents table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                content TEXT NOT NULL,
                processing_status TEXT NOT NULL,
                error_message TEXT,
                source_url TEXT,
                checksum TEXT,
                file_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """
        )

        # Document metadata table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                title TEXT,
                author TEXT,
                subject TEXT,
                keywords TEXT,  -- JSON array as string
                creation_date TEXT,
                modification_date TEXT,
                page_count INTEGER,
                word_count INTEGER,
                language TEXT DEFAULT 'en',
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """
        )

        # Processing logs table (for future use)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                processing_time_ms REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """
        )

        # Analysis results table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                confidence_level TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                model_used TEXT NOT NULL,
                warnings TEXT,  -- JSON array as string
                requires_human_review BOOLEAN NOT NULL,
                entities_source_analysis_id TEXT,  -- ID of analysis that provided entities
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                FOREIGN KEY (entities_source_analysis_id) REFERENCES analysis_results (id) ON DELETE SET NULL
            )
        """
        )

        # Extracted entities table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS extracted_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_text TEXT NOT NULL,
                start_position INTEGER NOT NULL,
                end_position INTEGER NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id) ON DELETE CASCADE
            )
        """
        )

        # Document summaries table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                executive_summary TEXT NOT NULL,
                key_points TEXT,  -- JSON array as string
                affected_groups TEXT,  -- JSON array as string
                legal_precedents TEXT,  -- JSON array as string
                implementation_timeline TEXT,
                confidence_score REAL NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id) ON DELETE CASCADE
            )
        """
        )

        # Key provisions table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS key_provisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_id INTEGER NOT NULL,
                provision_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                impact_assessment TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_section TEXT NOT NULL,
                affected_groups TEXT,  -- JSON array as string
                legal_references TEXT,  -- JSON array as string
                FOREIGN KEY (summary_id) REFERENCES document_summaries (id) ON DELETE CASCADE
            )
        """
        )

        # Risk assessments table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_id INTEGER NOT NULL,
                risk_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                description TEXT NOT NULL,
                affected_rights TEXT,  -- JSON array as string
                mitigation_suggestions TEXT,  -- JSON array as string
                confidence REAL NOT NULL,
                FOREIGN KEY (summary_id) REFERENCES document_summaries (id) ON DELETE CASCADE
            )
        """
        )

        # Embeddings table
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                vector_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """
        )

        # Create indexes for better performance
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_document_id ON document_metadata(document_id)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_results_document_id ON analysis_results(document_id)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_extracted_entities_analysis_id ON extracted_entities(analysis_id)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_summaries_analysis_id ON document_summaries(analysis_id)"
        )


async def init_database() -> None:
    """Initialize the database (call on app startup)."""
    db_instance = Database()
    await db_instance.init_database()


async def get_db_connection() -> aiosqlite.Connection:
    """Get database connection for dependency injection."""
    db_instance = Database()
    return await db_instance.get_connection()
