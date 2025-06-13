"""
Integration tests for pipeline router with database connectivity.

Tests the complete flow from document retrieval to analysis job management.
"""

import tempfile
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.core.database import Database
from src.core.models import DocumentType, ProcessingStatus
from src.core.repository import document_repository
from src.main import app

# Test client
client = TestClient(app)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    import sqlite3

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"

        # Initialize database synchronously for testing
        with sqlite3.connect(str(db_path)) as db:
            db_instance = Database(str(db_path))
            db_instance._create_tables_sync(db)
            db.commit()

        # Override the global repository to use our test database
        original_db_path = document_repository.db_path
        document_repository.db_path = str(db_path)

        yield Database(str(db_path))

        # Restore original database path
        document_repository.db_path = original_db_path


@pytest.mark.integration
class TestPipelineIntegration:
    """Test pipeline integration with database."""

    def test_analyze_nonexistent_document(self, temp_db):
        """Test analyzing a document that doesn't exist."""
        fake_id = "12345678-1234-5678-9abc-123456789012"

        response = client.post(f"/api/v1/pipeline/analyze/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analyze_document_not_ready(self, temp_db):
        """Test analyzing a document that's not in completed status."""
        # Create a document in pending status
        from src.core.models import Document, DocumentMetadata

        document = Document(
            filename="test.txt",
            file_type=DocumentType.TEXT,
            file_size=100,
            content="Test content",
            metadata=DocumentMetadata(title="Test Doc"),
            processing_status=ProcessingStatus.PENDING,  # Not completed!
        )
        document_id = document_repository.create(document).id

        response = client.post(f"/api/v1/pipeline/analyze/{document_id}")

        assert response.status_code == 400
        assert "not ready for analysis" in response.json()["detail"]

    def test_successful_analysis_workflow(self, temp_db):
        """Test the complete analysis workflow."""
        # 1. Create a completed document
        from src.core.models import Document, DocumentMetadata

        document = Document(
            filename="legal_doc.txt",
            file_type=DocumentType.TEXT,
            file_size=150,
            content="This is a test legal document with some provisions and requirements.",
            metadata=DocumentMetadata(
                title="Test Legal Document", author="Test Author"
            ),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = document_repository.create(document).id

        # 2. Start analysis
        response = client.post(f"/api/v1/pipeline/analyze/{document_id}")

        assert response.status_code == 202
        data = response.json()
        assert "analysis_id" in data
        # Analysis runs immediately in test mode, so it should be completed
        assert data["status"] in ["completed", "queued", "failed"]

        analysis_id = data["analysis_id"]

        # 3. Check analysis status
        response = client.get(f"/api/v1/pipeline/analysis/{analysis_id}")

        assert response.status_code == 200
        analysis_data = response.json()
        assert analysis_data["analysis_id"] == analysis_id
        assert analysis_data["document_id"] == str(document_id)
        assert analysis_data["document_filename"] == "legal_doc.txt"

        # The analysis should be completed or failed (since we use mock LLM)
        assert analysis_data["status"] in ["completed", "failed", "processing"]

        # 4. If completed, check for results
        if analysis_data["status"] == "completed":
            assert "result" in analysis_data
            assert "confidence_level" in analysis_data
            assert "processing_time_ms" in analysis_data

            # Verify the result contains expected analysis components
            result = analysis_data["result"]
            assert "entities" in result
            assert "summary" in result
            assert "confidence_level" in result
            assert "processing_time_ms" in result

    def test_list_analyses(self, temp_db):
        """Test listing analysis jobs."""
        # Create a document and start analysis
        from src.core.models import Document, DocumentMetadata

        document = Document(
            filename="test_doc.txt",
            file_type=DocumentType.TEXT,
            file_size=75,
            content="Short test document for analysis listing.",
            metadata=DocumentMetadata(title="Test Document"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = document_repository.create(document).id

        # Start analysis
        start_response = client.post(f"/api/v1/pipeline/analyze/{document_id}")
        assert start_response.status_code == 202

        # List all analyses
        list_response = client.get("/api/v1/pipeline/analyses")

        assert list_response.status_code == 200
        data = list_response.json()
        assert "analyses" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

        # Should have at least one analysis
        assert data["total"] >= 1
        assert len(data["analyses"]) >= 1

        # Check analysis structure
        analysis = data["analyses"][0]
        assert "analysis_id" in analysis
        assert "document_id" in analysis
        assert "document_filename" in analysis
        assert "status" in analysis

    def test_list_analyses_with_filter(self, temp_db):
        """Test listing analyses with status filter."""
        # Create a document and start analysis
        from src.core.models import Document, DocumentMetadata

        document = Document(
            filename="filter_test.txt",
            file_type=DocumentType.TEXT,
            file_size=80,
            content="Document for testing status filtering.",
            metadata=DocumentMetadata(title="Filter Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = document_repository.create(document).id

        # Start analysis
        client.post(f"/api/v1/pipeline/analyze/{document_id}")

        # List completed analyses
        response = client.get("/api/v1/pipeline/analyses?status_filter=completed")

        assert response.status_code == 200
        data = response.json()
        assert data["status_filter"] == "completed"

        # All returned analyses should be completed
        for analysis in data["analyses"]:
            assert analysis["status"] == "completed"

    def test_pipeline_status(self, temp_db):
        """Test pipeline status endpoint."""
        response = client.get("/api/v1/pipeline/status")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "service" in data
        assert "model_available" in data
        assert "confidence_threshold" in data
        assert "jobs" in data

        # Check job statistics structure
        jobs = data["jobs"]
        assert "total" in jobs
        assert "completed" in jobs
        assert "failed" in jobs
        assert "processing" in jobs
        assert "queued" in jobs

    def test_analysis_pagination(self, temp_db):
        """Test pagination in analysis listing."""
        # Create multiple documents and analyses
        from src.core.models import Document, DocumentMetadata

        document_ids = []
        for i in range(5):
            document = Document(
                filename=f"test_doc_{i}.txt",
                file_type=DocumentType.TEXT,
                file_size=50 + i,
                content=f"Test document content {i}",
                metadata=DocumentMetadata(title=f"Test Document {i}"),
                processing_status=ProcessingStatus.COMPLETED,
            )
            doc_id = document_repository.create(document).id
            document_ids.append(doc_id)

            # Start analysis
            client.post(f"/api/v1/pipeline/analyze/{doc_id}")

        # Test pagination
        response = client.get("/api/v1/pipeline/analyses?limit=2&offset=0")
        assert response.status_code == 200

        data = response.json()
        assert len(data["analyses"]) <= 2
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert data["total"] >= 5  # At least the 5 we created

    def test_invalid_analysis_id(self, temp_db):
        """Test retrieving analysis with invalid ID."""
        fake_id = "invalid-uuid-format"

        response = client.get(f"/api/v1/pipeline/analysis/{fake_id}")

        # Should get a validation error for invalid UUID format
        assert response.status_code == 422

    def test_document_retrieval_error_handling(self, temp_db):
        """Test error handling when document retrieval fails."""
        # Use a valid UUID but one that doesn't exist
        nonexistent_id = "12345678-1234-5678-9abc-123456789012"

        response = client.post(f"/api/v1/pipeline/analyze/{nonexistent_id}")

        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
        assert nonexistent_id in error_data["detail"]
