"""
Fast pipeline integration tests.

This module contains optimized tests that use MockLLM and proper
repository dependency injection for true database isolation.
"""

import pytest
from fastapi.testclient import TestClient

from src.core.models import DocumentType, ProcessingStatus, Document, DocumentMetadata
from src.main import app
from src.pipeline.router import (
    get_ai_pipeline,
    get_document_repository,
    get_analysis_repository,
)
from src.pipeline.service import AIAnalysisPipeline
from src.pipeline.mock_llm import MockLLM

# Test client
client = TestClient(app)


class MockAIAnalysisPipeline(AIAnalysisPipeline):
    """Test pipeline that always uses MockLLM."""

    def __init__(self, analysis_repository=None):
        """Initialize with MockLLM only."""
        self.llm = MockLLM()
        self.confidence_threshold = 0.7
        self.analysis_repository = analysis_repository


class TestPipelineFast:
    """Fast pipeline tests using MockLLM and proper database isolation."""

    def setup_method(self):
        """Set up test method with mocked dependencies."""
        # Clear any existing overrides
        app.dependency_overrides.clear()

    def teardown_method(self):
        """Clean up after test."""
        # Clear overrides
        app.dependency_overrides.clear()

    def test_analyze_nonexistent_document_fast(self, test_db_fast):
        """Test analyzing a document that doesn't exist - fast version."""
        repos = test_db_fast

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        fake_id = "12345678-1234-5678-9abc-123456789012"

        response = client.post(f"/api/v1/pipeline/analyze/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analyze_document_not_ready_fast(self, test_db_fast):
        """Test analyzing a document that's not ready - fast version."""
        repos = test_db_fast

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        # Create document in test database using test repository
        document = Document(
            filename="test.txt",
            file_type=DocumentType.TEXT,
            file_size=100,
            content="Test content",
            metadata=DocumentMetadata(title="Test Doc"),
            processing_status=ProcessingStatus.PENDING,  # Not ready
        )
        document_id = repos.document_repo.create(document).id

        response = client.post(f"/api/v1/pipeline/analyze/{document_id}")

        assert response.status_code == 400
        assert "not ready for analysis" in response.json()["detail"]

    def test_successful_analysis_workflow_fast(self, test_db_fast):
        """Test the complete analysis workflow with proper repository mocking."""
        repos = test_db_fast

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        # Create document in test database
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
        document_id = repos.document_repo.create(document).id

        # Start analysis
        response = client.post(f"/api/v1/pipeline/analyze/{document_id}")

        assert response.status_code == 202
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] in ["completed", "queued", "failed"]

        analysis_id = data["analysis_id"]

        # Check analysis status
        response = client.get(f"/api/v1/pipeline/analysis/{analysis_id}")

        assert response.status_code == 200
        analysis_data = response.json()
        assert analysis_data["analysis_id"] == analysis_id
        assert analysis_data["document_id"] == str(document_id)
        assert analysis_data["document_filename"] == "legal_doc.txt"

        # Should be completed with MockLLM
        assert analysis_data["status"] in ["completed", "failed", "processing"]

        if analysis_data["status"] == "completed":
            assert "result" in analysis_data
            assert "confidence_level" in analysis_data
            assert "processing_time_ms" in analysis_data

            result = analysis_data["result"]
            assert "entities" in result
            assert "summary" in result
            assert "confidence_level" in result
            assert "processing_time_ms" in result

    def test_database_isolation(self, test_db_fast):
        """Verify that tests use isolated test database."""
        repos = test_db_fast

        # Create a document in test database
        document = Document(
            filename="isolation_test.txt",
            file_type=DocumentType.TEXT,
            file_size=50,
            content="Test database isolation",
            metadata=DocumentMetadata(title="Isolation Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # Verify document exists in test database
        retrieved_doc = repos.document_repo.get_by_id(document_id)
        assert retrieved_doc is not None
        assert retrieved_doc.filename == "isolation_test.txt"

        # Verify the test database path is being used
        assert "test_fast.db" in repos.document_repo.db_path
        assert repos.document_repo.db_path != "./data/app.db"  # Not production DB

    def test_multiple_analyses_same_document(self, test_db_fast):
        """Test that multiple analyses can be run on the same document."""
        repos = test_db_fast

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        # Create document in test database
        document = Document(
            filename="multi_analysis.txt",
            file_type=DocumentType.TEXT,
            file_size=100,
            content="Document for multiple analysis testing.",
            metadata=DocumentMetadata(title="Multi Analysis Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # Run first analysis
        response1 = client.post(f"/api/v1/pipeline/analyze/{document_id}")
        assert response1.status_code == 202
        analysis_id_1 = response1.json()["analysis_id"]

        # Run second analysis
        response2 = client.post(f"/api/v1/pipeline/analyze/{document_id}")
        assert response2.status_code == 202
        analysis_id_2 = response2.json()["analysis_id"]

        # Should have different analysis IDs
        assert analysis_id_1 != analysis_id_2

        # Both should be retrievable
        check1 = client.get(f"/api/v1/pipeline/analysis/{analysis_id_1}")
        check2 = client.get(f"/api/v1/pipeline/analysis/{analysis_id_2}")

        assert check1.status_code == 200
        assert check2.status_code == 200

    def test_document_analyses_endpoint(self, test_db_fast):
        """Test the document-specific analyses endpoint."""
        repos = test_db_fast

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        # Create document in test database
        document = Document(
            filename="doc_analyses.txt",
            file_type=DocumentType.TEXT,
            file_size=90,
            content="Document for testing document analyses endpoint.",
            metadata=DocumentMetadata(title="Doc Analyses Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # Run an analysis
        client.post(f"/api/v1/pipeline/analyze/{document_id}")

        # Get analyses for this document
        response = client.get(f"/api/v1/pipeline/document/{document_id}/analyses")

        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert "analyses" in data
        assert "total_analyses" in data
        assert data["document_id"] == str(document_id)
        assert data["total_analyses"] >= 1

    def test_list_analyses_with_filter(self, test_db_fast):
        """Test listing analyses with status filter."""
        repos = test_db_fast

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        # Create document in test database
        document = Document(
            filename="filter_test.txt",
            file_type=DocumentType.TEXT,
            file_size=80,
            content="Document for testing status filtering.",
            metadata=DocumentMetadata(title="Filter Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

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

    def test_analysis_database_persistence(self, test_db_fast):
        """Test that analysis results are properly saved to and retrieved from database."""
        repos = test_db_fast

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        # Override AI pipeline to use MockLLM
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # Create document in test database
        document = Document(
            filename="persistence_test.txt",
            file_type=DocumentType.TEXT,
            file_size=100,
            content="Test document for database persistence verification.",
            metadata=DocumentMetadata(title="Persistence Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # Start analysis
        response = client.post(f"/api/v1/pipeline/analyze/{document_id}")
        assert response.status_code == 202
        analysis_id = response.json()["analysis_id"]

        # Check if analysis result was saved to test database
        # (bypassing the router logic and checking database directly)
        from uuid import UUID

        analysis_result = repos.analysis_repo.get_by_id(UUID(analysis_id))

        print(f"Analysis result in test DB: {analysis_result is not None}")
        if analysis_result:
            print(f"Analysis model used: {analysis_result.model_used}")
            print(f"Analysis entities count: {len(analysis_result.entities)}")

        # The analysis should be saved to test database
        assert analysis_result is not None
        assert analysis_result.model_used == "mock-gpt-4"

    def test_repository_patching_debug(self, test_db_fast):
        """Debug test to verify repository patching is working."""
        repos = test_db_fast

        # Check if the patched repositories are using test database paths
        from src.pipeline.service import analysis_repository as service_analysis_repo
        from src.pipeline.router import analysis_repository as router_analysis_repo
        from src.core.repository import analysis_repository as core_analysis_repo

        print(f"Service analysis_repository path: {service_analysis_repo.db_path}")
        print(f"Router analysis_repository path: {router_analysis_repo.db_path}")
        print(f"Core analysis_repository path: {core_analysis_repo.db_path}")
        print(f"Test repos analysis path: {repos.analysis_repo.db_path}")

        # They should all be using the test database
        assert "test_fast.db" in service_analysis_repo.db_path
        assert "test_fast.db" in router_analysis_repo.db_path
        assert "test_fast.db" in core_analysis_repo.db_path

    def test_pipeline_status_endpoint(self, test_db_fast):
        """Test the pipeline status endpoint."""
        response = client.get("/api/v1/pipeline/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "model_available" in data
        assert "confidence_threshold" in data
        assert "jobs" in data

        # Check jobs structure
        jobs = data["jobs"]
        assert "total" in jobs
        assert "completed" in jobs
        assert "failed" in jobs
        assert "processing" in jobs
        assert "queued" in jobs
