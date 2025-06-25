"""
Integration tests for pipeline router with database connectivity.

Tests the complete flow from document retrieval to analysis job management
using proper repository dependency injection for true database isolation.
"""

import warnings

# Suppress the FastAPI TestClient deprecation warning
warnings.filterwarnings(
    "ignore",
    message="The 'app' shortcut is now deprecated.*",
    category=DeprecationWarning,
)

from fastapi.testclient import TestClient

from src.core.models import (
    DocumentType,
    ProcessingStatus,
    Document,
    DocumentMetadata,
    AnalysisResult,
)
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
    """Test pipeline that always uses MockLLM for integration tests."""

    def __init__(self, analysis_repository=None):
        """Initialize with MockLLM only."""
        self.llm = MockLLM()
        self.confidence_threshold = 0.7
        self.analysis_repository = analysis_repository

    async def analyze_document(
        self, document, force_regenerate_entities: bool = False
    ) -> AnalysisResult:
        """Override to support the new parameter."""
        return await super().analyze_document(document, force_regenerate_entities)


class TestPipelineIntegration:
    """Integration tests for pipeline with proper database isolation."""

    def setup_method(self):
        """Set up test method with mocked dependencies."""
        # Clear any existing overrides
        app.dependency_overrides.clear()

    def teardown_method(self):
        """Clean up after test."""
        # Clear overrides
        app.dependency_overrides.clear()

    def test_analyze_nonexistent_document(self, test_db):
        """Test analyzing a document that doesn't exist."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        fake_id = "12345678-1234-5678-9abc-123456789012"

        response = client.post(f"/api/v1/pipeline/analyze/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analyze_document_not_ready(self, test_db):
        """Test analyzing a document that's not in completed status."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        # Create a document in pending status in test database
        document = Document(
            filename="test.txt",
            file_type=DocumentType.TEXT,
            file_size=100,
            content="Test content",
            metadata=DocumentMetadata(title="Test Doc"),
            processing_status=ProcessingStatus.PENDING,  # Not completed!
        )
        document_id = repos.document_repo.create(document).id

        response = client.post(f"/api/v1/pipeline/analyze/{document_id}")

        assert response.status_code == 400
        assert "not ready for analysis" in response.json()["detail"]

    def test_successful_analysis_workflow(self, test_db):
        """Test the complete analysis workflow."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # 1. Create a completed document in test database
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

        # The analysis should be completed (since we use mock LLM)
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

    def test_list_analyses(self, test_db):
        """Test listing analysis jobs."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # Create a document and start analysis in test database
        document = Document(
            filename="test_doc.txt",
            file_type=DocumentType.TEXT,
            file_size=75,
            content="Short test document for analysis listing.",
            metadata=DocumentMetadata(title="Test Document"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

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

    def test_list_analyses_with_filter(self, test_db):
        """Test listing analyses with status filter."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # Create a document and start analysis in test database
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

    def test_pipeline_status(self, test_db):
        """Test getting pipeline status."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

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

    def test_analysis_pagination(self, test_db):
        """Test pagination of analysis results."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # Create multiple documents and analyses in test database
        for i in range(3):
            document = Document(
                filename=f"pagination_test_{i}.txt",
                file_type=DocumentType.TEXT,
                file_size=60 + i,
                content=f"Test document {i} for pagination testing.",
                metadata=DocumentMetadata(title=f"Pagination Test {i}"),
                processing_status=ProcessingStatus.COMPLETED,
            )
            document_id = repos.document_repo.create(document).id
            client.post(f"/api/v1/pipeline/analyze/{document_id}")

        # Test pagination
        response = client.get("/api/v1/pipeline/analyses?limit=2&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert len(data["analyses"]) <= 2

    def test_invalid_analysis_id(self, test_db):
        """Test retrieving analysis with invalid ID."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        fake_analysis_id = "99999999-9999-9999-9999-999999999999"

        response = client.get(f"/api/v1/pipeline/analysis/{fake_analysis_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_document_retrieval_error_handling(self, test_db):
        """Test error handling for document retrieval issues."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo

        # This test verifies that the database isolation is working
        # by ensuring no documents exist initially
        fake_id = "11111111-1111-1111-1111-111111111111"

        response = client.post(f"/api/v1/pipeline/analyze/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_database_isolation_verification(self, test_db):
        """Verify that tests are using isolated test database."""
        repos = test_db

        # Create a document with a unique name
        document = Document(
            filename="isolation_verification.txt",
            file_type=DocumentType.TEXT,
            file_size=42,
            content="This document verifies database isolation.",
            metadata=DocumentMetadata(title="Isolation Verification"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # Verify document exists
        retrieved_doc = repos.document_repo.get_by_id(document_id)
        assert retrieved_doc is not None
        assert retrieved_doc.filename == "isolation_verification.txt"

        # Verify we're using the test database
        assert "test.db" in repos.document_repo.db_path
        assert repos.document_repo.db_path != "./data/app.db"  # Not production DB

    def test_requires_review_filter(self, test_db):
        """Test filtering analyses that require human review."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # Create document and analysis in test database
        document = Document(
            filename="review_test.txt",
            file_type=DocumentType.TEXT,
            file_size=90,
            content="Document that might require human review.",
            metadata=DocumentMetadata(title="Review Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # Start analysis
        client.post(f"/api/v1/pipeline/analyze/{document_id}")

        # List analyses requiring review
        response = client.get("/api/v1/pipeline/analyses?requires_review=true")
        assert response.status_code == 200

        # List analyses not requiring review
        response = client.get("/api/v1/pipeline/analyses?requires_review=false")
        assert response.status_code == 200

    def test_document_specific_analyses(self, test_db):
        """Test getting analyses for a specific document."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # Create document in test database
        document = Document(
            filename="specific_doc.txt",
            file_type=DocumentType.TEXT,
            file_size=70,
            content="Document for testing document-specific analyses.",
            metadata=DocumentMetadata(title="Specific Doc Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # Run multiple analyses
        client.post(f"/api/v1/pipeline/analyze/{document_id}")
        client.post(f"/api/v1/pipeline/analyze/{document_id}")

        # Get analyses for this document
        response = client.get(f"/api/v1/pipeline/document/{document_id}/analyses")

        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert "analyses" in data
        assert "total_analyses" in data
        assert data["document_id"] == str(document_id)
        assert data["total_analyses"] >= 2

    def test_entity_caching(self, test_db):
        """Test that entities are cached and reused from previous analyses."""
        repos = test_db

        # Override dependencies to use test repositories
        app.dependency_overrides[get_document_repository] = lambda: repos.document_repo
        app.dependency_overrides[get_analysis_repository] = lambda: repos.analysis_repo
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline(
            analysis_repository=repos.analysis_repo
        )

        # Create a document in test database
        document = Document(
            filename="caching_test.txt",
            file_type=DocumentType.TEXT,
            file_size=80,
            content="Document for testing entity caching functionality.",
            metadata=DocumentMetadata(title="Caching Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = repos.document_repo.create(document).id

        # First analysis - should extract new entities
        response1 = client.post(f"/api/v1/pipeline/analyze/{document_id}")
        assert response1.status_code == 202
        analysis_id1 = response1.json()["analysis_id"]

        # Wait for first analysis to complete
        import time

        time.sleep(1)

        # Get first analysis results
        response1_result = client.get(f"/api/v1/pipeline/analysis/{analysis_id1}")
        assert response1_result.status_code == 200
        analysis1_data = response1_result.json()
        assert analysis1_data["status"] == "completed"

        # Second analysis - should reuse entities from first analysis
        response2 = client.post(f"/api/v1/pipeline/analyze/{document_id}")
        assert response2.status_code == 202
        analysis_id2 = response2.json()["analysis_id"]

        # Wait for second analysis to complete
        time.sleep(1)

        # Get second analysis results
        response2_result = client.get(f"/api/v1/pipeline/analysis/{analysis_id2}")
        assert response2_result.status_code == 200
        analysis2_data = response2_result.json()
        assert analysis2_data["status"] == "completed"

        # Verify that second analysis references entities from first analysis
        result2 = analysis2_data["result"]
        assert "entities_source_analysis_id" in result2
        assert result2["entities_source_analysis_id"] == str(analysis_id1)

        # Third analysis with force_regenerate_entities=True - should extract new entities
        response3 = client.post(
            f"/api/v1/pipeline/analyze/{document_id}?force_regenerate_entities=true"
        )
        assert response3.status_code == 202
        analysis_id3 = response3.json()["analysis_id"]

        # Wait for third analysis to complete
        time.sleep(1)

        # Get third analysis results
        response3_result = client.get(f"/api/v1/pipeline/analysis/{analysis_id3}")
        assert response3_result.status_code == 200
        analysis3_data = response3_result.json()
        assert analysis3_data["status"] == "completed"

        # Verify that third analysis has no entities_source_analysis_id (new entities)
        result3 = analysis3_data["result"]
        assert result3["entities_source_analysis_id"] is None
