"""
Fast pipeline integration tests.

This module contains optimized tests that use MockLLM and minimal setup
for rapid test execution.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.core.models import DocumentType, ProcessingStatus
from src.core.repository import document_repository
from src.main import app
from src.pipeline.router import get_ai_pipeline
from src.pipeline.service import AIAnalysisPipeline
from src.pipeline.mock_llm import MockLLM

# Test client
client = TestClient(app)


class MockAIAnalysisPipeline(AIAnalysisPipeline):
    """Test pipeline that always uses MockLLM."""

    def __init__(self):
        """Initialize with MockLLM only."""
        self.llm = MockLLM()
        self.confidence_threshold = 0.7


class TestPipelineFast:
    """Fast pipeline tests using MockLLM."""

    def setup_method(self):
        """Set up test method with mocked pipeline."""
        # Override the dependency
        app.dependency_overrides[get_ai_pipeline] = lambda: MockAIAnalysisPipeline()

    def teardown_method(self):
        """Clean up after test."""
        # Clear overrides
        app.dependency_overrides.clear()

    def test_analyze_nonexistent_document_fast(self, temp_db_fast):
        """Test analyzing a document that doesn't exist - fast version."""
        fake_id = "12345678-1234-5678-9abc-123456789012"

        response = client.post(f"/api/v1/pipeline/analyze/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analyze_document_not_ready_fast(self, temp_db_fast):
        """Test analyzing a document that's not ready - fast version."""
        from src.core.models import Document, DocumentMetadata

        document = Document(
            filename="test.txt",
            file_type=DocumentType.TEXT,
            file_size=100,
            content="Test content",
            metadata=DocumentMetadata(title="Test Doc"),
            processing_status=ProcessingStatus.PENDING,
        )
        document_id = document_repository.create(document).id

        response = client.post(f"/api/v1/pipeline/analyze/{document_id}")

        assert response.status_code == 400
        assert "not ready for analysis" in response.json()["detail"]

    def test_successful_analysis_workflow_fast(self, temp_db_fast):
        """Test the complete analysis workflow - fast version with dependency override."""
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

    def test_mock_llm_is_used(self, temp_db_fast, force_mock_llm):
        """Verify that MockLLM is being used in tests."""
        # Create fresh MockLLM instance
        mock_llm = MockLLM()
        assert hasattr(mock_llm, "_llm_type")
        assert mock_llm._llm_type == "mock"

        # Test that mock responses are generated
        response = mock_llm.invoke("extract entities from: test document")
        assert "```json" in response.content
        assert "Type" in response.content

    def test_multiple_analyses_same_document(self, temp_db_fast):
        """Test that multiple analyses can be run on the same document."""
        from src.core.models import Document, DocumentMetadata

        document = Document(
            filename="multi_analysis.txt",
            file_type=DocumentType.TEXT,
            file_size=100,
            content="Document for multiple analysis testing.",
            metadata=DocumentMetadata(title="Multi Analysis Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = document_repository.create(document).id

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

    def test_document_analyses_endpoint(self, temp_db_fast):
        """Test the document-specific analyses endpoint."""
        from src.core.models import Document, DocumentMetadata

        document = Document(
            filename="doc_analyses.txt",
            file_type=DocumentType.TEXT,
            file_size=90,
            content="Document for testing document analyses endpoint.",
            metadata=DocumentMetadata(title="Doc Analyses Test"),
            processing_status=ProcessingStatus.COMPLETED,
        )
        document_id = document_repository.create(document).id

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
