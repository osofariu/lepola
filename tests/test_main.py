"""
Tests for the main FastAPI application.

This module contains tests for the FastAPI application setup,
routes, and basic functionality.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import create_app


class TestMainApplication:
    """Test suite for the main FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI application."""
        app = create_app()
        return TestClient(app)

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "legal-policy-assistant"

    def test_app_creation(self):
        """Test that the app can be created without errors."""
        app = create_app()

        assert app.title == "AI Legal & Policy Research Assistant"
        assert "AI-powered research assistant" in app.description
        assert app.version == "0.1.0"

    def test_docs_endpoints_available(self, client):
        """Test that API documentation endpoints are available."""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_ingestion_status_endpoint(self, client):
        """Test the ingestion service status endpoint."""
        response = client.get("/api/v1/ingestion/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "document-ingestion"
        assert "supported_types" in data
        assert "max_file_size" in data

    def test_pipeline_status_endpoint(self, client):
        """Test the AI pipeline service status endpoint."""
        response = client.get("/api/v1/pipeline/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai-pipeline"

    def test_querying_status_endpoint(self, client):
        """Test the querying service status endpoint."""
        response = client.get("/api/v1/query/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "interactive-querying"

    def test_outputs_status_endpoint(self, client):
        """Test the outputs service status endpoint."""
        response = client.get("/api/v1/outputs/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "output-generation"

    def test_cors_headers(self, client):
        """Test that CORS headers are properly configured."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS preflight should not return an error
        assert response.status_code in [200, 204]
