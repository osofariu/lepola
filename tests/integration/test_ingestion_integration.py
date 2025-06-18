import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.main import app

pytestmark = pytest.mark.usefixtures("mock_repositories")

client = TestClient(app)


@pytest.mark.asyncio
@patch("src.ingestion.embedding.process_document_embeddings", new_callable=AsyncMock)
def test_upload_document_with_async_embedding(mock_embed, caplog):
    file_content = b"Test paragraph one.\n\nTest paragraph two."
    files = {"file": ("test.txt", file_content)}
    with caplog.at_level("INFO"):
        response = client.post(
            "/api/v1/ingestion/upload?async_embedding=true", files=files
        )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "completed"
    assert any(
        "Started async embedding/indexing job" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
@patch("src.ingestion.embedding.process_document_embeddings", new_callable=AsyncMock)
def test_upload_document_without_async_embedding(mock_embed, caplog):
    file_content = b"Test paragraph one.\n\nTest paragraph two."
    files = {"file": ("test.txt", file_content)}
    with caplog.at_level("INFO"):
        response = client.post(
            "/api/v1/ingestion/upload?async_embedding=false", files=files
        )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "completed"
    assert not any(
        "Started async embedding/indexing job" in record.message
        for record in caplog.records
    )
