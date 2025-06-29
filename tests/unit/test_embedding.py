import pytest
from src.ingestion import embedding
import numpy as np
import tempfile
import os
from unittest.mock import patch, AsyncMock


def test_chunk_paragraphs_basic():
    text = "Para1.\n\nPara2.\n\nPara3."
    chunks = embedding.chunk_paragraphs(text)
    assert len(chunks) == 3
    assert chunks[0][0] == "Para1."
    assert chunks[1][0] == "Para2."
    assert chunks[2][0] == "Para3."


def test_chunk_paragraphs_empty():
    text = "\n\n\n"
    chunks = embedding.chunk_paragraphs(text)
    assert chunks == []


def test_chunk_paragraphs_whitespace():
    text = "  Para1.  \n\n  Para2.  "
    chunks = embedding.chunk_paragraphs(text)
    assert len(chunks) == 2
    assert chunks[0][0] == "Para1."
    assert chunks[1][0] == "Para2."


def test_save_and_load_faiss_index(tmp_path):
    import faiss

    arr = np.random.rand(5, 8).astype("float32")
    index = faiss.IndexFlatL2(8)
    index.add(arr)
    path = tmp_path / "test.index"
    embedding.save_faiss_index(index, str(path))
    loaded = embedding.load_faiss_index(str(path))
    assert loaded is not None
    assert loaded.ntotal == 5


def test_create_or_update_faiss_index(tmp_path):
    import faiss

    arr = np.random.rand(3, 4).astype("float32")
    path = tmp_path / "test2.index"
    index = embedding.create_or_update_faiss_index(arr.tolist(), str(path))
    assert index.ntotal == 3
    arr2 = np.random.rand(2, 4).astype("float32")
    index2 = embedding.create_or_update_faiss_index(arr2.tolist(), str(path))
    assert index2.ntotal == 5


@pytest.mark.asyncio
async def test_embed_with_ollama_mock():
    """Test embedding with mocked embedding service."""
    paragraphs = ["A", "B"]
    dummy_vectors = [[0.1, 0.2], [0.3, 0.4]]

    # Mock the embedding service's embed_texts method
    with patch("src.ingestion.embedding.embedding_service.embed_texts") as mock_embed:
        mock_embed.return_value = dummy_vectors

        result = await embedding.embed_texts(paragraphs)

        # Verify the mock was called correctly
        mock_embed.assert_called_once_with(paragraphs)
        assert result == dummy_vectors
