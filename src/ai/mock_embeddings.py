"""
Mock embeddings for testing purposes.

This module provides a mock embedding service that returns
deterministic fake embeddings for testing.
"""

import random
from typing import List

from langchain_core.embeddings import Embeddings


class MockEmbeddings(Embeddings):
    """Mock embedding service for testing."""

    def __init__(self, embedding_dimension: int = 384):
        """Initialize mock embeddings.

        Args:
            embedding_dimension: Dimension of the mock embeddings.
        """
        self.embedding_dimension = embedding_dimension
        # Use a fixed seed for deterministic results
        self.random = random.Random(42)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            List of mock embedding vectors.
        """
        embeddings = []
        for text in texts:
            # Generate deterministic embedding based on text content
            self.random.seed(hash(text) % 1000000)
            embedding = [
                self.random.uniform(-1, 1) for _ in range(self.embedding_dimension)
            ]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate mock embedding for a single text.

        Args:
            text: Text string.

        Returns:
            Mock embedding vector.
        """
        return self.embed_documents([text])[0]
