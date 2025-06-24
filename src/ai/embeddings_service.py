"""
Embedding service abstraction for multiple providers.

This module provides a unified interface for embedding operations,
supporting multiple providers like Ollama, OpenAI, AWS Bedrock, etc.
"""

import asyncio
from typing import List, Optional
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

from src.core.config import settings
from src.core.logging import LoggingMixin, debug_log


class EmbeddingServiceError(Exception):
    """Custom exception for embedding service errors."""

    pass


class BaseEmbeddingService(LoggingMixin):
    """Base class for embedding services with provider abstraction."""

    def __init__(self):
        """Initialize the embedding service."""
        self.embedder = self._initialize_embedder()

    def _initialize_embedder(self) -> Embeddings:
        """Initialize the embedding provider based on configuration.

        Returns:
            Embeddings: Initialized embedding provider instance.

        Raises:
            EmbeddingServiceError: If initialization fails.
        """
        try:
            embedding_config = settings.get_embedding_config()

            # Check if we should use a mock embedder for testing
            if embedding_config.get("mock", False) or (
                embedding_config.get("api_key", "").startswith(
                    ("sk-test-", "sk-ant-test-")
                )
            ):
                from src.ai.mock_embeddings import MockEmbeddings

                debug_log("Using MockEmbeddings for testing")
                return MockEmbeddings()

            provider = embedding_config.get("provider", "ollama")
            model = embedding_config.get("model", "snowflake-arctic-embed:335m")
            api_key = embedding_config.get("api_key", "")

            if provider == "ollama":
                debug_log("Initializing OllamaEmbeddings", model=model)
                return OllamaEmbeddings(
                    model=model,
                    base_url=embedding_config.get("base_url", "http://localhost:11434"),
                )
            elif provider == "openai":
                debug_log("Initializing OpenAIEmbeddings", model=model)
                return OpenAIEmbeddings(
                    model=model,
                    openai_api_key=api_key,
                )
            elif provider == "bedrock":
                debug_log("Initializing BedrockEmbeddings", model=model)
                return BedrockEmbeddings(
                    model_id=model,
                    region_name=embedding_config.get("region", "us-east-1"),
                    aws_access_key_id=embedding_config.get("aws_access_key_id"),
                    aws_secret_access_key=embedding_config.get("aws_secret_access_key"),
                )
            elif provider == "cohere":
                debug_log("Initializing CohereEmbeddings", model=model)
                from langchain_community.embeddings import CohereEmbeddings

                return CohereEmbeddings(
                    model=model,
                    cohere_api_key=api_key,
                )
            else:
                raise EmbeddingServiceError(
                    f"Unsupported embedding provider: {provider}"
                )

        except Exception as e:
            raise EmbeddingServiceError(f"Failed to initialize embedder: {str(e)}")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingServiceError: If embedding fails.
        """
        try:
            # Use asyncio to run the sync embedder in a thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.embedder.embed_documents, texts
            )

            self.logger.info(
                "Texts embedded successfully",
                text_count=len(texts),
                embedding_dimension=len(embeddings[0]) if embeddings else 0,
            )

            return embeddings

        except Exception as e:
            self.logger.error(
                "Embedding failed",
                text_count=len(texts),
                error=str(e),
                exc_info=True,
            )
            raise EmbeddingServiceError(f"Embedding failed: {str(e)}")

    async def embed_single_text(self, text: str) -> List[float]:
        """Embed a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingServiceError: If embedding fails.
        """
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.embedder.embed_query, text
            )

            self.logger.debug(
                "Single text embedded successfully",
                text_length=len(text),
                embedding_dimension=len(embedding),
            )

            return embedding

        except Exception as e:
            self.logger.error(
                "Single text embedding failed",
                text_length=len(text),
                error=str(e),
                exc_info=True,
            )
            raise EmbeddingServiceError(f"Single text embedding failed: {str(e)}")


# Global embedding service instance
embedding_service = BaseEmbeddingService()
