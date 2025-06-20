"""
Configuration management for AI Legal & Policy Research Assistant.

This module handles all application configuration using Pydantic BaseSettings
for environment variable management and validation.
"""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support.

    This class manages all configuration settings for the application,
    automatically loading values from environment variables with sensible defaults.
    """

    # Application settings
    app_name: str = Field(
        default="AI Legal & Policy Research Assistant", description="Application name"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    log_level: str = Field(default="INFO", description="Logging level")

    # CORS settings
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Allowed CORS origins (comma-separated)",
    )

    # LLM API settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    ollama_api_key: Optional[str] = Field(default=None, description="Ollama API key")

    default_llm_provider: str = Field(
        default="ollama", description="Default LLM provider"
    )

    # Vector database settings
    vector_db_type: str = Field(
        default="faiss", description="Vector database type (faiss, chroma)"
    )
    vector_db_path: str = Field(
        default="./data/vectordb", description="Vector database storage path"
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002", description="Embedding model name"
    )

    # Document processing settings
    max_file_size: int = Field(
        default=50 * 1024 * 1024, description="Maximum file size in bytes (50MB)"
    )
    supported_file_types: str = Field(
        default="pdf,txt,docx,html",
        description="Supported file types for ingestion (comma-separated)",
    )

    # AI Pipeline settings
    confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence threshold for AI outputs"
    )
    max_context_length: int = Field(
        default=4000, description="Maximum context length for LLM"
    )
    enable_fact_checking: bool = Field(
        default=True, description="Enable fact-checking pipeline"
    )

    # AWS settings (for cloud deployment)
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_s3_bucket: Optional[str] = Field(
        default=None, description="S3 bucket for document storage"
    )
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key ID"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret access key"
    )

    # Security settings
    secret_key: str = Field(
        default="your-secret-key-change-this", description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="JWT token expiration in minutes"
    )

    # Database settings (for metadata storage)
    database_url: str = Field(
        default="sqlite:///./data/app.db",
        description="Database URL for metadata storage",
    )

    # Rate limiting
    rate_limit_requests: int = Field(
        default=100, description="Requests per minute rate limit"
    )
    rate_limit_period: int = Field(
        default=60, description="Rate limit period in seconds"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            os.path.dirname(self.vector_db_path),
            "./data",
            "./outputs",
            "./logs",
        ]

        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug

    def get_allowed_origins(self) -> List[str]:
        """Get allowed origins as a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    def get_supported_file_types(self) -> List[str]:
        """Get supported file types as a list."""
        return [file_type.strip() for file_type in self.supported_file_types.split(",")]

    def get_llm_config(self) -> dict:
        """Get LLM configuration based on the default provider.

        Returns:
            dict: LLM configuration dictionary.

        Raises:
            ValueError: If no API key is configured for the selected provider.
        """
        if self.default_llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")

            # Check for test/mock API keys
            if self.openai_api_key.startswith("sk-test-"):
                return {
                    "provider": "openai",
                    "api_key": self.openai_api_key,
                    "model": "gpt-4-turbo-preview",
                    "mock": True,
                }

            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": "gpt-4-turbo-preview",
            }
        elif self.default_llm_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")

            # Check for test/mock API keys
            if self.anthropic_api_key.startswith("sk-ant-test-"):
                return {
                    "provider": "anthropic",
                    "api_key": self.anthropic_api_key,
                    "model": "claude-3-sonnet-20240229",
                    "mock": True,
                }

            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": "claude-3-sonnet-20240229",
            }

        elif self.default_llm_provider == "ollama":
            if not self.ollama_api_key:
                raise ValueError("Ollama API key not configured")

            # Check for test/mock API keys
            if self.ollama_api_key.startswith("sk-oll-test-"):
                return {
                    "provider": "ollama",
                    "api_key": self.ollama_api_key,
                    # "model": "llama3.1:8b",
                    "model": "gemma3:27b",
                    "mock": True,
                }

            return {
                "provider": "ollama",
                "api_key": self.ollama_api_key,
                "model": "gemma3:27b",
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.default_llm_provider}")


# Global settings instance
settings = Settings()
