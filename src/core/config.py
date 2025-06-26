"""
Configuration management for AI Legal & Policy Research Assistant.

This module handles all application configuration using Pydantic BaseSettings
for environment variable management and validation.
"""

import os
from pathlib import Path
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

    # Embedding API settings
    default_embedding_provider: str = Field(
        default="ollama", description="Default embedding provider"
    )
    embedding_model: str = Field(
        default="snowflake-arctic-embed:335m", description="Embedding model name"
    )
    embedding_api_key: Optional[str] = Field(
        default=None, description="Embedding API key (for OpenAI, Cohere, etc.)"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )

    # Vector database settings
    vector_db_type: str = Field(
        default="faiss", description="Vector database type (faiss, chroma)"
    )
    vector_db_path: str = Field(
        default="./data/vectordb", description="Vector database storage path"
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
        default=0.6, description="Minimum confidence threshold for AI outputs"
    )
    max_context_length: int = Field(
        default=6000, description="Maximum context length for LLM"
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
    database_timeout: float = Field(
        default=30.0, description="Database connection timeout in seconds"
    )

    # HTTP timeout settings for long-running operations
    http_timeout: float = Field(
        default=600.0, description="HTTP request timeout in seconds (10 minutes)"
    )
    http_connect_timeout: float = Field(
        default=60.0, description="HTTP connection timeout in seconds"
    )
    http_read_timeout: float = Field(
        default=300.0, description="HTTP read timeout in seconds (5 minutes)"
    )

    # Rate limiting
    rate_limit_requests: int = Field(
        default=100, description="Requests per minute rate limit"
    )
    rate_limit_period: int = Field(
        default=60, description="Rate limit period in seconds"
    )

    # JSON Parsing settings
    json_strict_mode: bool = Field(
        default=False,
        description="Enable strict JSON parsing mode (fails fast on errors)",
    )
    json_enable_robust_parsing: bool = Field(
        default=True, description="Enable robust JSON parsing for malformed data"
    )
    json_log_parsing_errors: bool = Field(
        default=True, description="Log JSON parsing errors for monitoring"
    )
    json_max_recovery_attempts: int = Field(
        default=3, description="Maximum attempts to recover from JSON parsing errors"
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

    def _get_project_root(self) -> Path:
        """Get the project root directory.

        Returns:
            Path: Path to the project root directory.
        """
        # Start from the current file's directory and walk up to find the project root
        current_file = Path(__file__)
        current_dir = current_file.parent

        # Walk up the directory tree to find the project root
        # Look for pyproject.toml or README.md as indicators of the project root
        while current_dir.parent != current_dir:  # Stop at filesystem root
            if (current_dir / "pyproject.toml").exists() or (
                current_dir / "README.md"
            ).exists():
                return current_dir
            current_dir = current_dir.parent

        # Fallback: if we can't find the project root, use the current working directory
        return Path.cwd()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist.

        All directories are created relative to the project root to ensure
        consistent behavior regardless of the current working directory.
        """
        project_root = self._get_project_root()

        # Convert relative paths to absolute paths relative to project root
        directories = [
            project_root / "data",
            project_root / "logs",
        ]

        # Handle vector_db_path specially - it might be relative or absolute
        if os.path.isabs(self.vector_db_path):
            # If it's already absolute, use it as is
            vector_db_dir = Path(self.vector_db_path)
        else:
            # If it's relative, make it relative to project root
            vector_db_dir = project_root / self.vector_db_path.lstrip("./")

        directories.append(vector_db_dir)

        for directory in directories:
            if directory and not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)

        # Update the vector_db_path to be absolute for consistency
        self.vector_db_path = str(vector_db_dir)

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
                    "model": "llama3.1:8b",
                    # "model": "gemma3:1b",
                    "mock": True,
                }

            # Use a larger model for better performance
            return {
                "provider": "ollama",
                "api_key": self.ollama_api_key,
                "model": "llama3.1:8b",  # Better than gemma3:4b for analysis
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.default_llm_provider}")

    def get_embedding_config(self) -> dict:
        """Get embedding configuration based on the default provider.

        Returns:
            dict: Embedding configuration dictionary.

        Raises:
            ValueError: If no API key is configured for the selected provider.
        """
        if self.default_embedding_provider == "openai":
            if not self.embedding_api_key:
                raise ValueError("OpenAI API key not configured for embeddings")

            # Check for test/mock API keys
            if self.embedding_api_key.startswith("sk-test-"):
                return {
                    "provider": "openai",
                    "api_key": self.embedding_api_key,
                    "model": "text-embedding-ada-002",
                    "mock": True,
                }

            return {
                "provider": "openai",
                "api_key": self.embedding_api_key,
                "model": "text-embedding-ada-002",
            }
        elif self.default_embedding_provider == "bedrock":
            if not self.aws_access_key_id or not self.aws_secret_access_key:
                raise ValueError(
                    "AWS credentials not configured for Bedrock embeddings"
                )

            return {
                "provider": "bedrock",
                "model": "amazon.titan-embed-text-v1",
                "region": self.aws_region,
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
            }
        elif self.default_embedding_provider == "cohere":
            if not self.embedding_api_key:
                raise ValueError("Cohere API key not configured for embeddings")

            return {
                "provider": "cohere",
                "api_key": self.embedding_api_key,
                "model": "embed-english-v3.0",
            }
        elif self.default_embedding_provider == "ollama":
            # Ollama doesn't require API key for local usage
            return {
                "provider": "ollama",
                "model": self.embedding_model,
                "base_url": self.ollama_base_url,
            }
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.default_embedding_provider}"
            )

    def get_json_parsing_config(self) -> dict:
        """Get JSON parsing configuration based on environment.

        Returns:
            dict: JSON parsing configuration with appropriate settings for the environment.
        """
        # In production, default to strict mode unless explicitly disabled
        # In development, default to lenient mode for better debugging
        if self.is_production:
            # Production defaults: strict mode for performance and data quality
            strict_mode = (
                self.json_strict_mode if hasattr(self, "json_strict_mode") else True
            )
            enable_robust = (
                self.json_enable_robust_parsing
                if hasattr(self, "json_enable_robust_parsing")
                else False
            )
        else:
            # Development defaults: lenient mode for debugging and recovery
            strict_mode = (
                self.json_strict_mode if hasattr(self, "json_strict_mode") else False
            )
            enable_robust = (
                self.json_enable_robust_parsing
                if hasattr(self, "json_enable_robust_parsing")
                else True
            )

        return {
            "strict": strict_mode,
            "enable_robust_parsing": enable_robust,
            "log_errors": (
                self.json_log_parsing_errors
                if hasattr(self, "json_log_parsing_errors")
                else True
            ),
            "max_recovery_attempts": (
                self.json_max_recovery_attempts
                if hasattr(self, "json_max_recovery_attempts")
                else 3
            ),
            "environment": "production" if self.is_production else "development",
        }


# Global settings instance
settings = Settings()
