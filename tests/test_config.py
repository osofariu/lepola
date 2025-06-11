"""
Tests for the configuration module.

This module contains tests for the Settings class and configuration
management functionality.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.core.config import Settings


class TestSettings:
    """Test suite for the Settings class."""

    def test_settings_initialization(self):
        """Test that Settings can be initialized with default values."""
        settings = Settings()

        assert settings.app_name == "AI Legal & Policy Research Assistant"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.log_level == "INFO"

    def test_settings_with_environment_variables(self):
        """Test that Settings properly loads from environment variables."""
        with patch.dict(
            os.environ, {"DEBUG": "true", "PORT": "9000", "LOG_LEVEL": "DEBUG"}
        ):
            settings = Settings()

            assert settings.debug is True
            assert settings.port == 9000
            assert settings.log_level == "DEBUG"

    def test_cors_settings(self):
        """Test CORS configuration settings."""
        settings = Settings()

        # Test that the raw allowed_origins is a string
        assert isinstance(settings.allowed_origins, str)

        # Test that get_allowed_origins() returns a list
        allowed_origins_list = settings.get_allowed_origins()
        assert isinstance(allowed_origins_list, list)
        assert "http://localhost:3000" in allowed_origins_list
        assert "http://localhost:8080" in allowed_origins_list

    def test_llm_configuration_openai(self):
        """Test OpenAI LLM configuration."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "DEFAULT_LLM_PROVIDER": "openai"}
        ):
            settings = Settings()
            llm_config = settings.get_llm_config()

            assert llm_config["provider"] == "openai"
            assert llm_config["api_key"] == "test-key"
            assert "model" in llm_config

    def test_llm_configuration_missing_key(self):
        """Test that missing API key raises ValueError."""
        settings = Settings()
        settings.default_llm_provider = "ollama"
        settings.openai_api_key = None
        settings.anthropic_api_key = None
        settings.ollama_api_key = None

        with pytest.raises(ValueError, match="Ollama API key not configured"):
            settings.get_llm_config()

    def test_is_development_property(self):
        """Test the is_development property."""
        settings = Settings(debug=True)
        assert settings.is_development is True

        settings = Settings(debug=False)
        assert settings.is_development is False

    def test_is_production_property(self):
        """Test the is_production property."""
        settings = Settings(debug=False)
        assert settings.is_production is True

        settings = Settings(debug=True)
        assert settings.is_production is False

    def test_directory_creation(self):
        """Test that necessary directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change the vector_db_path to use temp directory
            with patch.dict(os.environ, {"VECTOR_DB_PATH": f"{temp_dir}/vectordb"}):
                settings = Settings()

                # Check that the parent directory of vector_db_path exists
                vector_db_parent = os.path.dirname(settings.vector_db_path)
                assert os.path.exists(vector_db_parent)

    def test_file_processing_settings(self):
        """Test file processing configuration."""
        settings = Settings()

        assert settings.max_file_size > 0
        supported_types = settings.get_supported_file_types()
        assert isinstance(supported_types, list)
        assert "pdf" in supported_types
        assert "txt" in supported_types

    def test_ai_pipeline_settings(self):
        """Test AI pipeline configuration."""
        settings = Settings()

        assert 0 <= settings.confidence_threshold <= 1
        assert settings.max_context_length > 0
        assert isinstance(settings.enable_fact_checking, bool)
