"""
Global pytest configuration for the AI Legal & Policy Research Assistant.

This module sets up test fixtures and configurations that are available
across all test modules.
"""

import os
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def force_mock_llm():
    """Force the use of MockLLM in all tests by setting test API keys."""
    # Store original values
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    original_ollama_key = os.environ.get("OLLAMA_API_KEY")
    original_provider = os.environ.get("DEFAULT_LLM_PROVIDER")

    # Set test values that trigger MockLLM
    os.environ["OPENAI_API_KEY"] = "sk-test-mock-key-for-testing"
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-mock-key-for-testing"
    os.environ["OLLAMA_API_KEY"] = "sk-oll-test-mock-key-for-testing"
    os.environ["DEFAULT_LLM_PROVIDER"] = "openai"  # Use openai with test key

    # Clear any cached settings and force reload
    import sys

    modules_to_reload = [
        "src.core.config",
        "src.pipeline.service",
    ]

    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Now import fresh settings
    from src.core.config import Settings

    test_settings = Settings()

    # Patch the global settings instance
    with patch("src.core.config.settings", test_settings):
        yield test_settings

    # Restore original values
    if original_openai_key is not None:
        os.environ["OPENAI_API_KEY"] = original_openai_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)

    if original_anthropic_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = original_anthropic_key
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)

    if original_ollama_key is not None:
        os.environ["OLLAMA_API_KEY"] = original_ollama_key
    else:
        os.environ.pop("OLLAMA_API_KEY", None)

    if original_provider is not None:
        os.environ["DEFAULT_LLM_PROVIDER"] = original_provider
    else:
        os.environ.pop("DEFAULT_LLM_PROVIDER", None)


@pytest.fixture
def mock_llm_config():
    """Direct fixture for MockLLM configuration."""
    return {
        "provider": "openai",
        "api_key": "sk-test-mock-key-for-testing",
        "model": "gpt-4-turbo-preview",
        "mock": True,
    }


@pytest.fixture
def fast_analysis_config():
    """Fixture for tests that need fast analysis configuration."""
    fast_config = {
        "max_context_length": 1000,  # Reduced from 4000
        "confidence_threshold": 0.5,  # Reduced from 0.7 for faster completion
        "enable_fact_checking": False,  # Disable for speed
    }
    return fast_config


@pytest.fixture
def temp_db_fast():
    """Fast temporary database fixture with minimal setup."""
    import tempfile
    import sqlite3
    from pathlib import Path
    from src.core.database import Database
    from src.core.repository import document_repository, analysis_repository

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_fast.db"

        # Initialize database
        with sqlite3.connect(str(db_path)) as db:
            db_instance = Database(str(db_path))
            db_instance._create_tables_sync(db)
            db.commit()

        # Override repository paths
        original_doc_path = document_repository.db_path
        original_analysis_path = analysis_repository.db_path

        document_repository.db_path = str(db_path)
        analysis_repository.db_path = str(db_path)

        yield Database(str(db_path))

        # Restore paths
        document_repository.db_path = original_doc_path
        analysis_repository.db_path = original_analysis_path
