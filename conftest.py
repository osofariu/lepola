"""
Global pytest configuration for the AI Legal & Policy Research Assistant.

This module sets up test fixtures and configurations that are available
across all test modules.
"""

import os
import pytest
import tempfile
import sqlite3
import warnings
from pathlib import Path
from unittest.mock import patch

from src.core.database import Database
from src.core.repository import DocumentRepository, AnalysisRepository

# Suppress the FastAPI TestClient deprecation warning
warnings.filterwarnings(
    "ignore",
    message="The 'app' shortcut is now deprecated.*",
    category=DeprecationWarning,
)

# Suppress the httpx deprecation warning about 'app' shortcut
warnings.filterwarnings(
    "ignore",
    message=".*app.*shortcut.*deprecated.*",
    category=DeprecationWarning,
)

# Alternative approach: filter by module
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="httpx",
)


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


class TestRepositories:
    """Container for test-specific repository instances."""

    def __init__(self, db_path: str):
        """Initialize repositories with test database path."""
        self.document_repo = DocumentRepository()
        self.analysis_repo = AnalysisRepository()

        # Override paths to use test database
        self.document_repo.db_path = db_path
        self.analysis_repo.db_path = db_path


@pytest.fixture
def test_db():
    """Create isolated test database with repository instances.

    This fixture creates a temporary database and returns repository
    instances that use the test database instead of production.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = str(Path(temp_dir) / "test.db")

        # Initialize database with tables
        with sqlite3.connect(db_path) as db:
            db_instance = Database(db_path)
            db_instance._create_tables_sync(db)
            db.commit()

        # Return test repositories that use the test database
        yield TestRepositories(db_path)


@pytest.fixture
def test_db_fast():
    """Fast test database fixture for unit tests.

    Similar to test_db but optimized for speed.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = str(Path(temp_dir) / "test_fast.db")

        # Initialize database
        with sqlite3.connect(db_path) as db:
            db_instance = Database(db_path)
            db_instance._create_tables_sync(db)
            db.commit()

        # Return test repositories
        yield TestRepositories(db_path)


@pytest.fixture
def mock_repositories(test_db_fast):
    """Fixture that patches repository imports with test instances.

    This ensures that FastAPI routes and any code importing repositories
    will use the test database instead of production.
    """
    repos = test_db_fast

    # Patch repository imports in ALL locations where they're used
    with patch("src.core.repository.document_repository", repos.document_repo), patch(
        "src.core.repository.analysis_repository", repos.analysis_repo
    ), patch("src.ingestion.router.document_repository", repos.document_repo):
        yield repos


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
