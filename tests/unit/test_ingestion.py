"""
Tests for the document ingestion service.

This module contains tests for document ingestion functionality
including file upload, URL processing, and content extraction.
"""

import io
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.core.models import DocumentType, ProcessingStatus, Document, DocumentMetadata
from src.ingestion.service import DocumentIngestionError, DocumentIngestionService
from datetime import datetime


class TestDocumentIngestionService:
    """Test suite for the DocumentIngestionService class."""

    @pytest.fixture
    def ingestion_service(self):
        """Create a DocumentIngestionService instance for testing."""
        return DocumentIngestionService()

    def test_service_initialization(self, ingestion_service):
        """Test that the service initializes correctly."""
        assert ingestion_service.max_file_size > 0
        assert isinstance(ingestion_service.supported_types, list)
        assert "pdf" in ingestion_service.supported_types
        assert "txt" in ingestion_service.supported_types

    def test_detect_file_type_pdf(self, ingestion_service):
        """Test file type detection for PDF files."""
        file_type = ingestion_service._detect_file_type("document.pdf")
        assert file_type == DocumentType.PDF

    def test_detect_file_type_text(self, ingestion_service):
        """Test file type detection for text files."""
        file_type = ingestion_service._detect_file_type("document.txt")
        assert file_type == DocumentType.TEXT

    def test_detect_file_type_html(self, ingestion_service):
        """Test file type detection for HTML files."""
        file_type = ingestion_service._detect_file_type("page.html")
        assert file_type == DocumentType.HTML

        file_type = ingestion_service._detect_file_type("page.htm")
        assert file_type == DocumentType.HTML

    def test_detect_file_type_unknown(self, ingestion_service):
        """Test file type detection for unknown files defaults to text."""
        file_type = ingestion_service._detect_file_type("document.unknown")
        assert file_type == DocumentType.TEXT

    @pytest.mark.asyncio
    async def test_ingest_text_file_success(self, ingestion_service):
        """Test successful text file ingestion."""
        content = "This is a sample legal document for testing purposes."
        file_data = io.BytesIO(content.encode("utf-8"))

        # Create a mock document that would be returned by the repository
        mock_document = Document(
            filename="test.txt",
            file_type=DocumentType.TEXT,
            file_size=len(content),
            content=content,
            metadata=DocumentMetadata(word_count=len(content.split())),
            processing_status=ProcessingStatus.COMPLETED,
            checksum="mock_checksum",
        )

        # Create a mock repository
        mock_repo = MagicMock()
        mock_repo.create.return_value = mock_document

        # Patch the document repository
        with patch("src.ingestion.service.document_repository", mock_repo):
            document = await ingestion_service.ingest_file(
                file_data=file_data, filename="test.txt", file_size=len(content)
            )

            # Verify the document was created with correct data
            assert document.filename == "test.txt"
            assert document.file_type == DocumentType.TEXT
            assert document.content == content
            assert document.processing_status == ProcessingStatus.COMPLETED
            assert document.checksum == "mock_checksum"
            assert document.metadata.word_count > 0

            # Verify the repository was called
            mock_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_file_too_large(self, ingestion_service):
        """Test that files exceeding max size are rejected."""
        content = "x" * (ingestion_service.max_file_size + 1)
        file_data = io.BytesIO(content.encode("utf-8"))

        with pytest.raises(
            DocumentIngestionError, match="exceeds maximum allowed size"
        ):
            await ingestion_service.ingest_file(
                file_data=file_data, filename="large.txt", file_size=len(content)
            )

    @pytest.mark.asyncio
    async def test_ingest_unsupported_file_type(self, ingestion_service):
        """Test that unsupported file types are rejected."""
        content = "test content"
        file_data = io.BytesIO(content.encode("utf-8"))

        # Mock supported types to exclude 'pdf' - use a PDF filename
        with patch.object(ingestion_service, "supported_types", ["txt", "html"]):
            with pytest.raises(DocumentIngestionError, match="Unsupported file type"):
                await ingestion_service.ingest_file(
                    file_data=file_data, filename="document.pdf", file_size=len(content)
                )

    def test_extract_text_content(self, ingestion_service):
        """Test text content extraction."""
        content = "Sample legal text for testing."
        file_data = io.BytesIO(content.encode("utf-8"))

        extracted_content, metadata = ingestion_service._extract_text_content(file_data)

        assert extracted_content == content
        assert metadata.word_count == len(content.split())
        assert metadata.page_count == 1

    def test_extract_html_content(self, ingestion_service):
        """Test HTML content extraction."""
        html_content = """
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Legal Document</h1>
            <p>This is a test paragraph.</p>
            <script>console.log('should be removed');</script>
        </body>
        </html>
        """

        extracted_content, metadata = ingestion_service._extract_html_content(
            html_content
        )

        assert "Legal Document" in extracted_content
        assert "test paragraph" in extracted_content
        assert "should be removed" not in extracted_content  # Script should be removed
        assert metadata.title == "Test Document"
        assert metadata.word_count > 0

    @pytest.mark.asyncio
    async def test_ingest_url_invalid(self, ingestion_service):
        """Test that invalid URLs are rejected."""
        with pytest.raises(DocumentIngestionError, match="Invalid URL"):
            await ingestion_service.ingest_url("not-a-url")

    @pytest.mark.asyncio
    async def test_get_document_by_id_placeholder(self, ingestion_service):
        """Test the placeholder get_document_by_id method."""
        from uuid import uuid4

        document_id = uuid4()
        result = await ingestion_service.get_document_by_id(document_id)

        # Placeholder implementation returns None
        assert result is None


@pytest.mark.asyncio
async def test_ingest_url_with_mock():
    """Test URL ingestion with mocked HTTP response."""
    service = DocumentIngestionService()

    mock_response_content = b"<html><body>Test content</body></html>"

    # Create a mock document that would be returned by the repository
    mock_document = Document(
        filename="test.html",
        file_type=DocumentType.URL,
        file_size=len(mock_response_content),
        content="Test content",
        metadata=DocumentMetadata(word_count=2),
        processing_status=ProcessingStatus.COMPLETED,
        source_url="https://example.com/test.html",
        checksum="mock_checksum",
    )

    # Create a mock repository
    mock_repo = MagicMock()
    mock_repo.create.return_value = mock_document

    with patch("aiohttp.ClientSession.get") as mock_get, patch(
        "src.ingestion.service.document_repository", mock_repo
    ):
        # Mock the async context manager
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.read = AsyncMock(return_value=mock_response_content)

        mock_get.return_value.__aenter__.return_value = mock_response

        document = await service.ingest_url("https://example.com/test.html")

        assert document.source_url == "https://example.com/test.html"
        assert document.file_type == DocumentType.URL
        assert "Test content" in document.content
        assert document.checksum == "mock_checksum"

        # Verify the repository was called
        mock_repo.create.assert_called_once()


def test_pdf_date_parsing():
    """Test PDF date parsing functionality."""
    service = DocumentIngestionService()

    # Test various PDF date formats
    test_cases = [
        (
            "D:20250525092926-04'00'",
            datetime(2025, 5, 25, 13, 29, 26),
        ),  # UTC conversion
        ("D:20240101120000+05'30'", datetime(2024, 1, 1, 6, 30, 0)),  # UTC conversion
        ("D:20230615143000", datetime(2023, 6, 15, 14, 30, 0)),  # No timezone
        (None, None),  # None input
        ("", None),  # Empty string
        ("invalid_date", None),  # Invalid format
    ]

    for pdf_date, expected in test_cases:
        result = service._parse_pdf_date(pdf_date)
        if expected is None:
            assert result is None, f"Expected None for {pdf_date}, got {result}"
        else:
            assert result is not None, f"Expected datetime for {pdf_date}, got None"
            assert (
                abs((result - expected).total_seconds()) < 60
            ), f"Date mismatch for {pdf_date}: expected {expected}, got {result}"
