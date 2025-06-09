"""
Integration test for PDF date parsing functionality.

This test demonstrates how the PDF date parsing works in a realistic scenario.
"""

import io
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import pypdf

from src.ingestion.service import DocumentIngestionService


@pytest.mark.asyncio
async def test_pdf_metadata_extraction_with_dates():
    """Test PDF metadata extraction including date parsing."""
    service = DocumentIngestionService()

    # Mock PDF with metadata including dates
    mock_pdf_reader = Mock(spec=pypdf.PdfReader)
    mock_pdf_reader.pages = [Mock(), Mock()]  # 2 pages

    # Mock page content
    for page in mock_pdf_reader.pages:
        page.extract_text.return_value = "Sample PDF content for testing."

    # Mock PDF metadata with dates in PDF format
    mock_pdf_reader.metadata = {
        "/Title": "Test Legal Document",
        "/Author": "Legal Department",
        "/Subject": "Policy Guidelines",
        "/Creator": "Legal Office",
        "/Producer": "PDF Generator v2.0",
        "/CreationDate": "D:20250525092926-04'00'",  # May 25, 2025, 9:29:26 AM EDT
        "/ModDate": "D:20241215143012+00'00'",  # Dec 15, 2024, 2:30:12 PM UTC
    }

    # Create mock file data
    file_data = io.BytesIO(b"fake pdf content")

    # Patch the pypdf.PdfReader to return our mock
    with patch("src.ingestion.service.pypdf.PdfReader", return_value=mock_pdf_reader):
        content, metadata = await service._extract_pdf_content(file_data)

    # Verify content extraction
    assert content == "Sample PDF content for testing.\nSample PDF content for testing."

    # Verify metadata extraction
    assert metadata.title == "Test Legal Document"
    assert metadata.author == "Legal Department"
    assert metadata.subject == "Policy Guidelines"
    assert metadata.page_count == 2
    assert metadata.word_count == 10  # "Sample PDF content for testing." x2

    # Verify date parsing - creation date should be converted to UTC
    assert metadata.creation_date is not None
    expected_creation = datetime(2025, 5, 25, 13, 29, 26)  # Converted to UTC
    assert abs((metadata.creation_date - expected_creation).total_seconds()) < 60

    # Verify modification date parsing
    assert metadata.modification_date is not None
    expected_modification = datetime(2024, 12, 15, 14, 30, 12)  # Already UTC
    assert (
        abs((metadata.modification_date - expected_modification).total_seconds()) < 60
    )


def test_pdf_date_format_edge_cases():
    """Test edge cases in PDF date formatting."""
    service = DocumentIngestionService()

    # Test cases for various date formats found in real PDFs
    test_cases = [
        # Standard format with timezone
        ("D:20250525092926-04'00'", datetime(2025, 5, 25, 13, 29, 26)),
        # Format without seconds
        ("D:202505250929-04'00'", None),  # Should fail gracefully
        # Format with positive timezone
        ("D:20240101120000+05'30'", datetime(2024, 1, 1, 6, 30, 0)),
        # Format without timezone
        ("D:20230615143000", datetime(2023, 6, 15, 14, 30, 0)),
        # Format without 'D:' prefix (some PDFs)
        ("20230615143000", datetime(2023, 6, 15, 14, 30, 0)),
        # Malformed dates should return None
        ("D:invalid", None),
        ("", None),
        (None, None),
    ]

    for pdf_date, expected in test_cases:
        result = service._parse_pdf_date(pdf_date)

        if expected is None:
            assert result is None, f"Expected None for '{pdf_date}', got {result}"
        else:
            assert result is not None, f"Expected datetime for '{pdf_date}', got None"
            # Allow for small differences due to timezone conversion
            diff = abs((result - expected).total_seconds())
            assert (
                diff < 60
            ), f"Date mismatch for '{pdf_date}': expected {expected}, got {result}, diff: {diff}s"


def test_pdf_date_logging():
    """Test that PDF date parsing includes proper logging."""
    # Test with a valid date
    with patch("structlog.get_logger") as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        service = DocumentIngestionService()
        result = service._parse_pdf_date("D:20250525092926-04'00'")

        # Verify debug log was called
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args

        assert "PDF date parsed successfully" in call_args[0][0]
        assert call_args[1]["original_date"] == "D:20250525092926-04'00'"
        assert "parsed_date" in call_args[1]

    # Test with an invalid date
    with patch("structlog.get_logger") as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        service = DocumentIngestionService()
        result = service._parse_pdf_date("invalid_date")

        # Verify warning log was called
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args

        assert "Failed to parse PDF date" in call_args[0][0]
        assert call_args[1]["pdf_date"] == "invalid_date"
        assert "error" in call_args[1]
