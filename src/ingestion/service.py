"""
Document ingestion service for AI Legal & Policy Research Assistant.

This module handles the ingestion of various document types including PDFs,
text files, DOCX files, and web URLs, extracting content and metadata.
"""

import hashlib
import mimetypes
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional, Tuple
from urllib.parse import urlparse
from uuid import UUID

import aiohttp
import pypdf
from bs4 import BeautifulSoup

from src.core.config import settings
from src.core.logging import LoggingMixin, log_document_processing
from src.core.models import Document, DocumentMetadata, DocumentType, ProcessingStatus


class DocumentIngestionError(Exception):
    """Custom exception for document ingestion errors."""

    pass


class DocumentIngestionService(LoggingMixin):
    """Service for ingesting and processing various document types."""

    def __init__(self):
        """Initialize the document ingestion service."""
        self.max_file_size = settings.max_file_size
        self.supported_types = settings.get_supported_file_types()

    async def ingest_file(
        self, file_data: BinaryIO, filename: str, file_size: int
    ) -> Document:
        """Ingest a file and extract its content and metadata.

        Args:
            file_data: Binary file data.
            filename: Original filename.
            file_size: Size of the file in bytes.

        Returns:
            Document: Processed document with extracted content and metadata.

        Raises:
            DocumentIngestionError: If ingestion fails.
        """
        try:
            # Validate file size
            if file_size > self.max_file_size:
                raise DocumentIngestionError(
                    f"File size {file_size} exceeds maximum allowed size {self.max_file_size}"
                )

            # Determine file type
            file_type = self._detect_file_type(filename)
            if file_type.value not in self.supported_types:
                raise DocumentIngestionError(
                    f"Unsupported file type: {file_type.value}"
                )

            # Calculate checksum
            file_data.seek(0)
            checksum = hashlib.sha256(file_data.read()).hexdigest()
            file_data.seek(0)

            # Extract content based on file type
            content, metadata = await self._extract_content(
                file_data, file_type, filename
            )

            # Create document instance
            document = Document(
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                content=content,
                metadata=metadata,
                processing_status=ProcessingStatus.COMPLETED,
                checksum=checksum,
            )

            log_document_processing(
                document_id=str(document.id),
                file_type=file_type.value,
                file_size=file_size,
                pages=metadata.page_count,
                success=True,
            )

            self.logger.info(
                "Document ingested successfully",
                document_id=str(document.id),
                filename=filename,
                file_type=file_type.value,
            )

            return document

        except Exception as e:
            self.logger.error(
                "Document ingestion failed",
                filename=filename,
                error=str(e),
                exc_info=True,
            )
            raise DocumentIngestionError(f"Failed to ingest document: {str(e)}")

    async def ingest_url(self, url: str) -> Document:
        """Ingest content from a web URL.

        Args:
            url: URL to fetch content from.

        Returns:
            Document: Processed document with extracted content.

        Raises:
            DocumentIngestionError: If URL ingestion fails.
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise DocumentIngestionError(f"Invalid URL: {url}")

            # Fetch content
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise DocumentIngestionError(
                            f"Failed to fetch URL {url}: HTTP {response.status}"
                        )

                    content_type = response.headers.get("content-type", "").lower()
                    raw_content = await response.read()

            # Process based on content type
            if "application/pdf" in content_type:
                with tempfile.NamedTemporaryFile() as temp_file:
                    temp_file.write(raw_content)
                    temp_file.seek(0)
                    content, metadata = await self._extract_pdf_content(temp_file)
            elif "text/html" in content_type:
                content, metadata = self._extract_html_content(
                    raw_content.decode("utf-8")
                )
            else:
                # Treat as plain text
                content = raw_content.decode("utf-8", errors="ignore")
                metadata = DocumentMetadata(
                    word_count=len(content.split()),
                    title=parsed_url.path.split("/")[-1] or parsed_url.netloc,
                )

            # Calculate checksum
            checksum = hashlib.sha256(raw_content).hexdigest()

            # Create document instance
            document = Document(
                filename=parsed_url.path.split("/")[-1] or f"{parsed_url.netloc}.html",
                file_type=DocumentType.URL,
                file_size=len(raw_content),
                content=content,
                metadata=metadata,
                processing_status=ProcessingStatus.COMPLETED,
                source_url=url,
                checksum=checksum,
            )

            log_document_processing(
                document_id=str(document.id),
                file_type="url",
                file_size=len(raw_content),
                success=True,
            )

            self.logger.info(
                "URL content ingested successfully",
                document_id=str(document.id),
                url=url,
            )

            return document

        except Exception as e:
            self.logger.error(
                "URL ingestion failed", url=url, error=str(e), exc_info=True
            )
            raise DocumentIngestionError(f"Failed to ingest URL: {str(e)}")

    def _detect_file_type(self, filename: str) -> DocumentType:
        """Detect file type from filename extension.

        Args:
            filename: Name of the file.

        Returns:
            DocumentType: Detected file type. Unknown extensions default to TEXT.
        """
        extension = Path(filename).suffix.lower().lstrip(".")

        type_mapping = {
            "pdf": DocumentType.PDF,
            "txt": DocumentType.TEXT,
            "docx": DocumentType.DOCX,
            "html": DocumentType.HTML,
            "htm": DocumentType.HTML,
        }

        return type_mapping.get(extension, DocumentType.TEXT)

    async def _extract_content(
        self, file_data: BinaryIO, file_type: DocumentType, filename: str
    ) -> Tuple[str, DocumentMetadata]:
        """Extract content and metadata based on file type.

        Args:
            file_data: Binary file data.
            file_type: Type of the file.
            filename: Original filename.

        Returns:
            Tuple of extracted content and metadata.
        """
        if file_type == DocumentType.PDF:
            return await self._extract_pdf_content(file_data)
        elif file_type == DocumentType.TEXT:
            return self._extract_text_content(file_data)
        elif file_type == DocumentType.HTML:
            return self._extract_html_content(file_data.read().decode("utf-8"))
        elif file_type == DocumentType.DOCX:
            return await self._extract_docx_content(file_data)
        else:
            # Fallback to text extraction
            return self._extract_text_content(file_data)

    async def _extract_pdf_content(
        self, file_data: BinaryIO
    ) -> Tuple[str, DocumentMetadata]:
        """Extract content from PDF files.

        Args:
            file_data: PDF file data.

        Returns:
            Tuple of extracted text and metadata.
        """
        try:
            pdf_reader = pypdf.PdfReader(file_data)

            # Extract text from all pages
            text_content = []
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())

            content = "\n".join(text_content)

            # Extract metadata
            pdf_info = pdf_reader.metadata
            metadata = DocumentMetadata(
                title=pdf_info.get("/Title"),
                author=pdf_info.get("/Author"),
                subject=pdf_info.get("/Subject"),
                page_count=len(pdf_reader.pages),
                word_count=len(content.split()) if content else 0,
                creation_date=pdf_info.get("/CreationDate"),
                modification_date=pdf_info.get("/ModDate"),
            )

            return content, metadata

        except Exception as e:
            raise DocumentIngestionError(f"Failed to extract PDF content: {str(e)}")

    def _extract_text_content(
        self, file_data: BinaryIO
    ) -> Tuple[str, DocumentMetadata]:
        """Extract content from text files.

        Args:
            file_data: Text file data.

        Returns:
            Tuple of extracted text and metadata.
        """
        try:
            content = file_data.read().decode("utf-8", errors="ignore")

            metadata = DocumentMetadata(
                word_count=len(content.split()),
                page_count=1,  # Text files are considered single page
            )

            return content, metadata

        except Exception as e:
            raise DocumentIngestionError(f"Failed to extract text content: {str(e)}")

    def _extract_html_content(self, html_content: str) -> Tuple[str, DocumentMetadata]:
        """Extract content from HTML.

        Args:
            html_content: Raw HTML content.

        Returns:
            Tuple of extracted text and metadata.
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else None

            # Extract meta keywords
            keywords = []
            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords and meta_keywords.get("content"):
                keywords = [k.strip() for k in meta_keywords["content"].split(",")]

            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            content = soup.get_text()
            # Clean up whitespace
            content = "\n".join(
                line.strip() for line in content.splitlines() if line.strip()
            )

            metadata = DocumentMetadata(
                title=title,
                keywords=keywords,
                word_count=len(content.split()),
                page_count=1,
            )

            return content, metadata

        except Exception as e:
            raise DocumentIngestionError(f"Failed to extract HTML content: {str(e)}")

    async def _extract_docx_content(
        self, file_data: BinaryIO
    ) -> Tuple[str, DocumentMetadata]:
        """Extract content from DOCX files.

        Note: This is a placeholder implementation. A full implementation would
        require the python-docx library.

        Args:
            file_data: DOCX file data.

        Returns:
            Tuple of extracted text and metadata.
        """
        # For now, treat as unsupported and fall back to text
        raise DocumentIngestionError(
            "DOCX support not yet implemented. Please convert to PDF or text format."
        )

    async def get_document_by_id(self, document_id: UUID) -> Optional[Document]:
        """Retrieve a document by its ID.

        Note: This is a placeholder. In a real implementation, this would
        query a database or storage system.

        Args:
            document_id: ID of the document to retrieve.

        Returns:
            Document if found, None otherwise.
        """
        # Placeholder implementation
        # In a real application, this would query a database
        self.logger.info("Retrieving document", document_id=str(document_id))
        return None
