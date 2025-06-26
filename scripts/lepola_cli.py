#!/usr/bin/env python3
"""
Lepola CLI - Command Line Interface for AI Legal & Policy Research Assistant.

This CLI tool provides easy access to all server functionality including:
- Document upload and ingestion
- Document querying and analysis
- AI pipeline operations
- Output generation
- Service status monitoring
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from src.core.config import settings


class LepolaCLI:
    """CLI client for the Lepola server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the CLI client.

        Args:
            base_url: Base URL of the Lepola server.
        """
        self.base_url = base_url.rstrip("/")
        self.console = Console()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Configure timeouts for long-running operations with local LLMs
        timeout = aiohttp.ClientTimeout(
            total=settings.http_timeout,
            connect=settings.http_connect_timeout,
            sock_read=settings.http_read_timeout,
        )
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request to the server.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for the request

        Returns:
            Response data as dictionary

        Raises:
            click.ClickException: If the request fails
        """
        if not self.session:
            raise click.ClickException("Session not initialized")

        url = f"{self.base_url}{endpoint}"

        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("detail", error_text)
                    except:
                        error_msg = error_text

                    raise click.ClickException(f"HTTP {response.status}: {error_msg}")

                if response.status == 204:  # No content
                    return {}

                return await response.json()
        except aiohttp.ClientError as e:
            raise click.ClickException(f"Connection error: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        return await self._make_request("GET", "/health")

    async def upload_file(
        self,
        file_path: str,
        metadata: Optional[str] = None,
        async_embedding: bool = True,
    ) -> Dict[str, Any]:
        """Upload a file for ingestion.

        Args:
            file_path: Path to the file to upload
            metadata: Optional metadata as JSON string
            async_embedding: Whether to run embedding asynchronously

        Returns:
            Upload response data
        """
        if not os.path.exists(file_path):
            raise click.ClickException(f"File not found: {file_path}")

        data = aiohttp.FormData()

        # Read file content into memory to avoid "closed file" error
        with open(file_path, "rb") as f:
            file_content = f.read()

        data.add_field("file", file_content, filename=os.path.basename(file_path))

        if metadata:
            data.add_field("metadata", metadata)

        data.add_field("async_embedding", str(async_embedding).lower())

        return await self._make_request("POST", "/api/v1/ingestion/upload", data=data)

    async def ingest_url(
        self, url: str, metadata: Optional[str] = None, async_embedding: bool = True
    ) -> Dict[str, Any]:
        """Ingest content from a URL.

        Args:
            url: URL to ingest
            metadata: Optional metadata as JSON string
            async_embedding: Whether to run embedding asynchronously

        Returns:
            Ingestion response data
        """
        data = aiohttp.FormData()
        data.add_field("url", url)

        if metadata:
            data.add_field("metadata", metadata)

        data.add_field("async_embedding", str(async_embedding).lower())

        return await self._make_request("POST", "/api/v1/ingestion/url", data=data)

    async def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List ingested documents.

        Args:
            limit: Number of documents to return
            offset: Number of documents to skip
            file_type: Filter by file type
            status: Filter by processing status

        Returns:
            List of documents
        """
        params = {"limit": limit, "offset": offset}
        if file_type:
            params["file_type"] = file_type
        if status:
            params["status"] = status

        return await self._make_request(
            "GET", "/api/v1/ingestion/documents", params=params
        )

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get document information.

        Args:
            document_id: Document ID

        Returns:
            Document information
        """
        return await self._make_request(
            "GET", f"/api/v1/ingestion/document/{document_id}"
        )

    async def ask_question(
        self, question: str, document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Ask a question about documents.

        Args:
            question: The question to ask
            document_ids: Optional list of document IDs to search in

        Returns:
            Query result
        """
        data = {"question": question}
        if document_ids:
            data["document_ids"] = document_ids

        return await self._make_request("POST", "/api/v1/query/ask", json=data)

    async def analyze_document(
        self, document_id: str, force_regenerate_entities: bool = False
    ) -> Dict[str, Any]:
        """Start AI analysis of a document.

        Args:
            document_id: Document ID to analyze
            force_regenerate_entities: If True, regenerate entities even if they exist

        Returns:
            Analysis start response
        """
        params = {}
        if force_regenerate_entities:
            params["force_regenerate_entities"] = "true"

        return await self._make_request(
            "POST", f"/api/v1/pipeline/analyze/{document_id}", params=params
        )

    async def get_analysis_results(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis results.

        Args:
            analysis_id: Analysis ID

        Returns:
            Analysis results
        """
        return await self._make_request(
            "GET", f"/api/v1/pipeline/analysis/{analysis_id}"
        )

    async def list_analyses(
        self, limit: int = 10, offset: int = 0, status_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """List all analysis jobs.

        Args:
            limit: Number of analyses to return
            offset: Number of analyses to skip
            status_filter: Filter by status

        Returns:
            List of analyses
        """
        params = {"limit": limit, "offset": offset}
        if status_filter:
            params["status_filter"] = status_filter

        return await self._make_request(
            "GET", "/api/v1/pipeline/analyses", params=params
        )

    async def generate_output(
        self, analysis_id: str, output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """Generate output from analysis.

        Args:
            analysis_id: Analysis ID
            output_format: Output format (markdown, html, json, pdf)

        Returns:
            Generated output information
        """
        data = {"analysis_id": analysis_id, "format": output_format}

        return await self._make_request("POST", "/api/v1/outputs/generate", json=data)

    async def get_service_status(self, service: str) -> Dict[str, Any]:
        """Get service status.

        Args:
            service: Service name (ingestion, query, pipeline, outputs, embeddings)

        Returns:
            Service status
        """
        return await self._make_request("GET", f"/api/v1/{service}/status")


def display_health_check(health_data: Dict[str, Any]):
    """Display health check results."""
    console = Console()

    status = health_data.get("status", "unknown")
    service = health_data.get("service", "unknown")

    if status == "healthy":
        console.print(f"✅ [green]Server is healthy[/green] - {service}")
    else:
        console.print(f"❌ [red]Server is unhealthy[/red] - {service}")


def display_documents(documents_data: Dict[str, Any]):
    """Display documents in a table."""
    console = Console()

    documents = documents_data.get("documents", [])
    total = documents_data.get("total", 0)

    if not documents:
        console.print("No documents found.")
        return

    table = Table(title=f"Documents ({total} total)")
    table.add_column("ID", style="cyan", no_wrap=True, min_width=36)  # UUID is 36 chars
    table.add_column("Filename", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Size", style="blue")
    table.add_column("Status", style="magenta")
    table.add_column("Created", style="white")

    for doc in documents:
        table.add_row(
            str(doc.get("id", "")),
            doc.get("filename", ""),
            doc.get("file_type", ""),
            f"{doc.get('file_size', 0):,} bytes",
            doc.get("processing_status", ""),
            doc.get("created_at", "")[:19] if doc.get("created_at") else "",
        )

    console.print(table)


def display_analyses(analyses_data: Dict[str, Any]):
    """Display analyses in a table."""
    console = Console()

    analyses = analyses_data.get("analyses", [])
    total = analyses_data.get("total", 0)

    if not analyses:
        console.print("No analyses found.")
        return

    table = Table(title=f"Analyses ({total} total)")
    table.add_column(
        "Analysis ID", style="cyan", no_wrap=True, min_width=36
    )  # UUID is 36 chars
    table.add_column("Document", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Confidence", style="blue")
    table.add_column("Entities", style="magenta", justify="center")
    table.add_column("Warnings", style="red", justify="center")
    table.add_column("Created", style="white")

    for analysis in analyses:
        # Get entity count and warning count from the analysis data
        entity_count = analysis.get("entity_count", 0)
        warning_count = analysis.get("warning_count", 0)

        # Format entity count with emoji
        entity_display = f"📊 {entity_count}" if entity_count > 0 else "0"

        # Format warning count with emoji
        warning_display = f"⚠️ {warning_count}" if warning_count > 0 else "✅ 0"

        table.add_row(
            str(
                analysis.get("analysis_id", "")
            ),  # Fixed: use analysis_id instead of id
            analysis.get("document_filename", ""),
            analysis.get("status", ""),
            str(analysis.get("confidence_level", "")),
            entity_display,
            warning_display,
            analysis.get("created_at", "")[:19] if analysis.get("created_at") else "",
        )

    console.print(table)


def display_query_result(result: Dict[str, Any]):
    """Display query results."""
    console = Console()

    answer = result.get("answer", "")
    confidence = result.get("confidence", 0)
    sources = result.get("sources", [])
    suggestions = result.get("suggestions", [])
    warnings = result.get("warnings", [])

    # Display answer
    console.print(Panel(answer, title="Answer", border_style="green"))

    # Display confidence
    console.print(f"Confidence: {confidence:.2%}")

    # Display sources if any
    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for source in sources:
            console.print(f"  • {source}")

    # Display suggestions if any
    if suggestions:
        console.print("\n[bold]Suggestions:[/bold]")
        for suggestion in suggestions:
            console.print(f"  • {suggestion}")

    # Display warnings if any
    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")


@click.group()
@click.option("--server", default="http://localhost:8000", help="Server URL")
@click.pass_context
def cli(ctx, server):
    """Lepola CLI - AI Legal & Policy Research Assistant."""
    ctx.ensure_object(dict)
    ctx.obj["server"] = server


@cli.command()
@click.pass_context
def health(ctx):
    """Check server health."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Checking server health...", total=None)
                health_data = await client.health_check()
                progress.update(task, completed=True)

            display_health_check(health_data)

    asyncio.run(run())


@cli.group()
def documents():
    """Manage documents."""
    pass


@documents.command("upload")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--metadata", help="Metadata as JSON string")
@click.option("--no-embedding", is_flag=True, help="Disable async embedding")
@click.pass_context
def upload_document(ctx, file_path, metadata, no_embedding):
    """Upload a document for ingestion."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Uploading document...", total=None)
                result = await client.upload_file(file_path, metadata, not no_embedding)
                progress.update(task, completed=True)

            console = Console()
            console.print(f"✅ Document uploaded successfully!")
            console.print(f"Document ID: {result.get('document_id')}")
            console.print(f"Status: {result.get('status')}")
            console.print(f"Message: {result.get('message')}")

    asyncio.run(run())


@documents.command("ingest-url")
@click.argument("url")
@click.option("--metadata", help="Metadata as JSON string")
@click.option("--no-embedding", is_flag=True, help="Disable async embedding")
@click.pass_context
def ingest_url(ctx, url, metadata, no_embedding):
    """Ingest content from a URL."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Ingesting URL...", total=None)
                result = await client.ingest_url(url, metadata, not no_embedding)
                progress.update(task, completed=True)

            console = Console()
            console.print(f"✅ URL ingested successfully!")
            console.print(f"Document ID: {result.get('document_id')}")
            console.print(f"Status: {result.get('status')}")
            console.print(f"Message: {result.get('message')}")

    asyncio.run(run())


@documents.command("list")
@click.option("--limit", default=50, help="Number of documents to return")
@click.option("--offset", default=0, help="Number of documents to skip")
@click.option("--file-type", help="Filter by file type")
@click.option("--status", help="Filter by processing status")
@click.pass_context
def list_documents(ctx, limit, offset, file_type, status):
    """List ingested documents."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Fetching documents...", total=None)
                result = await client.list_documents(limit, offset, file_type, status)
                progress.update(task, completed=True)

            display_documents(result)

    asyncio.run(run())


@documents.command("get")
@click.argument("document_id")
@click.pass_context
def get_document(ctx, document_id):
    """Get document information."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Fetching document...", total=None)
                result = await client.get_document(document_id)
                progress.update(task, completed=True)

            console = Console()
            console.print(
                Panel(
                    f"ID: {result.get('id')}\n"
                    f"Filename: {result.get('filename')}\n"
                    f"Type: {result.get('file_type')}\n"
                    f"Size: {result.get('file_size'):,} bytes\n"
                    f"Status: {result.get('processing_status')}\n"
                    f"Created: {result.get('created_at')}",
                    title="Document Information",
                    border_style="blue",
                )
            )

    asyncio.run(run())


@cli.group()
def query():
    """Query documents."""
    pass


@query.command("ask")
@click.argument("question")
@click.option("--document-ids", multiple=True, help="Document IDs to search in")
@click.pass_context
def ask_question(ctx, question, document_ids):
    """Ask a question about documents."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Processing question...", total=None)
                result = await client.ask_question(
                    question, list(document_ids) if document_ids else None
                )
                progress.update(task, completed=True)

            display_query_result(result)

    asyncio.run(run())


@cli.group()
def pipeline():
    """AI pipeline operations."""
    pass


@pipeline.command("analyze")
@click.argument("document_id")
@click.option(
    "--force-regenerate-entities",
    is_flag=True,
    help="Regenerate entities even if they exist",
)
@click.pass_context
def analyze_document(ctx, document_id, force_regenerate_entities):
    """Start AI analysis of a document."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Starting analysis...", total=None)
                result = await client.analyze_document(
                    document_id, force_regenerate_entities
                )
                progress.update(task, completed=True)

            console = Console()
            console.print(f"✅ Analysis started successfully!")
            console.print(f"Analysis ID: {result.get('analysis_id')}")
            console.print(f"Status: {result.get('status')}")
            console.print(f"Message: {result.get('message')}")

    asyncio.run(run())


@pipeline.command("results")
@click.argument("analysis_id")
@click.pass_context
def get_analysis_results(ctx, analysis_id):
    """Get analysis results."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Fetching analysis results...", total=None)
                result = await client.get_analysis_results(analysis_id)
                progress.update(task, completed=True)

            console = Console()

            # Basic analysis info
            console.print(
                Panel(
                    f"Analysis ID: {result.get('analysis_id')}\n"
                    f"Document: {result.get('document_filename')}\n"
                    f"Status: {result.get('status')}\n"
                    f"Confidence: {result.get('confidence_level')}\n"
                    f"Processing Time: {result.get('processing_time_ms')}ms\n"
                    f"Created: {result.get('created_at')}",
                    title="Analysis Results",
                    border_style="green",
                )
            )

            # Display detailed results if available
            if result.get("status") == "completed" and "result" in result:
                analysis_data = result["result"]

                # Entity count information
                entities = analysis_data.get("entities", [])
                entity_count = len(entities)
                console.print(
                    Panel(
                        f"📊 Entities Found: {entity_count}\n"
                        f"🔍 Entity Types: {', '.join(set(e.get('entity_type', 'unknown') for e in entities)) if entities else 'None'}",
                        title="Entity Extraction",
                        border_style="blue",
                    )
                )

                # Summary information
                summary = analysis_data.get("summary", {})
                if summary:
                    executive_summary = summary.get(
                        "executive_summary", "No summary available"
                    )
                    key_points = summary.get("key_points", [])

                    # Always show executive summary, then key points if present
                    panel_content = f"📝 Executive Summary:\n{executive_summary}\n\n"
                    if key_points:
                        panel_content += (
                            f"🔑 Key Points ({len(key_points)}):\n"
                            + "\n".join(f"• {point}" for point in key_points)
                        )
                    else:
                        panel_content += "🔑 No key points available"

                    console.print(
                        Panel(
                            panel_content,
                            title="Document Summary",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            "📝 No summary data available",
                            title="Document Summary",
                            border_style="yellow",
                        )
                    )

                # Warnings information
                warnings = analysis_data.get("warnings", [])
                if warnings:
                    console.print(
                        Panel(
                            "⚠️ Warnings:\n"
                            + "\n".join(f"• {warning}" for warning in warnings),
                            title="Analysis Warnings",
                            border_style="red",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            "✅ No warnings generated",
                            title="Analysis Warnings",
                            border_style="green",
                        )
                    )

                # Additional analysis details
                requires_review = analysis_data.get("requires_human_review", False)
                model_used = analysis_data.get("model_used", "Unknown")

                console.print(
                    Panel(
                        f"🤖 Model Used: {model_used}\n"
                        f"👁️ Requires Human Review: {'Yes' if requires_review else 'No'}",
                        title="Analysis Details",
                        border_style="cyan",
                    )
                )

            elif result.get("status") == "failed":
                error_msg = result.get("error", "Unknown error occurred")
                console.print(
                    Panel(
                        f"❌ Analysis failed: {error_msg}",
                        title="Analysis Error",
                        border_style="red",
                    )
                )

    asyncio.run(run())


@pipeline.command("list")
@click.option("--limit", default=10, help="Number of analyses to return")
@click.option("--offset", default=0, help="Number of analyses to skip")
@click.option("--status", help="Filter by status")
@click.pass_context
def list_analyses(ctx, limit, offset, status):
    """List all analysis jobs."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Fetching analyses...", total=None)
                result = await client.list_analyses(limit, offset, status)
                progress.update(task, completed=True)

            display_analyses(result)

    asyncio.run(run())


@cli.group()
def outputs():
    """Output generation."""
    pass


@outputs.command("generate")
@click.argument("analysis_id")
@click.option(
    "--format",
    default="markdown",
    type=click.Choice(["markdown", "html", "json", "pdf"]),
    help="Output format",
)
@click.pass_context
def generate_output(ctx, analysis_id, format):
    """Generate output from analysis."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Generating output...", total=None)
                result = await client.generate_output(analysis_id, format)
                progress.update(task, completed=True)

            console = Console()
            console.print(f"✅ Output generated successfully!")
            console.print(f"Analysis ID: {result.get('analysis_id')}")
            console.print(f"Format: {result.get('format')}")
            console.print(f"Size: {result.get('size_bytes'):,} bytes")

            # Display content if it's text-based
            if format in ["markdown", "html", "json"]:
                console.print("\n[bold]Content:[/bold]")
                console.print(result.get("content", ""))

    asyncio.run(run())


@cli.group()
def status():
    """Service status."""
    pass


@status.command("ingestion")
@click.pass_context
def ingestion_status(ctx):
    """Get ingestion service status."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Checking ingestion status...", total=None)
                result = await client.get_service_status("ingestion")
                progress.update(task, completed=True)

            console = Console()
            console.print(
                Panel(
                    f"Status: {result.get('status')}\n"
                    f"Service: {result.get('service')}\n"
                    f"Supported Types: {', '.join(result.get('supported_file_types', []))}\n"
                    f"Max File Size: {result.get('max_file_size', 0):,} bytes",
                    title="Ingestion Service Status",
                    border_style="blue",
                )
            )

    asyncio.run(run())


@status.command("query")
@click.pass_context
def query_status(ctx):
    """Get query service status."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Checking query status...", total=None)
                result = await client.get_service_status("query")
                progress.update(task, completed=True)

            console = Console()
            console.print(
                Panel(
                    f"Status: {result.get('status')}\n"
                    f"Service: {result.get('service')}\n"
                    f"Vector DB Available: {result.get('vector_db_available')}\n"
                    f"Features: {', '.join(result.get('features', []))}",
                    title="Query Service Status",
                    border_style="green",
                )
            )

    asyncio.run(run())


@status.command("pipeline")
@click.pass_context
def pipeline_status(ctx):
    """Get pipeline service status."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Checking pipeline status...", total=None)
                result = await client.get_service_status("pipeline")
                progress.update(task, completed=True)

            console = Console()
            console.print(
                Panel(
                    f"Status: {result.get('status')}\n"
                    f"Service: {result.get('service')}\n"
                    f"LLM Provider: {result.get('llm_provider')}\n"
                    f"Model: {result.get('model')}",
                    title="Pipeline Service Status",
                    border_style="yellow",
                )
            )

    asyncio.run(run())


@status.command("outputs")
@click.pass_context
def outputs_status(ctx):
    """Get outputs service status."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Checking outputs status...", total=None)
                result = await client.get_service_status("outputs")
                progress.update(task, completed=True)

            console = Console()
            console.print(
                Panel(
                    f"Status: {result.get('status')}\n"
                    f"Service: {result.get('service')}\n"
                    f"Supported Formats: {', '.join(result.get('supported_formats', []))}\n"
                    f"Template Engine: {result.get('template_engine')}",
                    title="Outputs Service Status",
                    border_style="magenta",
                )
            )

    asyncio.run(run())


@status.command("embeddings")
@click.pass_context
def embeddings_status(ctx):
    """Get embeddings service status."""

    async def run():
        async with LepolaCLI(ctx.obj["server"]) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=Console(),
            ) as progress:
                task = progress.add_task("Checking embeddings status...", total=None)
                result = await client.get_service_status("embeddings")
                progress.update(task, completed=True)

            console = Console()
            console.print(
                Panel(
                    f"Status: {result.get('status')}\n"
                    f"Service: {result.get('service')}\n"
                    f"Provider: {result.get('provider')}\n"
                    f"Model: {result.get('model')}",
                    title="Embeddings Service Status",
                    border_style="cyan",
                )
            )

    asyncio.run(run())


@cli.command()
@click.option("--confidence", is_flag=True, help="Prioritize confidence over speed")
@click.option("--speed", is_flag=True, help="Prioritize speed over confidence")
@click.option("--report", is_flag=True, help="Show detailed system report")
def model_select(confidence: bool, speed: bool, report: bool):
    """Select the optimal model for your system."""
    try:
        from src.utils.model_selector import ModelSelector

        selector = ModelSelector()

        if report:
            import json

            report_data = selector.get_system_report()
            console = Console()
            console.print("[bold green]System Report[/bold green]")
            console.print(json.dumps(report_data, indent=2))
        else:
            prioritize_confidence = confidence or not speed
            best_model = selector.get_best_model(prioritize_confidence)

            console = Console()

            if best_model:
                console.print(
                    f"[bold green]Recommended model:[/bold green] {best_model.name}"
                )
                console.print(
                    f"[bold blue]Expected confidence:[/bold blue] {best_model.expected_confidence:.2f}"
                )
                console.print(
                    f"[bold yellow]Speed rating:[/bold yellow] {best_model.speed_rating}/10"
                )
                console.print(
                    f"[bold magenta]RAM requirement:[/bold magenta] {best_model.recommended_ram_gb:.1f} GB"
                )

                # Show how to configure it
                console.print(
                    "\n[bold]To use this model, set in your .env file:[/bold]"
                )
                console.print(f"DEFAULT_LLM_PROVIDER=ollama")
                console.print(f"# The model will be automatically selected")
            else:
                console.print("[bold red]No compatible models found![/bold red]")
                console.print(
                    f"Available system RAM: {selector.system_info['available_ram_gb']:.1f} GB"
                )
                console.print("\n[bold yellow]Recommendations:[/bold yellow]")
                console.print("1. Close other applications to free up RAM")
                console.print("2. Consider using a smaller model like gemma3:1b")
                console.print("3. Or use cloud-based models (OpenAI/Anthropic)")

    except ImportError as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("Make sure you have installed all dependencies: poetry install")
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    cli()
