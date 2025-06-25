#!/usr/bin/env python3
"""
Entity Caching Demo

This script demonstrates the entity caching functionality in the AI pipeline.
It shows how entities from previous analyses are reused to improve performance.
"""

import asyncio
import time
from uuid import uuid4

from src.core.models import Document, DocumentType, ProcessingStatus, DocumentMetadata
from src.core.repository import DocumentRepository, AnalysisRepository
from src.pipeline.service import AIAnalysisPipeline


async def demo_entity_caching():
    """Demonstrate entity caching functionality."""

    # Initialize repositories
    doc_repo = DocumentRepository()
    analysis_repo = AnalysisRepository()

    # Create a test document
    document = Document(
        filename="entity_caching_demo.txt",
        file_type=DocumentType.TEXT,
        file_size=200,
        content="""
        This is a test legal document for demonstrating entity caching.
        
        The document contains several key entities:
        - Federal Civil Rights Act of 1964
        - Department of Justice
        - Equal Employment Opportunity Commission
        - Protected classes including race, color, religion, sex, and national origin
        
        The legislation establishes comprehensive anti-discrimination protections
        in employment, housing, education, and public accommodations.
        """,
        metadata=DocumentMetadata(
            title="Entity Caching Demo Document", author="Demo Author"
        ),
        processing_status=ProcessingStatus.COMPLETED,
    )

    # Save document to database
    saved_doc = doc_repo.create(document)
    print(f"Created document: {saved_doc.filename} (ID: {saved_doc.id})")

    # Initialize AI pipeline
    pipeline = AIAnalysisPipeline(analysis_repository=analysis_repo)

    print("\n" + "=" * 60)
    print("FIRST ANALYSIS - Extracting new entities")
    print("=" * 60)

    # First analysis - should extract new entities
    start_time = time.time()
    result1 = await pipeline.analyze_document(saved_doc)
    duration1 = time.time() - start_time

    print(f"Analysis completed in {duration1:.2f} seconds")
    print(f"Analysis ID: {result1.id}")
    print(f"Entities extracted: {len(result1.entities)}")
    print(f"Entities source analysis: {result1.entities_source_analysis_id}")
    print(f"Processing time: {result1.processing_time_ms:.2f}ms")

    # Show some extracted entities
    print("\nExtracted entities:")
    for i, entity in enumerate(result1.entities[:3]):  # Show first 3
        print(
            f"  {i+1}. {entity.entity_type}: {entity.entity_value} (confidence: {entity.confidence:.2f})"
        )

    print("\n" + "=" * 60)
    print("SECOND ANALYSIS - Reusing cached entities")
    print("=" * 60)

    # Second analysis - should reuse entities from first analysis
    start_time = time.time()
    result2 = await pipeline.analyze_document(saved_doc)
    duration2 = time.time() - start_time

    print(f"Analysis completed in {duration2:.2f} seconds")
    print(f"Analysis ID: {result2.id}")
    print(f"Entities extracted: {len(result2.entities)}")
    print(f"Entities source analysis: {result2.entities_source_analysis_id}")
    print(f"Processing time: {result2.processing_time_ms:.2f}ms")

    # Verify entities are the same
    if result2.entities_source_analysis_id == result1.id:
        print("✅ SUCCESS: Entities were reused from previous analysis!")
        print(f"   Time saved: {duration1 - duration2:.2f} seconds")
    else:
        print("❌ FAILED: Entities were not reused")

    print("\n" + "=" * 60)
    print("THIRD ANALYSIS - Force regenerate entities")
    print("=" * 60)

    # Third analysis - force regenerate entities
    start_time = time.time()
    result3 = await pipeline.analyze_document(saved_doc, force_regenerate_entities=True)
    duration3 = time.time() - start_time

    print(f"Analysis completed in {duration3:.2f} seconds")
    print(f"Analysis ID: {result3.id}")
    print(f"Entities extracted: {len(result3.entities)}")
    print(f"Entities source analysis: {result3.entities_source_analysis_id}")
    print(f"Processing time: {result3.processing_time_ms:.2f}ms")

    # Verify entities were regenerated
    if result3.entities_source_analysis_id is None:
        print("✅ SUCCESS: Entities were regenerated as requested!")
    else:
        print("❌ FAILED: Entities were not regenerated")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"First analysis (new entities):  {duration1:.2f}s")
    print(f"Second analysis (cached):       {duration2:.2f}s")
    print(f"Third analysis (forced new):    {duration3:.2f}s")
    print(f"Time saved with caching:        {duration1 - duration2:.2f}s")
    print(
        f"Performance improvement:        {((duration1 - duration2) / duration1 * 100):.1f}%"
    )


if __name__ == "__main__":
    print("Entity Caching Demo")
    print("==================")
    print("This demo shows how entity extraction is cached to improve performance.")
    print("The first analysis extracts entities, subsequent analyses reuse them.")
    print()

    asyncio.run(demo_entity_caching())
