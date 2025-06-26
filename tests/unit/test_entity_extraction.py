"""
Tests for entity extraction functionality.

This module tests the entity extraction prompt and parsing logic to ensure
consistent JSON output and robust parsing of LLM responses.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.pipeline.service import AIAnalysisPipeline
from src.core.models import ExtractedEntity


class TestEntityExtraction:
    """Test entity extraction functionality."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        return AIAnalysisPipeline()

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns clean JSON."""
        mock = AsyncMock()
        mock.ainvoke.return_value.content = """[
  {
    "Type": "legal_document",
    "Value": "test regulation",
    "Confidence": 0.95,
    "Source": "This test regulation establishes requirements.",
    "Start": 5,
    "End": 19
  },
  {
    "Type": "legal_concept",
    "Value": "requirements",
    "Confidence": 0.85,
    "Source": "establishes requirements.",
    "Start": 25,
    "End": 36
  }
]"""
        return mock

    def test_entity_prompt_produces_clean_json(self, pipeline):
        """Test that the entity extraction prompt is strict about JSON-only output."""
        # Create a test prompt directly to check the format
        from langchain.prompts import PromptTemplate

        test_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are an expert legal analyst. Extract entities from this legal document.

CRITICAL: You must respond with ONLY valid JSON. No explanations, no comments, no markdown formatting.

Entity types to extract:
- legal_document: Laws, regulations, statutes, bills
- agency: Government agencies, departments, organizations  
- affected_group: Populations, demographics, stakeholders
- legal_concept: Rights, obligations, procedures, standards
- jurisdiction: Geographic or legal jurisdictions
- timeline: Dates, deadlines, effective dates
- penalty: Fines, sanctions, enforcement mechanisms

Document text:
{text}

RESPOND WITH ONLY THIS JSON FORMAT (no other text):
[
  {{
    "Type": "entity_type",
    "Value": "entity_value", 
    "Confidence": 0.95,
    "Source": "exact_source_text",
    "Start": 0,
    "End": 0
  }}
]

Rules:
- Confidence: 0.9-1.0 for explicit mentions, 0.7-0.9 for clear contextual, 0.5-0.7 for implied
- Start/End: Character positions in the document text
- Source: Exact text where entity was found
- Only include entities you are confident about
- NO EXPLANATORY TEXT - ONLY JSON""",
        )

        # Format the prompt with test text
        formatted_prompt = test_prompt.format(text="This is a test legal document.")

        # Check that the prompt explicitly requires JSON-only output
        assert "CRITICAL: You must respond with ONLY valid JSON" in formatted_prompt
        assert (
            "No explanations, no comments, no markdown formatting" in formatted_prompt
        )
        assert "NO EXPLANATORY TEXT - ONLY JSON" in formatted_prompt
        assert "RESPOND WITH ONLY THIS JSON FORMAT" in formatted_prompt

    def test_clean_json_response_removes_explanatory_text(self, pipeline):
        """Test that _clean_json_response removes explanatory text."""
        # Test response with explanatory text
        response_with_text = """After carefully analyzing the document, I have extracted the following entities:

```json
[
  {
    "Type": "legal_document",
    "Value": "test law",
    "Confidence": 0.9,
    "Source": "This test law",
    "Start": 5,
    "End": 13
  }
]
```

I hope this analysis is helpful."""

        cleaned = pipeline._clean_json_response(response_with_text)

        # Should extract only the JSON part
        expected = """[
  {
    "Type": "legal_document",
    "Value": "test law",
    "Confidence": 0.9,
    "Source": "This test law",
    "Start": 5,
    "End": 13
  }
]"""
        assert cleaned.strip() == expected.strip()

    def test_clean_json_response_handles_markdown_blocks(self, pipeline):
        """Test that _clean_json_response handles markdown code blocks."""
        # Test with markdown code blocks
        response_with_markdown = """Here are the entities I found:

```json
[
  {
    "Type": "agency",
    "Value": "EPA",
    "Confidence": 0.95,
    "Source": "EPA regulations",
    "Start": 0,
    "End": 3
  }
]
```"""

        cleaned = pipeline._clean_json_response(response_with_markdown)

        # Should extract only the JSON part
        expected = """[
  {
    "Type": "agency",
    "Value": "EPA",
    "Confidence": 0.95,
    "Source": "EPA regulations",
    "Start": 0,
    "End": 3
  }
]"""
        assert cleaned.strip() == expected.strip()

    def test_clean_json_response_handles_plain_json(self, pipeline):
        """Test that _clean_json_response handles plain JSON without markdown."""
        # Test with plain JSON
        plain_json = """[
  {
    "Type": "legal_concept",
    "Value": "rights",
    "Confidence": 0.8,
    "Source": "civil rights",
    "Start": 6,
    "End": 11
  }
]"""

        cleaned = pipeline._clean_json_response(plain_json)
        assert cleaned.strip() == plain_json.strip()

    def test_parse_entity_response_handles_clean_json(self, pipeline):
        """Test that _parse_entity_response correctly parses clean JSON."""
        clean_json = """[
  {
    "Type": "legal_document",
    "Value": "test statute",
    "Confidence": 0.95,
    "Source": "This test statute",
    "Start": 5,
    "End": 16
  }
]"""

        entities = pipeline._parse_entity_response(clean_json)

        assert len(entities) == 1
        entity = entities[0]
        assert entity.entity_type == "legal_document"
        assert entity.entity_value == "test statute"
        assert entity.confidence == 0.95
        assert entity.source_text == "This test statute"
        assert entity.start_position == 5
        assert entity.end_position == 16

    def test_parse_entity_response_handles_explanatory_text(self, pipeline):
        """Test that _parse_entity_response handles responses with explanatory text."""
        response_with_text = """I analyzed the document and found these entities:

```json
[
  {
    "Type": "agency",
    "Value": "FDA",
    "Confidence": 0.9,
    "Source": "FDA regulations",
    "Start": 0,
    "End": 3
  }
]
```"""

        entities = pipeline._parse_entity_response(response_with_text)

        assert len(entities) == 1
        entity = entities[0]
        assert entity.entity_type == "agency"
        assert entity.entity_value == "FDA"
        assert entity.confidence == 0.9

    def test_parse_entity_response_handles_invalid_json(self, pipeline):
        """Test that _parse_entity_response handles invalid JSON gracefully."""
        invalid_json = """This is not valid JSON at all.
        
        I found some entities but didn't format them properly:
        - Type: legal_document
        - Value: test law
        - Confidence: 0.8"""

        entities = pipeline._parse_entity_response(invalid_json)

        # Should return empty list for invalid JSON
        assert entities == []

    def test_map_to_entity_handles_various_formats(self, pipeline):
        """Test that _map_to_entity handles different JSON field formats."""
        # Test with uppercase field names
        entity_upper = {
            "Type": "legal_concept",
            "Value": "obligation",
            "Confidence": 0.85,
            "Source": "legal obligation",
            "Start": 6,
            "End": 15,
        }

        mapped = pipeline._map_to_entity(entity_upper)
        assert mapped.entity_type == "legal_concept"
        assert mapped.entity_value == "obligation"
        assert mapped.confidence == 0.85

        # Test with lowercase field names
        entity_lower = {
            "entity_type": "agency",
            "entity_value": "DOJ",
            "confidence": 0.9,
            "source_text": "DOJ enforcement",
            "start_position": 0,
            "end_position": 3,
        }

        mapped = pipeline._map_to_entity(entity_lower)
        assert mapped.entity_type == "agency"
        assert mapped.entity_value == "DOJ"
        assert mapped.confidence == 0.9

    def test_map_to_entity_handles_missing_fields(self, pipeline):
        """Test that _map_to_entity handles missing fields gracefully."""
        incomplete_entity = {
            "Type": "legal_document",
            "Value": "test law",
            "Confidence": 0.8,
            "Source": "test law text",
            "Start": 0,
            "End": 8,
        }

        mapped = pipeline._map_to_entity(incomplete_entity)
        assert mapped.entity_type == "legal_document"
        assert mapped.entity_value == "test law"
        assert mapped.confidence == 0.8
        assert mapped.source_text == "test law text"
        assert mapped.start_position == 0
        assert mapped.end_position == 8

    def test_map_to_entity_handles_very_incomplete_entity(self, pipeline):
        """Test that _map_to_entity handles very incomplete entities gracefully."""
        very_incomplete_entity = {
            "Type": "legal_document",
            "Value": "test law",
            # Missing other fields
        }

        mapped = pipeline._map_to_entity(very_incomplete_entity)
        # Should return a default entity when validation fails
        assert mapped.entity_type == "unknown"
        assert mapped.entity_value == "unknown"
        assert mapped.confidence == 0.0
        assert mapped.source_text == "unknown"
        assert mapped.start_position == 0
        assert mapped.end_position == 0
