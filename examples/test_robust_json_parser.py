#!/usr/bin/env python3
"""
Test script for the robust JSON parser.

This script demonstrates how the robust JSON parser can handle various
types of malformed JSON and extract valid entries.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.robust_json_parser import (
    RobustJSONParser,
    parse_entities_json,
    validate_entity_structure,
)


def test_robust_parser():
    """Test the robust JSON parser with various malformed JSON examples."""

    # Test case 1: JSON with unescaped single quotes (the original issue)
    malformed_json_1 = """[
  {
    "Type": "legal_document",
    "Value": "HR 1 EH1S",
    "Confidence": 0.95,
    "Source": "•HR 1 EH1S SEC.",
    "Start": 19,
    "End": 24
  },
  {
    "Type": "affected_group",
    "Value": "elderly or disabled member",
    "Confidence": 0.95,
    "Source": "with an elderly or disabled member'' after ''household'",
    "Start": 16,
    "End": 31
  }
]"""

    print("=== Test 1: JSON with unescaped single quotes ===")
    try:
        # This should fail with standard JSON parser
        import json

        json.loads(malformed_json_1)
        print("❌ Standard JSON parser should have failed!")
    except json.JSONDecodeError as e:
        print(f"✅ Standard JSON parser correctly failed: {e}")

    # This should work with robust parser
    try:
        entities = parse_entities_json(malformed_json_1)
        print(f"✅ Robust parser succeeded! Found {len(entities)} entities")
        for i, entity in enumerate(entities):
            print(f"  Entity {i}: {entity['Type']} - {entity['Value']}")
    except Exception as e:
        print(f"❌ Robust parser failed: {e}")

    print()

    # Test case 2: JSON with missing commas
    malformed_json_2 = """[
  {
    "Type": "agency"
    "Value": "Secretary of Agriculture",
    "Confidence": 0.95,
    "Source": "The Secretary of Agriculture",
    "Start": 7,
    "End": 20
  },
  {
    "Type": "legal_concept",
    "Value": "excess shelter expense deduction",
    "Confidence": 0.95,
    "Source": "computing the excess shelter expense deduction",
    "Start": 6,
    "End": 28
  }
]"""

    print("=== Test 2: JSON with missing commas ===")
    try:
        entities = parse_entities_json(malformed_json_2)
        print(f"✅ Robust parser succeeded! Found {len(entities)} entities")
        for i, entity in enumerate(entities):
            print(f"  Entity {i}: {entity['Type']} - {entity['Value']}")
    except Exception as e:
        print(f"❌ Robust parser failed: {e}")

    print()

    # Test case 3: JSON with mixed issues
    malformed_json_3 = """[
  {
    "Type": "timeline",
    "Value": "fiscal year 2028",
    "Confidence": 0.95,
    "Source": "Beginning in fiscal year 2028",
    "Start": 1,
    "End": 11
  },
  {
    "Type": "penalty",
    "Value": "payment error rate",
    "Confidence": 0.95,
    "Source": "State that has a payment error rate",
    "Start": 1,
    "End": 15
  },
  {
    "Type": "agency",
    "Value": "Census Bureau",
    "Confidence": 0.95,
    "Source": "as recognized by the Census Bureau",
    "Start": 20,
    "End": 28
  }
]"""

    print("=== Test 3: Valid JSON (should work with both parsers) ===")
    try:
        entities = parse_entities_json(malformed_json_3)
        print(f"✅ Robust parser succeeded! Found {len(entities)} entities")
        for i, entity in enumerate(entities):
            print(f"  Entity {i}: {entity['Type']} - {entity['Value']}")
    except Exception as e:
        print(f"❌ Robust parser failed: {e}")

    print()

    # Test case 4: Entity validation
    print("=== Test 4: Entity validation ===")
    valid_entity = {
        "Type": "agency",
        "Value": "Secretary of Agriculture",
        "Confidence": 0.95,
        "Source": "The Secretary of Agriculture",
        "Start": 7,
        "End": 20,
    }

    invalid_entity = {
        "Type": "agency",
        "Value": "Secretary of Agriculture",
        "Confidence": 0.95,
        # Missing Source, Start, End
    }

    print(f"Valid entity: {validate_entity_structure(valid_entity)}")
    print(f"Invalid entity: {validate_entity_structure(invalid_entity)}")


def test_with_file():
    """Test the robust parser with the actual bad_json.json file."""
    print("\n=== Test with actual bad_json.json file ===")

    try:
        with open("bad_json.json", "r") as f:
            json_content = f.read()

        entities = parse_entities_json(json_content)
        print(f"✅ Successfully parsed {len(entities)} entities from bad_json.json")

        # Validate all entities
        valid_entities = [e for e in entities if validate_entity_structure(e)]
        print(f"✅ {len(valid_entities)} out of {len(entities)} entities are valid")

        # Show some examples
        for i, entity in enumerate(valid_entities[:3]):
            print(f"  Entity {i}: {entity['Type']} - {entity['Value']}")

    except FileNotFoundError:
        print("❌ bad_json.json file not found")
    except Exception as e:
        print(f"❌ Failed to parse bad_json.json: {e}")


if __name__ == "__main__":
    test_robust_parser()
    test_with_file()
