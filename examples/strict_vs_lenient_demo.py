#!/usr/bin/env python3
"""
Demonstration of strict vs lenient JSON parsing modes.

This script shows the performance and behavior differences between
strict and lenient parsing modes.
"""

import sys
import os
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.robust_json_parser import RobustJSONParser


def performance_comparison():
    """Compare performance between strict and lenient modes."""

    # Valid JSON (should be fast in both modes)
    valid_json = """[
  {
    "Type": "agency",
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

    # Malformed JSON (will be slow in lenient mode)
    malformed_json = """[
  {
    "Type": "affected_group",
    "Value": "elderly or disabled member",
    "Confidence": 0.95,
    "Source": "with an elderly or disabled member'' after ''household'",
    "Start": 16,
    "End": 31
  }
]"""

    print("=== Performance Comparison ===")

    # Test valid JSON
    print("\n--- Valid JSON ---")

    # Strict mode
    start_time = time.time()
    strict_parser = RobustJSONParser(strict=True)
    try:
        result = strict_parser.parse_json_array(valid_json)
        strict_time = time.time() - start_time
        print(f"✅ Strict mode: {len(result)} entities in {strict_time:.4f}s")
    except Exception as e:
        print(f"❌ Strict mode failed: {e}")

    # Lenient mode
    start_time = time.time()
    lenient_parser = RobustJSONParser(strict=False)
    result = lenient_parser.parse_json_array(valid_json)
    lenient_time = time.time() - start_time
    print(f"✅ Lenient mode: {len(result)} entities in {lenient_time:.4f}s")

    print(
        f"Performance difference: {lenient_time/strict_time:.1f}x slower in lenient mode"
    )

    # Test malformed JSON
    print("\n--- Malformed JSON ---")

    # Strict mode
    start_time = time.time()
    try:
        result = strict_parser.parse_json_array(malformed_json)
        strict_time = time.time() - start_time
        print(f"✅ Strict mode: {len(result)} entities in {strict_time:.4f}s")
    except Exception as e:
        strict_time = time.time() - start_time
        print(f"❌ Strict mode failed in {strict_time:.4f}s: {e}")

    # Lenient mode
    start_time = time.time()
    result = lenient_parser.parse_json_array(malformed_json)
    lenient_time = time.time() - start_time
    print(f"✅ Lenient mode: {len(result)} entities in {lenient_time:.4f}s")

    if "strict_time" in locals():
        print(
            f"Performance difference: {lenient_time/strict_time:.1f}x slower in lenient mode"
        )


def behavior_comparison():
    """Compare behavior differences between modes."""

    print("\n=== Behavior Comparison ===")

    # Test case with multiple issues
    problematic_json = """[
  {
    "Type": "agency",
    "Value": "Secretary of Agriculture",
    "Confidence": 0.95,
    "Source": "The Secretary of Agriculture",
    "Start": 7,
    "End": 20
  },
  {
    "Type": "affected_group",
    "Value": "elderly or disabled member",
    "Confidence": 0.95,
    "Source": "with an elderly or disabled member'' after ''household'",
    "Start": 16,
    "End": 31
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

    print("\n--- Strict Mode Behavior ---")
    strict_parser = RobustJSONParser(strict=True)
    try:
        result = strict_parser.parse_json_array(problematic_json)
        print(f"✅ Parsed {len(result)} entities")
    except Exception as e:
        print(f"❌ Failed completely: {e}")
        print("   → No data recovered")

    print("\n--- Lenient Mode Behavior ---")
    lenient_parser = RobustJSONParser(strict=False)
    result = lenient_parser.parse_json_array(problematic_json)
    print(f"✅ Parsed {len(result)} entities")
    print("   → Partial data recovered")

    for i, entity in enumerate(result):
        print(f"   Entity {i}: {entity['Type']} - {entity['Value']}")


def use_case_examples():
    """Show different use cases for each mode."""

    print("\n=== Use Case Examples ===")

    print("\n--- Production Environment (Strict Mode) ---")
    print("When to use strict mode:")
    print("• High-volume data processing")
    print("• Trusted data sources")
    print("• Performance-critical applications")
    print("• Early error detection")

    print("\n--- Development/Debugging (Lenient Mode) ---")
    print("When to use lenient mode:")
    print("• Processing legacy data")
    print("• Dealing with untrusted sources")
    print("• Data recovery scenarios")
    print("• Development and testing")

    print("\n--- Hybrid Approach ---")
    print("Best practice: Try strict first, fall back to lenient")
    print(
        """
# Example hybrid approach
try:
    entities = parser.parse_json_array(json_str, strict=True)
except ValueError:
    # Log the issue for monitoring
    logger.warning("Data quality issue detected, attempting recovery")
    # Fall back to lenient mode
    entities = parser.parse_json_array(json_str, strict=False)
    # Potentially alert data source about quality issues
"""
    )


if __name__ == "__main__":
    performance_comparison()
    behavior_comparison()
    use_case_examples()
