#!/usr/bin/env python3
"""
Comprehensive performance benchmark for strict vs lenient JSON parsing.

This script demonstrates the real performance impact of using robust parsing
with larger datasets and more complex scenarios.
"""

import sys
import os
import time
import json
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.robust_json_parser import RobustJSONParser


def generate_test_data(size: int, malformed_ratio: float = 0.0) -> str:
    """
    Generate test JSON data with specified size and malformed ratio.

    Args:
        size: Number of entities to generate
        malformed_ratio: Ratio of malformed entities (0.0 to 1.0)

    Returns:
        JSON string with test data
    """
    entities = []

    entity_types = ["agency", "legal_concept", "affected_group", "timeline", "penalty"]
    values = [
        "Secretary of Agriculture",
        "excess shelter expense deduction",
        "elderly or disabled member",
        "fiscal year 2028",
        "payment error rate",
    ]

    for i in range(size):
        # Randomly decide if this entity should be malformed
        is_malformed = random.random() < malformed_ratio

        entity = {
            "Type": random.choice(entity_types),
            "Value": random.choice(values),
            "Confidence": round(random.uniform(0.8, 1.0), 2),
            "Start": random.randint(1, 100),
            "End": random.randint(101, 200),
        }

        if is_malformed:
            # Add malformed source with unescaped quotes
            entity["Source"] = (
                f"with an elderly or disabled member'' after ''household' {i}"
            )
        else:
            entity["Source"] = f"normal source text for entity {i}"

        entities.append(entity)

    return json.dumps(entities, indent=2)


def benchmark_parsing(json_str: str, iterations: int = 100) -> dict:
    """
    Benchmark parsing performance for both modes.

    Args:
        json_str: JSON string to parse
        iterations: Number of iterations to run

    Returns:
        Dictionary with timing results
    """
    strict_parser = RobustJSONParser(strict=True)
    lenient_parser = RobustJSONParser(strict=False)

    results = {
        "strict": {"times": [], "success_count": 0, "total_time": 0},
        "lenient": {"times": [], "success_count": 0, "total_time": 0},
    }

    # Benchmark strict mode
    print("Benchmarking strict mode...")
    for i in range(iterations):
        start_time = time.time()
        try:
            result = strict_parser.parse_json_array(json_str)
            parse_time = time.time() - start_time
            results["strict"]["times"].append(parse_time)
            results["strict"]["success_count"] += 1
        except Exception:
            results["strict"]["times"].append(0)

    # Benchmark lenient mode
    print("Benchmarking lenient mode...")
    for i in range(iterations):
        start_time = time.time()
        try:
            result = lenient_parser.parse_json_array(json_str)
            parse_time = time.time() - start_time
            results["lenient"]["times"].append(parse_time)
            results["lenient"]["success_count"] += 1
        except Exception:
            results["lenient"]["times"].append(0)

    # Calculate statistics
    for mode in ["strict", "lenient"]:
        times = [t for t in results[mode]["times"] if t > 0]
        if times:
            results[mode]["avg_time"] = sum(times) / len(times)
            results[mode]["min_time"] = min(times)
            results[mode]["max_time"] = max(times)
            results[mode]["total_time"] = sum(times)
        else:
            results[mode]["avg_time"] = 0
            results[mode]["min_time"] = 0
            results[mode]["max_time"] = 0
            results[mode]["total_time"] = 0

    return results


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmarks."""

    print("=== Comprehensive Performance Benchmark ===\n")

    test_scenarios = [
        {"size": 10, "malformed": 0.0, "name": "Small, Clean Data"},
        {"size": 100, "malformed": 0.0, "name": "Medium, Clean Data"},
        {"size": 1000, "malformed": 0.0, "name": "Large, Clean Data"},
        {"size": 10, "malformed": 0.3, "name": "Small, Some Malformed"},
        {"size": 100, "malformed": 0.3, "name": "Medium, Some Malformed"},
        {"size": 1000, "malformed": 0.3, "name": "Large, Some Malformed"},
        {"size": 10, "malformed": 1.0, "name": "Small, All Malformed"},
        {"size": 100, "malformed": 1.0, "name": "Medium, All Malformed"},
    ]

    for scenario in test_scenarios:
        print(
            f"--- {scenario['name']} ({scenario['size']} entities, {scenario['malformed']*100}% malformed) ---"
        )

        # Generate test data
        json_str = generate_test_data(scenario["size"], scenario["malformed"])

        # Run benchmark
        results = benchmark_parsing(json_str, iterations=50)

        # Display results
        print(f"Strict Mode:")
        print(f"  Success Rate: {results['strict']['success_count']}/50")
        print(f"  Avg Time: {results['strict']['avg_time']*1000:.2f}ms")
        print(f"  Min Time: {results['strict']['min_time']*1000:.2f}ms")
        print(f"  Max Time: {results['strict']['max_time']*1000:.2f}ms")

        print(f"Lenient Mode:")
        print(f"  Success Rate: {results['lenient']['success_count']}/50")
        print(f"  Avg Time: {results['lenient']['avg_time']*1000:.2f}ms")
        print(f"  Min Time: {results['lenient']['min_time']*1000:.2f}ms")
        print(f"  Max Time: {results['lenient']['max_time']*1000:.2f}ms")

        if results["strict"]["avg_time"] > 0 and results["lenient"]["avg_time"] > 0:
            speedup = results["lenient"]["avg_time"] / results["strict"]["avg_time"]
            print(f"Performance Ratio: {speedup:.1f}x slower in lenient mode")
        else:
            print("Performance Ratio: N/A (one mode failed completely)")

        print()


def demonstrate_memory_usage():
    """Demonstrate memory usage differences."""

    print("=== Memory Usage Demonstration ===\n")

    # Generate a large dataset
    large_json = generate_test_data(5000, malformed_ratio=0.1)
    print(f"Generated {len(large_json)} characters of JSON data")

    # Test memory usage by parsing multiple times
    strict_parser = RobustJSONParser(strict=True)
    lenient_parser = RobustJSONParser(strict=False)

    print("\nParsing with strict mode...")
    start_time = time.time()
    for i in range(10):
        try:
            result = strict_parser.parse_json_array(large_json)
        except Exception:
            pass
    strict_time = time.time() - start_time

    print("Parsing with lenient mode...")
    start_time = time.time()
    for i in range(10):
        result = lenient_parser.parse_json_array(large_json)
    lenient_time = time.time() - start_time

    print(f"\nStrict mode total time: {strict_time:.2f}s")
    print(f"Lenient mode total time: {lenient_time:.2f}s")
    print(f"Time difference: {lenient_time/strict_time:.1f}x slower in lenient mode")


if __name__ == "__main__":
    run_comprehensive_benchmark()
    demonstrate_memory_usage()
