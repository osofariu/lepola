#!/usr/bin/env python3
"""
Setup validation script for AI Legal & Policy Research Assistant.

This script validates that the project structure and dependencies
are correctly set up.
"""

import os
import sys
from pathlib import Path


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and report the result."""
    if os.path.exists(file_path):
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} (missing: {file_path})")
        return False


def check_directory_structure() -> bool:
    """Check that all required directories exist."""
    print("Checking directory structure...")

    required_dirs = [
        "src/core",
        "src/ingestion",
        "src/pipeline",
        "src/querying",
        "src/outputs",
        "tests",
        "examples",
        "data",
        "outputs",
    ]

    all_good = True
    for dir_path in required_dirs:
        if not check_file_exists(dir_path, f"Directory: {dir_path}"):
            all_good = False

    return all_good


def check_core_files() -> bool:
    """Check that all core files are present."""
    print("\nChecking core files...")

    required_files = [
        ("Pipfile", "Dependency management file"),
        ("pytest.ini", "Test configuration"),
        ("ruff.toml", "Code quality configuration"),
        ("README.md", "Project documentation"),
        ("src/main.py", "Main application file"),
        ("src/core/config.py", "Configuration module"),
        ("src/core/models.py", "Data models"),
        ("src/core/logging.py", "Logging setup"),
        ("examples/sample_bill.txt", "Sample document"),
    ]

    all_good = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_good = False

    return all_good


def check_test_files() -> bool:
    """Check that test files are present."""
    print("\nChecking test files...")

    test_files = [
        ("tests/test_config.py", "Configuration tests"),
        ("tests/test_ingestion.py", "Ingestion service tests"),
        ("tests/test_main.py", "Main application tests"),
    ]

    all_good = True
    for file_path, description in test_files:
        if not check_file_exists(file_path, description):
            all_good = False

    return all_good


def check_init_files() -> bool:
    """Check that __init__.py files are present."""
    print("\nChecking Python package structure...")

    package_dirs = [
        "src",
        "src/core",
        "src/ingestion",
        "src/pipeline",
        "src/querying",
        "src/outputs",
    ]

    all_good = True
    for package_dir in package_dirs:
        init_file = f"{package_dir}/__init__.py"
        if not check_file_exists(init_file, f"Package init: {init_file}"):
            all_good = False

    return all_good


def main():
    """Run all validation checks."""
    print("🔍 Validating AI Legal & Policy Research Assistant Setup\n")

    checks = [
        check_directory_structure(),
        check_core_files(),
        check_test_files(),
        check_init_files(),
    ]

    print("\n" + "=" * 60)

    if all(checks):
        print("🎉 All checks passed! Project setup is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pipenv install --dev")
        print("2. Configure environment: copy .env.example to .env")
        print("3. Add your API keys to .env")
        print("4. Run tests: pipenv run test")
        print("5. Start the application: pipenv run start")
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        print("Make sure all required files and directories are present.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
