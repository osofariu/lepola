#!/usr/bin/env python3
"""
Demonstration of JSON parsing configuration based on environment.

This script shows how the JSON parsing settings automatically adapt
based on the environment (development vs production).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import Settings
from src.utils.robust_json_parser import smart_parse_json, parse_json_with_config


def demonstrate_environment_config():
    """Demonstrate how configuration changes based on environment."""

    print("=== JSON Parsing Configuration by Environment ===\n")

    # Test malformed JSON
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

    # Test 1: Development Environment (DEBUG=true)
    print("--- Development Environment (DEBUG=true) ---")
    with patch_environment({"DEBUG": "true"}):
        settings = Settings()
        config = settings.get_json_parsing_config()

        print(f"Environment: {config['environment']}")
        print(f"Strict Mode: {config['strict']}")
        print(f"Robust Parsing: {config['enable_robust_parsing']}")
        print(f"Log Errors: {config['log_errors']}")

        try:
            entities = smart_parse_json(malformed_json)
            print(f"✅ Parsed {len(entities)} entities successfully")
        except Exception as e:
            print(f"❌ Failed: {e}")

    print()

    # Test 2: Production Environment (DEBUG=false)
    print("--- Production Environment (DEBUG=false) ---")
    with patch_environment({"DEBUG": "false"}):
        settings = Settings()
        config = settings.get_json_parsing_config()

        print(f"Environment: {config['environment']}")
        print(f"Strict Mode: {config['strict']}")
        print(f"Robust Parsing: {config['enable_robust_parsing']}")
        print(f"Log Errors: {config['log_errors']}")

        try:
            entities = smart_parse_json(malformed_json)
            print(f"✅ Parsed {len(entities)} entities successfully")
        except Exception as e:
            print(f"❌ Failed: {e}")

    print()

    # Test 3: Custom Configuration
    print("--- Custom Configuration ---")
    custom_config = {
        "strict": True,
        "enable_robust_parsing": True,
        "log_errors": True,
        "max_recovery_attempts": 1,
        "environment": "custom",
    }

    print(f"Environment: {custom_config['environment']}")
    print(f"Strict Mode: {custom_config['strict']}")
    print(f"Robust Parsing: {custom_config['enable_robust_parsing']}")
    print(f"Log Errors: {custom_config['log_errors']}")

    try:
        entities = parse_json_with_config(malformed_json, custom_config)
        print(f"✅ Parsed {len(entities)} entities successfully")
    except Exception as e:
        print(f"❌ Failed: {e}")


def demonstrate_environment_variables():
    """Demonstrate how environment variables override defaults."""

    print("\n=== Environment Variable Overrides ===\n")

    # Test malformed JSON
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

    # Test 1: Production with strict mode disabled
    print("--- Production with JSON_STRICT_MODE=false ---")
    with patch_environment(
        {
            "DEBUG": "false",
            "JSON_STRICT_MODE": "false",
            "JSON_ENABLE_ROBUST_PARSING": "true",
        }
    ):
        settings = Settings()
        config = settings.get_json_parsing_config()

        print(f"Environment: {config['environment']}")
        print(f"Strict Mode: {config['strict']}")
        print(f"Robust Parsing: {config['enable_robust_parsing']}")

        try:
            entities = smart_parse_json(malformed_json)
            print(f"✅ Parsed {len(entities)} entities successfully")
        except Exception as e:
            print(f"❌ Failed: {e}")

    print()

    # Test 2: Development with strict mode enabled
    print("--- Development with JSON_STRICT_MODE=true ---")
    with patch_environment(
        {
            "DEBUG": "true",
            "JSON_STRICT_MODE": "true",
            "JSON_ENABLE_ROBUST_PARSING": "false",
        }
    ):
        settings = Settings()
        config = settings.get_json_parsing_config()

        print(f"Environment: {config['environment']}")
        print(f"Strict Mode: {config['strict']}")
        print(f"Robust Parsing: {config['enable_robust_parsing']}")

        try:
            entities = smart_parse_json(malformed_json)
            print(f"✅ Parsed {len(entities)} entities successfully")
        except Exception as e:
            print(f"❌ Failed: {e}")


def demonstrate_usage_examples():
    """Show practical usage examples."""

    print("\n=== Usage Examples ===\n")

    print("1. Basic Usage (Recommended):")
    print(
        """
from src.utils.robust_json_parser import smart_parse_json

# Automatically uses the best settings for your environment
entities = smart_parse_json(json_string)
"""
    )

    print("\n2. Custom Configuration:")
    print(
        """
from src.utils.robust_json_parser import parse_json_with_config

# Use custom settings
config = {
    "strict": True,
    "enable_robust_parsing": True,
    "log_errors": True,
    "max_recovery_attempts": 5
}
entities = parse_json_with_config(json_string, config)
"""
    )

    print("\n3. Environment Variable Configuration:")
    print(
        """
# In your .env file:
DEBUG=false                    # Production mode
JSON_STRICT_MODE=true         # Strict parsing for performance
JSON_ENABLE_ROBUST_PARSING=false  # Disable robust parsing
JSON_LOG_PARSING_ERRORS=true  # Log errors for monitoring

# Or for development:
DEBUG=true                     # Development mode  
JSON_STRICT_MODE=false        # Lenient parsing for debugging
JSON_ENABLE_ROBUST_PARSING=true   # Enable robust parsing
JSON_LOG_PARSING_ERRORS=true  # Log errors for debugging
"""
    )


class patch_environment:
    """Context manager to temporarily patch environment variables."""

    def __init__(self, env_vars: dict):
        self.env_vars = env_vars
        self.original_env = {}

    def __enter__(self):
        # Store original values
        for key, value in self.env_vars.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for key, original_value in self.original_env.items():
            if original_value is None:
                del os.environ[key]
            else:
                os.environ[key] = original_value


if __name__ == "__main__":
    demonstrate_environment_config()
    demonstrate_environment_variables()
    demonstrate_usage_examples()
