"""
Robust JSON parsing utilities for handling malformed JSON data.

This module provides utilities to parse JSON data that may contain formatting issues,
allowing extraction of valid entries even when some entries are corrupted.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class RobustJSONParser:
    """
    A robust JSON parser that can handle malformed JSON and extract valid entries.

    This parser is designed to work with JSON data that may contain:
    - Unescaped quotes within strings
    - Missing commas
    - Extra commas
    - Malformed individual objects
    - Other common JSON formatting issues
    """

    def __init__(self, strict: bool = False):
        """
        Initialize the robust JSON parser.

        Args:
            strict: If True, raises exceptions on parsing errors. If False,
                   attempts to recover and continue parsing.
        """
        self.strict = strict

    def parse_json_array(self, json_str: str) -> List[Dict[str, Any]]:
        """
        Parse a JSON array string, attempting to extract all valid objects.

        Args:
            json_str: The JSON string to parse

        Returns:
            List of successfully parsed JSON objects

        Raises:
            ValueError: If strict mode is enabled and parsing fails
        """
        try:
            # First, try standard JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.strict:
                raise ValueError(f"JSON parsing failed in strict mode: {e}")

            logger.warning(
                f"Standard JSON parsing failed: {e}. Attempting robust parsing..."
            )
            return self._robust_parse_array(json_str)

    def _robust_parse_array(self, json_str: str) -> List[Dict[str, Any]]:
        """
        Robustly parse a JSON array by extracting individual objects.

        Args:
            json_str: The JSON string to parse

        Returns:
            List of successfully parsed JSON objects
        """
        # Clean up the string
        cleaned_str = self._clean_json_string(json_str)

        # Extract individual JSON objects
        objects = self._extract_json_objects(cleaned_str)

        valid_objects = []
        for i, obj_str in enumerate(objects):
            try:
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse object {i}: {e}")
                # Try to fix common issues and retry
                fixed_obj_str = self._fix_common_issues(obj_str)
                try:
                    obj = json.loads(fixed_obj_str)
                    valid_objects.append(obj)
                    logger.info(f"Successfully parsed object {i} after fixing issues")
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse object {i} even after fixing: {obj_str[:100]}..."
                    )

        return valid_objects

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean up common JSON formatting issues.

        Args:
            json_str: The JSON string to clean

        Returns:
            Cleaned JSON string
        """
        # Remove leading/trailing whitespace
        json_str = json_str.strip()

        # Ensure it starts and ends with brackets
        if not json_str.startswith("["):
            json_str = "[" + json_str
        if not json_str.endswith("]"):
            json_str = json_str + "]"

        return json_str

    def _extract_json_objects(self, json_str: str) -> List[str]:
        """
        Extract individual JSON objects from an array string.

        Args:
            json_str: The JSON array string

        Returns:
            List of individual JSON object strings
        """
        # Remove the outer brackets
        content = json_str[1:-1].strip()

        objects = []
        current_obj = ""
        brace_count = 0
        in_string = False
        escape_next = False

        for char in content:
            if escape_next:
                current_obj += char
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                current_obj += char
                continue

            if char == '"' and not escape_next:
                in_string = not in_string

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        current_obj += char
                        objects.append(current_obj.strip())
                        current_obj = ""
                        continue

            current_obj += char

        return [obj for obj in objects if obj.strip()]

    def _fix_common_issues(self, obj_str: str) -> str:
        """
        Fix common JSON formatting issues in an object string.

        Args:
            obj_str: The JSON object string to fix

        Returns:
            Fixed JSON object string
        """
        # Fix unescaped single quotes within double-quoted strings
        obj_str = self._fix_unescaped_quotes(obj_str)

        # Fix missing quotes around property names
        obj_str = self._fix_property_names(obj_str)

        # Fix missing commas between properties
        obj_str = self._fix_missing_commas(obj_str)

        # Fix trailing commas
        obj_str = re.sub(r",\s*}", "}", obj_str)

        return obj_str

    def _fix_unescaped_quotes(self, obj_str: str) -> str:
        """
        Fix unescaped quotes within JSON strings.

        Args:
            obj_str: The JSON object string to fix

        Returns:
            Fixed JSON object string
        """
        # Pattern to match strings with unescaped quotes
        # This is a simplified approach - for production use, consider using a proper JSON parser
        pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'

        def replace_quotes(match):
            content = match.group(1)
            # Escape single quotes that aren't already escaped
            content = re.sub(r"(?<!\\)'", r"\\'", content)
            return f'"{content}"'

        return re.sub(pattern, replace_quotes, obj_str)

    def _fix_property_names(self, obj_str: str) -> str:
        """
        Fix property names that might be missing quotes.

        Args:
            obj_str: The JSON object string to fix

        Returns:
            Fixed JSON object string
        """
        # Pattern to match unquoted property names
        pattern = r"(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)"

        def quote_property(match):
            return f'{match.group(1)}"{match.group(2)}"{match.group(3)}'

        return re.sub(pattern, quote_property, obj_str)

    def _fix_missing_commas(self, obj_str: str) -> str:
        """
        Fix missing commas between JSON properties.

        Args:
            obj_str: The JSON object string to fix

        Returns:
            Fixed JSON object string
        """
        # Pattern to match property-value pairs that are missing commas
        # Look for: "property": value followed by "property": value (missing comma)
        pattern = r'("(?:[^"\\]|\\.)*"\s*:\s*(?:"(?:[^"\\]|\\.)*"|(?:true|false|null|\d+(?:\.\d+)?)))\s*("(?:[^"\\]|\\.)*"\s*:)'

        def add_comma(match):
            return f"{match.group(1)},{match.group(2)}"

        return re.sub(pattern, add_comma, obj_str)


def parse_entities_json(json_str: str, strict: bool = False) -> List[Dict[str, Any]]:
    """
    Parse entities from JSON string with robust error handling.

    Args:
        json_str: The JSON string containing entities
        strict: If True, raises exceptions on parsing errors

    Returns:
        List of successfully parsed entity objects

    Raises:
        ValueError: If strict mode is enabled and parsing fails
    """
    parser = RobustJSONParser(strict=strict)
    return parser.parse_json_array(json_str)


def safe_parse_json(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON with fallback to robust parsing.

    This is a simple utility function that tries standard JSON parsing first,
    and falls back to robust parsing if that fails.

    Args:
        json_str: The JSON string to parse
        default: Default value to return if parsing fails completely

    Returns:
        Parsed JSON object or default value
    """
    try:
        # Try standard JSON parsing first
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Fall back to robust parsing
            return parse_entities_json(json_str)
        except Exception as e:
            logger.error(f"Both standard and robust JSON parsing failed: {e}")
            return default


def validate_entity_structure(entity: Dict[str, Any]) -> bool:
    """
    Validate that an entity has the required structure.

    Args:
        entity: The entity dictionary to validate

    Returns:
        True if the entity is valid, False otherwise
    """
    required_fields = ["Type", "Value", "Confidence", "Source", "Start", "End"]

    for field in required_fields:
        if field not in entity:
            return False

    # Validate data types
    if not isinstance(entity["Type"], str):
        return False
    if not isinstance(entity["Value"], str):
        return False
    if not isinstance(entity["Confidence"], (int, float)):
        return False
    if not isinstance(entity["Source"], str):
        return False
    if not isinstance(entity["Start"], int):
        return False
    if not isinstance(entity["End"], int):
        return False

    return True


def filter_valid_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out invalid entities from a list.

    Args:
        entities: List of entity dictionaries

    Returns:
        List of valid entity dictionaries
    """
    return [entity for entity in entities if validate_entity_structure(entity)]


def parse_json_with_config(
    json_str: str, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Parse JSON using configuration-aware settings.

    This function automatically determines the appropriate parsing mode based on
    the provided configuration or environment settings.

    Args:
        json_str: The JSON string to parse
        config: Optional configuration dict. If None, uses global settings.

    Returns:
        List of successfully parsed JSON objects

    Raises:
        ValueError: If strict mode is enabled and parsing fails
    """
    if config is None:
        # Import here to avoid circular imports
        from src.core.config import settings

        config = settings.get_json_parsing_config()

    strict_mode = config.get("strict", False)
    enable_robust = config.get("enable_robust_parsing", True)
    log_errors = config.get("log_errors", True)
    max_attempts = config.get("max_recovery_attempts", 3)
    environment = config.get("environment", "development")

    if log_errors:
        logger.info(
            f"Parsing JSON in {environment} mode (strict={strict_mode}, robust={enable_robust})"
        )

    # If robust parsing is disabled, use standard JSON parsing
    if not enable_robust:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if strict_mode:
                raise ValueError(f"JSON parsing failed in strict mode: {e}")
            else:
                logger.warning(
                    f"JSON parsing failed and robust parsing is disabled: {e}"
                )
                return []

    # Use the robust parser with the configured strict mode
    parser = RobustJSONParser(strict=strict_mode)

    # Try parsing with the configured number of recovery attempts
    for attempt in range(max_attempts):
        try:
            return parser.parse_json_array(json_str)
        except Exception as e:
            if attempt < max_attempts - 1:
                if log_errors:
                    logger.warning(f"JSON parsing attempt {attempt + 1} failed: {e}")
                continue
            else:
                if log_errors:
                    logger.error(f"All JSON parsing attempts failed: {e}")
                if strict_mode:
                    raise ValueError(
                        f"JSON parsing failed after {max_attempts} attempts: {e}"
                    )
                else:
                    return []


def smart_parse_json(json_str: str) -> List[Dict[str, Any]]:
    """
    Smart JSON parsing that automatically uses the best approach for the environment.

    This is the recommended function to use in your application. It automatically:
    - Uses strict mode in production for performance and data quality
    - Uses lenient mode in development for debugging and recovery
    - Logs appropriate messages for monitoring
    - Handles errors gracefully

    Args:
        json_str: The JSON string to parse

    Returns:
        List of successfully parsed JSON objects
    """
    return parse_json_with_config(json_str)
