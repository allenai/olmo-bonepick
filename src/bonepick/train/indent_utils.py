"""Utilities for indentation detection and conversion."""

from math import gcd


def has_space_indentation(text: str, sample_size: int = 4096) -> bool:
    """
    Fast heuristic to check if text likely has space indentation.

    Scans the first `sample_size` bytes looking for a newline followed by 2+ spaces.
    This is a quick pre-check to avoid running full detection on files that:
    - Have no indentation
    - Use tab indentation

    False positives are acceptable (we'll just run detection and find nothing).
    False negatives are NOT acceptable (we must not skip files with space indentation).

    Args:
        text: The text content to check
        sample_size: How many characters to scan (default 4KB)

    Returns:
        True if the text likely has space indentation, False otherwise
    """
    # Scan first sample_size chars for "\n  " (newline + 2 spaces)
    # This catches any space indentation of 2 or more
    sample = text[:sample_size]

    # Check for newline followed by 2+ spaces (indented line)
    # Also check start of text in case first line is indented
    if sample.startswith("  "):
        return True

    return "\n  " in sample


def detect_indentation(text: str) -> int | None:
    """
    Detect the indentation multiplier (e.g., 2, 4) for space-indented text.

    The algorithm:
    1. Extract leading spaces from each line
    2. Compute the GCD of all non-zero indentation levels
    3. The GCD is the indentation unit

    This is O(n) where n is the text length, with a single pass through the text.

    Args:
        text: The text content to analyze

    Returns:
        The detected indentation size (e.g., 2 or 4), or None if no indentation found
    """
    result_gcd = 0

    for line in text.split("\n"):
        # Count leading spaces efficiently
        stripped = line.lstrip(" ")
        if stripped and stripped[0] != "\t":  # Skip empty lines and tab-indented
            leading_spaces = len(line) - len(stripped)
            if leading_spaces > 0:
                result_gcd = gcd(result_gcd, leading_spaces)
                # Early exit: can't get smaller than 1
                if result_gcd == 1:
                    return 1

    return result_gcd if result_gcd > 0 else None


def convert_spaces_to_tabs(text: str, indent_size: int | None = None) -> str:
    """
    Convert space indentation to tabs.

    If indent_size is not provided, it will be auto-detected.
    Only converts leading spaces; spaces within lines are preserved.

    Args:
        text: The text content to convert
        indent_size: The number of spaces per indent level (auto-detected if None)

    Returns:
        Text with leading spaces converted to tabs
    """
    if indent_size is None:
        indent_size = detect_indentation(text)
        if indent_size is None:
            return text  # No indentation detected, return as-is

    lines = text.split("\n")
    result = []

    for line in lines:
        if not line or line[0] != " ":
            result.append(line)
            continue

        # Count leading spaces
        stripped = line.lstrip(" ")
        leading_spaces = len(line) - len(stripped)

        # Convert to tabs + remainder spaces
        num_tabs = leading_spaces // indent_size
        remainder = leading_spaces % indent_size
        result.append("\t" * num_tabs + " " * remainder + stripped)

    return "\n".join(result)
