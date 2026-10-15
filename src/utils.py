"""Small shared helpers."""

from __future__ import annotations


def normalize_path(path: str) -> str:
    """Normalize a path for consistent cache matching across platforms.

    Converts to forward slashes and lowercases, so the same file produces the
    same cache key regardless of OS or of how the path was written.

    Args:
        path: File path string.

    Returns:
        Normalized path, or "" for an empty input.
    """
    if not path:
        return ""
    return str(path).replace("\\", "/").lower()