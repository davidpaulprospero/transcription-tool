"""Structured error logging helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

# Optional process-wide correlation ID. The host pipeline sets this so that
# every line emitted while handling one request can be tied together; this
# module works fine without it.
_correlation_id: Optional[str] = None


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set the correlation ID applied to subsequent log lines."""
    global _correlation_id
    _correlation_id = correlation_id


def get_correlation_id() -> Optional[str]:
    """Return the current correlation ID, if any."""
    return _correlation_id


def _format_correlation(correlation_id: Optional[str]) -> str:
    """Format a correlation ID into a log prefix."""
    if correlation_id:
        return f"[corr:{correlation_id}]"
    return ""


def log_error_with_context(
    logger: logging.Logger,
    error_code: str,
    message: str,
    correlation_id: Optional[str] = None,
    **context: Any,
) -> None:
    """Log an error with a standardized code, correlation ID and context.

    Args:
        logger: The logger instance.
        error_code: Error code identifying the failure site.
        message: Human-readable description of the failure.
        correlation_id: Overrides the process-wide correlation ID.
        **context: Extra key=value pairs appended to the line.
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)
    base_msg = f"[{error_code}]{corr} {message}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.error(f"{base_msg} - {context_str}")
    else:
        logger.error(base_msg)