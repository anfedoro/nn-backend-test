"""Shared backend constants and errors.

This module is intentionally tiny and import-safe for both runtime and backends.
"""

SUPPORTED_QUANT_BITS = {2, 3, 4, 5, 6, 8}
QUANT_GROUP_SIZE_PREFERRED = 64
QUANT_GROUP_SIZE_FALLBACK = 32
EXACT_KERNEL_OPS_PER_ELEMENT = 5


class UnsupportedFeatureError(RuntimeError):
    """Raised when requested backend feature is unavailable."""
