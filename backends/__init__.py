"""Public exports for benchmark backend modules.

main.py imports backend classes and shared constants from this package-level
module to keep runtime code concise and easy to scan.
"""

from .common import (
    QUANT_GROUP_SIZE_FALLBACK,
    QUANT_GROUP_SIZE_PREFERRED,
    SUPPORTED_QUANT_BITS,
    TORCH_INT_MATMUL_DTYPE_TOKENS,
    UnsupportedFeatureError,
)
from .mlx_backend import MlxBackend
from .torch_backend import TorchBackend, resolve_torch_dtype, try_import_torch

__all__ = [
    "MlxBackend",
    "TorchBackend",
    "UnsupportedFeatureError",
    "SUPPORTED_QUANT_BITS",
    "QUANT_GROUP_SIZE_PREFERRED",
    "QUANT_GROUP_SIZE_FALLBACK",
    "TORCH_INT_MATMUL_DTYPE_TOKENS",
    "try_import_torch",
    "resolve_torch_dtype",
]
