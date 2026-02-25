"""Backend package namespace.

This file intentionally avoids eager imports so that CLI help and metadata
commands do not initialize heavy runtime backends (for example MLX/Metal)
before they are actually needed.
"""
