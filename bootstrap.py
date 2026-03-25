"""Runtime bootstrap for packaged CLI entrypoints.

This module keeps CLI startup lightweight while handling Linux-specific CUDA
runtime library discovery for MLX wheels that vendor NVIDIA shared objects
inside ``site-packages/nvidia/*/lib``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def discover_linux_cuda_lib_dirs():
    """Return bundled NVIDIA library directories visible on sys.path."""
    lib_dirs = []
    seen = set()
    for entry in sys.path:
        if not entry:
            continue
        base = Path(entry)
        if not base.is_dir():
            continue
        nvidia_root = base / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            resolved = lib_dir.resolve()
            if resolved.is_dir() and resolved not in seen:
                seen.add(resolved)
                lib_dirs.append(resolved)
    return lib_dirs


def extend_ld_library_path(lib_dirs):
    """Prepend bundled NVIDIA library directories to LD_LIBRARY_PATH."""
    existing = [Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged = []
    seen = set()
    for path in [*lib_dirs, *existing]:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            merged.append(str(resolved))
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def preload_linux_cuda_libs(lib_dirs):
    """Preload bundled CUDA libraries so MLX can resolve shared deps."""
    load_mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)
    load_patterns = [
        "libcublas.so*",
        "libcublasLt.so*",
        "libnvrtc.so*",
        "libcudnn*.so*",
        "libnccl.so*",
    ]
    loaded = set()
    for pattern in load_patterns:
        for lib_dir in lib_dirs:
            for candidate in sorted(lib_dir.glob(pattern)):
                if candidate.is_dir():
                    continue
                lib_name = candidate.name
                if lib_name in loaded:
                    continue
                ctypes.CDLL(str(candidate), mode=load_mode)
                loaded.add(lib_name)


def configure_linux_cuda_runtime():
    """Wire bundled CUDA libraries into the current process on Linux."""
    if sys.platform != "linux":
        return
    lib_dirs = discover_linux_cuda_lib_dirs()
    if not lib_dirs:
        return
    extend_ld_library_path(lib_dirs)
    preload_linux_cuda_libs(lib_dirs)


def cli():
    """CLI entrypoint that prepares runtime dependencies before importing MLX."""
    configure_linux_cuda_runtime()
    from main import cli as main_cli

    return main_cli()
