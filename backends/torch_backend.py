"""Torch backend implementation.

This module contains Torch-specific dtype/device resolution, synchronization,
tensor preparation, and per-iteration execution utilities.
"""

from .common import UnsupportedFeatureError


def try_import_torch():
    """Import torch if available, otherwise return None for graceful fallback."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        return None
    return torch


def resolve_torch_dtype(torch_module, dtype_token):
    """Map CLI dtype token to Torch dtype object."""
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "float64": torch_module.float64,
        "int8": torch_module.int8,
        "int16": torch_module.int16,
        "int32": torch_module.int32,
        "int64": torch_module.int64,
        "uint8": torch_module.uint8,
        "bool": torch_module.bool,
        "complex64": torch_module.complex64,
        "complex128": torch_module.complex128,
    }
    return mapping.get(dtype_token)


def select_torch_device(torch_module, requested_device):
    """Resolve Torch device according to logical CLI device selection."""
    if requested_device == "cpu":
        return torch_module.device("cpu")
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    raise UnsupportedFeatureError("Torch GPU backend is unavailable. Use --device cpu or --backend mlx.")


def sync_torch_device(torch_module, torch_device):
    """Synchronize asynchronous Torch backends before stopping timers."""
    if torch_device.type == "mps":
        torch_module.mps.synchronize()
    elif torch_device.type == "cuda":
        torch_module.cuda.synchronize()


def configure_torch_cpu_threads(torch_module, cpu_workers):
    """Apply CPU threading limits for fair cross-backend comparison."""
    torch_module.set_num_threads(cpu_workers)
    try:
        torch_module.set_num_interop_threads(cpu_workers)
    except RuntimeError:
        pass


def format_torch_device_summary(torch_module, torch_device):
    """Build readable Torch device summary for console output."""
    if torch_device.type == "cuda":
        return f"{torch_device} [CUDA] ({torch_module.cuda.get_device_name(torch_device)})"
    if torch_device.type == "mps":
        return f"{torch_device} [Metal]"
    return f"{torch_device} [CPU]"


def make_torch_inputs(n, torch_module, torch_dtype, is_inexact, torch_device):
    """Create two Torch input matrices matching requested dtype behavior."""
    if torch_dtype == torch_module.bool:
        a = torch_module.randint(0, 2, (n, n), dtype=torch_module.int32, device=torch_device).to(torch_module.bool)
        b = torch_module.randint(0, 2, (n, n), dtype=torch_module.int32, device=torch_device).to(torch_module.bool)
    elif is_inexact:
        a = torch_module.rand((n, n), dtype=torch_module.float32, device=torch_device).to(torch_dtype)
        b = torch_module.rand((n, n), dtype=torch_module.float32, device=torch_device).to(torch_dtype)
    elif torch_dtype == torch_module.uint8:
        a = torch_module.randint(0, 127, (n, n), dtype=torch_dtype, device=torch_device)
        b = torch_module.randint(0, 127, (n, n), dtype=torch_dtype, device=torch_device)
    else:
        a = torch_module.randint(-64, 64, (n, n), dtype=torch_dtype, device=torch_device)
        b = torch_module.randint(-64, 64, (n, n), dtype=torch_dtype, device=torch_device)
    return a, b


class TorchBackend:
    """Concrete backend runner for Torch execution paths."""

    name = "torch"

    def __init__(self, torch_module, device, cpu_workers):
        """Initialize Torch backend and apply CPU thread settings when needed."""
        self.torch = torch_module
        self.device = select_torch_device(torch_module, device)
        if device == "cpu":
            configure_torch_cpu_threads(torch_module, cpu_workers)

    def device_summary(self):
        """Return printable summary of selected Torch device."""
        return format_torch_device_summary(self.torch, self.device)

    def parallelism_summary(self):
        """Return printable summary of Torch CPU parallelism settings."""
        return f"workers={self.torch.get_num_threads()} (Torch threads)"

    def prepare_case(self, work_n, torch_dtype, is_inexact):
        """Prepare and synchronize tensors for a single matrix size case."""
        a, b = make_torch_inputs(work_n, self.torch, torch_dtype, is_inexact, self.device)
        sync_torch_device(self.torch, self.device)
        return {"a": a, "b": b}

    def run_once(self, case, metric, graph_steps):
        """Execute one timed Torch iteration for the prepared case."""
        a = case["a"]
        b = case["b"]
        if metric == "bandwidth":
            out = a + b
        elif metric == "flops":
            out = a @ b
        else:
            z = a
            for _ in range(graph_steps):
                z = (z @ b) + z
            out = z
        sync_torch_device(self.torch, self.device)
        del out
