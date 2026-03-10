"""MLX backend implementation.

This module contains MLX-specific data creation, kernel execution, and
device/stream utilities used by the runtime orchestrator in main.py.
"""

import mlx.core as mx

from .common import UnsupportedFeatureError


def make_mlx_input(n, dtype, is_inexact):
    """Create one MLX input matrix matching requested dtype behavior."""
    if dtype == mx.bool_:
        a = mx.random.randint(0, 2, shape=(n, n), dtype=dtype)
    elif is_inexact:
        a = mx.random.uniform(shape=(n, n), dtype=mx.float32).astype(dtype)
    else:
        a = mx.random.randint(0, 127, shape=(n, n), dtype=dtype)
    return a


def exact_compute_once(x, y):
    """Simple exact integer kernel used as MLX fallback for IOPS mode."""
    p = x * y
    q = p + x
    r = mx.bitwise_xor(q, y)
    s = r + p
    return mx.bitwise_or(s, x)


def quantized_matmul_once(x, q_w, q_scales, q_biases, bits, group_size):
    """Run one MLX quantized matmul call with explicit quantization params."""
    return mx.quantized_matmul(
        x,
        q_w,
        q_scales,
        q_biases,
        transpose=False,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )


def ensure_gpu_qmm_supported(x, q_w, q_scales, q_biases, bits, group_size):
    """Probe GPU quantized matmul support and raise clear error if unavailable."""
    try:
        probe = quantized_matmul_once(x[:1], q_w, q_scales, q_biases, bits, group_size)
        mx.eval(probe)
    except RuntimeError as exc:
        if "QMM NYI" in str(exc):
            raise UnsupportedFeatureError(
                "Quantized dtypes (q*) are not implemented for MLX GPU backend in this setup "
                "(QMM NYI). Use --device cpu for q* or use a non-quantized dtype on GPU."
            ) from exc
        raise


def detect_mlx_backend(device):
    """Return human-readable MLX backend name for the selected device."""
    if device.type == mx.cpu:
        return "CPU"
    if device.type == mx.gpu:
        try:
            if mx.cuda.is_available():
                return "CUDA"
        except Exception:
            pass
        try:
            if mx.metal.is_available():
                return "Metal"
        except Exception:
            pass
        return "GPU"
    return str(device.type)


def mlx_gpu_is_available():
    """Return True when MLX can run GPU kernels on this machine."""
    try:
        if mx.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        if mx.metal.is_available():
            return True
    except Exception:
        pass
    return False


def format_mlx_device_summary(device):
    """Build a detailed MLX device summary for console output."""
    backend = detect_mlx_backend(device)
    details = []
    try:
        info = mx.device_info(device)
    except Exception:
        info = {}

    device_name = info.get("device_name")
    if device_name:
        details.append(str(device_name))

    architecture = info.get("architecture")
    if architecture and architecture != device_name:
        details.append(f"arch={architecture}")

    if details:
        return f"{device} [{backend}] ({', '.join(details)})"
    return f"{device} [{backend}]"


class MlxBackend:
    """Concrete backend runner for MLX execution paths."""

    name = "mlx"

    def __init__(self, device, cpu_workers):
        """Initialize MLX backend and CPU stream pool when needed."""
        self.device = device
        self.cpu_workers = cpu_workers
        if device == "gpu" and not mlx_gpu_is_available():
            raise UnsupportedFeatureError(
                "MLX GPU backend is unavailable on this system. "
                "No supported GPU was detected. Use --device cpu."
            )
        device_map = {"cpu": mx.cpu, "gpu": mx.gpu}
        mx.set_default_device(device_map[device])
        self.streams = [mx.new_stream(mx.cpu) for _ in range(cpu_workers)] if device == "cpu" else []

    def device_summary(self):
        """Return printable summary of currently selected MLX device."""
        return format_mlx_device_summary(mx.default_device())

    def parallelism_summary(self):
        """Return printable summary of MLX CPU parallelism settings."""
        return f"workers={self.cpu_workers} (MLX streams)"

    def prepare_case(self, work_n, dtype, is_inexact, use_quant_ops, quant_bits, quant_group_size):
        """Prepare and materialize tensors for a single matrix size case."""
        if use_quant_ops:
            x = mx.random.uniform(shape=(work_n, work_n), dtype=mx.float32)
            w = mx.random.uniform(shape=(work_n, work_n), dtype=mx.float32)
            q_w, q_scales, q_biases = mx.quantize(
                w, bits=quant_bits, mode="affine", group_size=quant_group_size
            )
            del w
            mx.eval(x, q_w, q_scales, q_biases)
            if self.device == "gpu":
                ensure_gpu_qmm_supported(x, q_w, q_scales, q_biases, quant_bits, quant_group_size)
            return {
                "x": x,
                "q_w": q_w,
                "q_scales": q_scales,
                "q_biases": q_biases,
            }

        a = make_mlx_input(work_n, dtype, is_inexact)
        b = make_mlx_input(work_n, dtype, is_inexact)
        mx.eval(a, b)
        return {"a": a, "b": b}

    def run_once(self, case, metric, graph_steps, work_n, use_quant_ops, use_iops, quant_bits, quant_group_size):
        """Execute one timed iteration for the prepared case."""
        if use_quant_ops:
            x = case["x"]
            q_w = case["q_w"]
            q_scales = case["q_scales"]
            q_biases = case["q_biases"]
            if self.device == "cpu":
                outs = []
                for stream_idx, stream in enumerate(self.streams):
                    start = stream_idx * work_n // self.cpu_workers
                    end = (stream_idx + 1) * work_n // self.cpu_workers
                    x_part = x[start:end]
                    with mx.stream(stream):
                        if metric == "flops":
                            out = quantized_matmul_once(x_part, q_w, q_scales, q_biases, quant_bits, quant_group_size)
                        else:
                            z = x_part
                            for _ in range(graph_steps):
                                z = quantized_matmul_once(
                                    z, q_w, q_scales, q_biases, quant_bits, quant_group_size
                                ) + z
                            out = z
                    outs.append(out)
            else:
                if metric == "flops":
                    out = quantized_matmul_once(x, q_w, q_scales, q_biases, quant_bits, quant_group_size)
                else:
                    z = x
                    for _ in range(graph_steps):
                        z = quantized_matmul_once(z, q_w, q_scales, q_biases, quant_bits, quant_group_size) + z
                    out = z
                outs = [out]
            mx.eval(*outs)
            return

        a = case["a"]
        b = case["b"]
        if self.device == "cpu":
            outs = []
            for stream_idx, stream in enumerate(self.streams):
                start = stream_idx * work_n // self.cpu_workers
                end = (stream_idx + 1) * work_n // self.cpu_workers
                a_part = a[start:end]
                b_part = b[start:end]
                with mx.stream(stream):
                    if metric == "bandwidth":
                        out = mx.array(a_part)
                    elif metric == "flops":
                        if use_iops:
                            out = exact_compute_once(a_part, b_part)
                        else:
                            out = a_part @ b
                    else:
                        z = a_part
                        if use_iops:
                            for _ in range(graph_steps):
                                z = exact_compute_once(z, b_part)
                        else:
                            for _ in range(graph_steps):
                                z = (z @ b) + z
                        out = z
                outs.append(out)
        else:
            if metric == "bandwidth":
                out = mx.array(a)
            elif metric == "flops":
                if use_iops:
                    out = exact_compute_once(a, b)
                else:
                    out = a @ b
            else:
                z = a
                if use_iops:
                    for _ in range(graph_steps):
                        z = exact_compute_once(z, b)
                else:
                    for _ in range(graph_steps):
                        z = (z @ b) + z
                out = z
            outs = [out]
        mx.eval(*outs)
