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


def materialize_copy(x):
    """Force a real copy in MLX.

    `mx.array(x)` can alias the same underlying storage and benchmark as a
    near no-op. Stacking a single tensor forces allocation and data movement,
    and slicing back removes the temporary axis after materialization.
    """
    return mx.stack([x], axis=0)[0]


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
    device_type = getattr(device, "type", device)
    if device_type == mx.cpu:
        return "CPU"
    if device_type == mx.gpu:
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
    device_type = getattr(device, "type", device)
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

    if hasattr(device, "type"):
        label = str(device)
    elif device_type == mx.cpu:
        label = "cpu"
    elif device_type == mx.gpu:
        label = "gpu"
    else:
        label = str(device)

    if details:
        return f"{label} [{backend}] ({', '.join(details)})"
    return f"{label} [{backend}]"


class MlxBackend:
    """Concrete backend runner for MLX execution paths."""

    name = "mlx"

    def __init__(self, device, cpu_workers):
        """Initialize MLX backend and CPU stream pool when needed."""
        self.device = device
        self.cpu_workers = cpu_workers
        self.gpu_backend_name = detect_mlx_backend(mx.gpu) if device in {"gpu", "hybrid"} else None
        if device in {"gpu", "hybrid"} and not mlx_gpu_is_available():
            raise UnsupportedFeatureError(
                "MLX GPU backend is unavailable on this system. "
                "No supported GPU was detected. Use --device cpu."
            )
        self.cpu_streams = []
        self.gpu_stream = None
        if device == "cpu":
            mx.set_default_device(mx.cpu)
            self.cpu_streams = [mx.new_stream(mx.cpu) for _ in range(cpu_workers)]
        elif device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            self.cpu_streams = [mx.new_stream(mx.cpu) for _ in range(cpu_workers)]
            self.gpu_stream = mx.new_stream(mx.gpu)

    def device_summary(self):
        """Return printable summary of currently selected MLX device."""
        if self.device == "hybrid":
            return f"{format_mlx_device_summary(mx.cpu)} + {format_mlx_device_summary(mx.gpu)}"
        return format_mlx_device_summary(mx.default_device())

    def parallelism_summary(self):
        """Return printable summary of MLX CPU parallelism settings."""
        if self.device == "hybrid":
            return f"cpu_workers={self.cpu_workers} (MLX streams), gpu_stream=1"
        return f"workers={self.cpu_workers} (MLX streams)"

    def uses_cuda_bandwidth_fallback(self):
        """Return True when bandwidth path should avoid CUDA copy kernels."""
        return self.gpu_backend_name == "CUDA"

    def bandwidth_compute_mode(self):
        """Return a user-facing description for the active bandwidth path."""
        if self.device == "hybrid":
            if self.uses_cuda_bandwidth_fallback():
                return "hybrid bandwidth (CPU copy + CUDA fallback a + b, experimental contention benchmark)"
            return "hybrid bandwidth (copy, experimental contention benchmark)"
        if self.uses_cuda_bandwidth_fallback():
            return "bandwidth (CUDA fallback: a + b)"
        return "bandwidth (copy)"

    def bandwidth_matrix_multiplier(self):
        """Return source-matrix footprint multiplier for reporting."""
        if self.device == "hybrid":
            if self.uses_cuda_bandwidth_fallback():
                return 3
            return 2
        return 1

    def bandwidth_io_multiplier(self):
        """Return per-run traffic multiplier for reporting."""
        if self.device == "hybrid":
            if self.uses_cuda_bandwidth_fallback():
                return 5
            return 4
        if self.uses_cuda_bandwidth_fallback():
            return 3
        return 2

    def prepare_case(self, metric, work_n, dtype, is_inexact, use_quant_ops, quant_bits, quant_group_size):
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

        if self.device == "hybrid":
            with mx.stream(mx.cpu):
                cpu_a = make_mlx_input(work_n, dtype, is_inexact)
            with mx.stream(self.gpu_stream):
                gpu_a = make_mlx_input(work_n, dtype, is_inexact)
                if self.uses_cuda_bandwidth_fallback():
                    gpu_b = make_mlx_input(work_n, dtype, is_inexact)
                    mx.eval(cpu_a, gpu_a, gpu_b)
                    return {"cpu_a": cpu_a, "gpu_a": gpu_a, "gpu_b": gpu_b}
            mx.eval(cpu_a, gpu_a)
            return {"cpu_a": cpu_a, "gpu_a": gpu_a}

        a = make_mlx_input(work_n, dtype, is_inexact)
        if metric == "bandwidth":
            if self.uses_cuda_bandwidth_fallback():
                b = make_mlx_input(work_n, dtype, is_inexact)
                mx.eval(a, b)
                return {"a": a, "b": b}
            mx.eval(a)
            return {"a": a}

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
                for stream_idx, stream in enumerate(self.cpu_streams):
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

        if self.device == "hybrid":
            cpu_a = case["cpu_a"]
            gpu_a = case["gpu_a"]
            cpu_outs = []
            for stream_idx, stream in enumerate(self.cpu_streams):
                start = stream_idx * work_n // self.cpu_workers
                end = (stream_idx + 1) * work_n // self.cpu_workers
                cpu_part = cpu_a[start:end]
                with mx.stream(stream):
                    cpu_out = materialize_copy(cpu_part)
                cpu_outs.append(cpu_out)
            with mx.stream(self.gpu_stream):
                if self.uses_cuda_bandwidth_fallback():
                    gpu_out = gpu_a + case["gpu_b"]
                else:
                    gpu_out = materialize_copy(gpu_a)
            mx.eval(*cpu_outs, gpu_out)
            return

        a = case["a"]
        if self.device == "cpu":
            outs = []
            for stream_idx, stream in enumerate(self.cpu_streams):
                start = stream_idx * work_n // self.cpu_workers
                end = (stream_idx + 1) * work_n // self.cpu_workers
                a_part = a[start:end]
                with mx.stream(stream):
                    if metric == "bandwidth":
                        out = materialize_copy(a_part)
                    elif metric == "flops":
                        b_part = case["b"][start:end]
                        if use_iops:
                            out = exact_compute_once(a_part, b_part)
                        else:
                            out = a_part @ case["b"]
                    else:
                        b_part = case["b"][start:end]
                        z = a_part
                        if use_iops:
                            for _ in range(graph_steps):
                                z = exact_compute_once(z, b_part)
                        else:
                            b = case["b"]
                            for _ in range(graph_steps):
                                z = (z @ b) + z
                        out = z
                outs.append(out)
        else:
            if metric == "bandwidth":
                if self.uses_cuda_bandwidth_fallback():
                    out = a + case["b"]
                else:
                    out = materialize_copy(a)
            elif metric == "flops":
                b = case["b"]
                if use_iops:
                    out = exact_compute_once(a, b)
                else:
                    out = a @ b
            else:
                b = case["b"]
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
