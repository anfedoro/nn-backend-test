"""Runtime entrypoint for the MLX benchmark utility.

This module keeps the benchmark flow intentionally direct:
- parse CLI arguments
- resolve dtype mode (FLOPS, IOPS, or quantized QOPS)
- initialize MLX backend
- run warmup and measured loops
- print and optionally save results
"""

import argparse
import csv
import math
import os
import statistics
import time

from backends.common import (
    EXACT_KERNEL_OPS_PER_ELEMENT,
    QUANT_GROUP_SIZE_FALLBACK,
    QUANT_GROUP_SIZE_PREFERRED,
    SUPPORTED_QUANT_BITS,
    UnsupportedFeatureError,
)


def resolve_dtype_info(mx, dtype_token, metric):
    """Resolve dtype behavior and numeric metadata for calculations."""
    use_quant_ops = False
    quant_bits = None
    dtype = None
    is_inexact = False
    use_iops = False
    itemsize = None

    if dtype_token.startswith("q") and dtype_token[1:].isdigit():
        quant_bits = int(dtype_token[1:])
        if quant_bits not in SUPPORTED_QUANT_BITS:
            supported = ", ".join(f"q{b}" for b in sorted(SUPPORTED_QUANT_BITS))
            raise ValueError(f"Unsupported quantized dtype '{dtype_token}'. MLX supports: {supported}.")
        if metric == "bandwidth":
            raise ValueError("Quantized dtypes q* are supported only for --metric flops and --metric flops_graph.")
        use_quant_ops = True
        return use_quant_ops, quant_bits, dtype, is_inexact, use_iops, itemsize

    if not hasattr(mx, dtype_token):
        raise ValueError(f"Unsupported dtype '{dtype_token}'.")

    dtype = getattr(mx, dtype_token)
    is_inexact = mx.issubdtype(dtype, mx.inexact)
    use_iops = metric in {"flops", "flops_graph"} and not is_inexact
    itemsize = dtype.size
    return use_quant_ops, quant_bits, dtype, is_inexact, use_iops, itemsize


def parse_args():
    """Build and parse CLI arguments for the benchmark runtime."""
    parser = argparse.ArgumentParser(description="Simple MLX matrix benchmark.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
        help="Matrix sizes N for NxN.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="MLX dtype (float32, int8, ...) or quantized alias (q2, q3, q4, q5, q6, q8).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "hybrid"],
        default="gpu",
        help="Target device type.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="CPU worker count. 0 = all available CPU cores.",
    )
    parser.add_argument(
        "--cpu-streams",
        type=int,
        default=None,
        help="Deprecated alias for --cpu-workers.",
    )
    parser.add_argument(
        "--metric",
        choices=["bandwidth", "flops", "flops_graph"],
        default="bandwidth",
        help="What to measure.",
    )
    parser.add_argument(
        "--graph-steps",
        type=int,
        default=4,
        help="Number of lazy graph steps for --metric flops_graph.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs.")
    parser.add_argument("--csv", default="", help="Optional path to save CSV.")
    return parser.parse_args()


def resolve_cpu_workers(args):
    """Resolve effective CPU worker count.

    Priority:
    1) --cpu-workers if provided and > 0
    2) --cpu-streams (deprecated alias) if provided and > 0
    3) os.cpu_count() fallback
    """
    if args.cpu_workers > 0:
        return args.cpu_workers
    if args.cpu_streams is not None and args.cpu_streams > 0:
        return args.cpu_streams
    return os.cpu_count() or 1


def array_nbytes(arr):
    """Return materialized byte size for an MLX array."""
    return math.prod(arr.shape) * arr.dtype.size


def quantized_storage_bytes(case):
    """Return storage bytes for quantized weights and float activations."""
    activation_bytes = array_nbytes(case["x"])
    weight_bytes = (
        array_nbytes(case["q_w"])
        + array_nbytes(case["q_scales"])
        + array_nbytes(case["q_biases"])
    )
    return activation_bytes, weight_bytes


def main():
    """Execute full benchmark workflow."""
    args = parse_args()

    import mlx.core as mx  # Imported lazily so --help does not require MLX runtime init.
    from backends.mlx_backend import MlxBackend

    cpu_workers = resolve_cpu_workers(args)
    dtype_token = args.dtype.lower()

    if args.device == "hybrid" and args.metric != "bandwidth":
        raise ValueError("--device hybrid is supported only for --metric bandwidth.")

    use_quant_ops, quant_bits, dtype, is_inexact, use_iops, itemsize = resolve_dtype_info(
        mx, dtype_token, args.metric
    )

    backend = MlxBackend(args.device, cpu_workers)
    results = []

    print(f"Device: {backend.device_summary()}")
    print("Backend: mlx")
    if args.device in {"cpu", "hybrid"}:
        print(f"Parallelism: {backend.parallelism_summary()}")
    print(f"Metric: {args.metric}")
    if args.metric == "flops_graph":
        print(f"Graph steps: {args.graph_steps}")
    print(f"DType: {args.dtype}")

    if args.metric == "bandwidth":
        if args.device == "hybrid":
            print("Compute mode: hybrid bandwidth (copy, experimental contention benchmark)")
        else:
            print("Compute mode: bandwidth (copy)")
    elif use_quant_ops:
        print(f"Compute mode: quantized_matmul (q{quant_bits}, affine)")
    elif use_iops:
        print("Compute mode: IOPS (MLX exact kernel)")
    else:
        print("Compute mode: FLOPS (MLX matmul)")

    print("Transfer accounting: excluded (device-local tensors reused across runs)")
    if args.device == "hybrid":
        print("Hybrid note: aggregate bandwidth depends on real CPU/GPU overlap inside MLX.")
    if use_iops and args.metric in {"flops", "flops_graph"}:
        print(
            "Warning: MLX integer matmul is unavailable for this path. "
            "Using a simple exact kernel, so IOPS may be below hardware peak."
        )
    print(f"Sizes: {args.sizes}")
    print()

    for n in args.sizes:
        work_n = n
        quant_group_size = None
        if use_quant_ops:
            if n % QUANT_GROUP_SIZE_PREFERRED == 0:
                quant_group_size = QUANT_GROUP_SIZE_PREFERRED
            elif n % QUANT_GROUP_SIZE_FALLBACK == 0:
                quant_group_size = QUANT_GROUP_SIZE_FALLBACK
            else:
                quant_group_size = QUANT_GROUP_SIZE_FALLBACK
                work_n = ((n + quant_group_size - 1) // quant_group_size) * quant_group_size
                print(f"Note: size {n} padded to {work_n} for quantization group_size={quant_group_size}.")

        case = backend.prepare_case(args.metric, work_n, dtype, is_inexact, use_quant_ops, quant_bits, quant_group_size)
        elements = work_n * work_n
        if use_quant_ops:
            activation_bytes, weight_bytes = quantized_storage_bytes(case)
        else:
            matrix_bytes = elements * itemsize

        times = []
        for i in range(args.warmup + args.runs):
            t0 = time.perf_counter()
            backend.run_once(
                case,
                args.metric,
                args.graph_steps,
                work_n,
                use_quant_ops,
                use_iops,
                quant_bits,
                quant_group_size,
            )
            elapsed = time.perf_counter() - t0
            if i >= args.warmup:
                times.append(elapsed)

        median_s = statistics.median(times)
        row = {"n": n, "median_ms": median_s * 1000}
        if use_quant_ops:
            row["activation_mib"] = activation_bytes / (1024**2)
            row["weight_mib"] = weight_bytes / (1024**2)
            row["effective_n"] = work_n
        else:
            if args.device == "hybrid" and args.metric == "bandwidth":
                row["matrix_mib"] = (2 * matrix_bytes) / (1024**2)
            else:
                row["matrix_mib"] = matrix_bytes / (1024**2)

        if args.metric == "bandwidth":
            if args.device == "hybrid":
                total_bytes_per_run = 4 * matrix_bytes
                row["io_mib"] = total_bytes_per_run / (1024**2)
                row["bandwidth_gbps"] = total_bytes_per_run / median_s / 1e9
            else:
                bytes_per_run = 2 * matrix_bytes
                row["io_mib"] = bytes_per_run / (1024**2)
                row["bandwidth_gbps"] = bytes_per_run / median_s / 1e9
        elif args.metric == "flops":
            if use_quant_ops:
                ops = 2 * (work_n**3)
                row["gqops"] = ops / median_s / 1e9
                row["tqops"] = ops / median_s / 1e12
            elif use_iops:
                ops = EXACT_KERNEL_OPS_PER_ELEMENT * elements
                row["giops"] = ops / median_s / 1e9
                row["tiops"] = ops / median_s / 1e12
            else:
                ops = 2 * (work_n**3)
                row["gflops"] = ops / median_s / 1e9
                row["tflops"] = ops / median_s / 1e12
        else:
            row["graph_steps"] = args.graph_steps
            if use_quant_ops:
                ops = args.graph_steps * ((2 * (work_n**3)) + (work_n**2))
                row["gqops"] = ops / median_s / 1e9
                row["tqops"] = ops / median_s / 1e12
            elif use_iops:
                ops = args.graph_steps * (EXACT_KERNEL_OPS_PER_ELEMENT * elements)
                row["giops"] = ops / median_s / 1e9
                row["tiops"] = ops / median_s / 1e12
            else:
                ops = args.graph_steps * ((2 * (work_n**3)) + (work_n**2))
                row["gflops"] = ops / median_s / 1e9
                row["tflops"] = ops / median_s / 1e12

        results.append(row)

    if args.metric == "bandwidth":
        header = f"{'n':>8}{'matrix MiB':>14}{'I/O MiB':>12}{'median ms':>12}{'GB/s':>12}"
    elif args.metric == "flops":
        if use_quant_ops:
            header = (
                f"{'n':>8}{'eff n':>10}{'act MiB':>12}{'weight MiB':>14}"
                f"{'median ms':>12}{'GQOPS':>12}{'TQOPS':>12}"
            )
        elif use_iops:
            header = f"{'n':>8}{'matrix MiB':>14}{'median ms':>12}{'GIOPS':>12}{'TIOPS':>12}"
        else:
            header = f"{'n':>8}{'matrix MiB':>14}{'median ms':>12}{'GFLOPS':>12}{'TFLOPS':>12}"
    else:
        if use_quant_ops:
            header = (
                f"{'n':>8}{'eff n':>10}{'act MiB':>12}{'weight MiB':>14}{'steps':>8}"
                f"{'median ms':>12}{'GQOPS':>12}{'TQOPS':>12}"
            )
        elif use_iops:
            header = f"{'n':>8}{'matrix MiB':>14}{'steps':>8}{'median ms':>12}{'GIOPS':>12}{'TIOPS':>12}"
        else:
            header = f"{'n':>8}{'matrix MiB':>14}{'steps':>8}{'median ms':>12}{'GFLOPS':>12}{'TFLOPS':>12}"

    print(header)
    print("-" * len(header))

    for row in results:
        if args.metric == "bandwidth":
            print(
                f"{row['n']:>8}{row['matrix_mib']:>14.2f}{row['io_mib']:>12.2f}"
                f"{row['median_ms']:>12.3f}{row['bandwidth_gbps']:>12.2f}"
            )
        elif args.metric == "flops":
            if use_quant_ops:
                print(
                    f"{row['n']:>8}{row['effective_n']:>10}{row['activation_mib']:>12.2f}"
                    f"{row['weight_mib']:>14.2f}"
                    f"{row['median_ms']:>12.3f}{row['gqops']:>12.2f}{row['tqops']:>12.4f}"
                )
            elif use_iops:
                print(
                    f"{row['n']:>8}{row['matrix_mib']:>14.2f}{row['median_ms']:>12.3f}"
                    f"{row['giops']:>12.2f}{row['tiops']:>12.4f}"
                )
            else:
                print(
                    f"{row['n']:>8}{row['matrix_mib']:>14.2f}{row['median_ms']:>12.3f}"
                    f"{row['gflops']:>12.2f}{row['tflops']:>12.4f}"
                )
        else:
            if use_quant_ops:
                print(
                    f"{row['n']:>8}{row['effective_n']:>10}{row['activation_mib']:>12.2f}"
                    f"{row['weight_mib']:>14.2f}{row['graph_steps']:>8}"
                    f"{row['median_ms']:>12.3f}{row['gqops']:>12.2f}{row['tqops']:>12.4f}"
                )
            elif use_iops:
                print(
                    f"{row['n']:>8}{row['matrix_mib']:>14.2f}{row['graph_steps']:>8}{row['median_ms']:>12.3f}"
                    f"{row['giops']:>12.2f}{row['tiops']:>12.4f}"
                )
            else:
                print(
                    f"{row['n']:>8}{row['matrix_mib']:>14.2f}{row['graph_steps']:>8}{row['median_ms']:>12.3f}"
                    f"{row['gflops']:>12.2f}{row['tflops']:>12.4f}"
                )

    if args.csv:
        if args.metric == "bandwidth":
            fieldnames = ["n", "matrix_mib", "io_mib", "median_ms", "bandwidth_gbps"]
        elif args.metric == "flops":
            if use_quant_ops:
                fieldnames = ["n", "effective_n", "activation_mib", "weight_mib", "median_ms", "gqops", "tqops"]
            elif use_iops:
                fieldnames = ["n", "matrix_mib", "median_ms", "giops", "tiops"]
            else:
                fieldnames = ["n", "matrix_mib", "median_ms", "gflops", "tflops"]
        else:
            if use_quant_ops:
                fieldnames = [
                    "n",
                    "effective_n",
                    "activation_mib",
                    "weight_mib",
                    "graph_steps",
                    "median_ms",
                    "gqops",
                    "tqops",
                ]
            elif use_iops:
                fieldnames = ["n", "matrix_mib", "graph_steps", "median_ms", "giops", "tiops"]
            else:
                fieldnames = ["n", "matrix_mib", "graph_steps", "median_ms", "gflops", "tflops"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print()
        print(f"Saved CSV: {args.csv}")


def cli():
    """Console entrypoint used by uv tool / project scripts."""
    try:
        main()
    except (UnsupportedFeatureError, ValueError, ModuleNotFoundError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
