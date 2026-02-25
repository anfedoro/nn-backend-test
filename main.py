"""Runtime entrypoint for the unified MLX/Torch benchmark.

This module intentionally keeps orchestration in one place:
- parse CLI arguments
- resolve dtype/metric behavior
- resolve and initialize backend
- run benchmark loops
- aggregate metrics and report them to console/CSV
"""

import argparse
import csv
import os
import statistics
import time

import mlx.core as mx

from backends import (
    MlxBackend,
    QUANT_GROUP_SIZE_FALLBACK,
    QUANT_GROUP_SIZE_PREFERRED,
    SUPPORTED_QUANT_BITS,
    TORCH_INT_MATMUL_DTYPE_TOKENS,
    TorchBackend,
    UnsupportedFeatureError,
    resolve_torch_dtype,
    try_import_torch,
)


def parse_args():
    """Build and parse CLI arguments for the benchmark runtime."""
    parser = argparse.ArgumentParser(description="Simple matrix benchmark with unified MLX/Torch backends.")
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
        "--backend",
        choices=["auto", "mlx", "torch"],
        default="auto",
        help="Execution backend.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Target device type.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="CPU worker count for both backends. 0 = all available CPU cores.",
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


def resolve_runtime_backend(requested_backend, metric, use_quant_ops, use_iops, dtype_token, torch_module):
    """Resolve backend for this run.

    In auto mode:
    - quantized aliases (q*) are MLX-only
    - exact integer compute prefers Torch when available
    - everything else uses MLX
    """
    if requested_backend in {"mlx", "torch"}:
        return requested_backend
    if use_quant_ops:
        return "mlx"
    if metric in {"flops", "flops_graph"} and use_iops and dtype_token in TORCH_INT_MATMUL_DTYPE_TOKENS:
        if torch_module is not None:
            return "torch"
    return "mlx"


def main():
    """Execute full benchmark workflow.

    Runtime flow:
    1) Parse CLI and resolve effective CPU parallelism.
    2) Decode dtype token into one of:
       - quantized q* path
       - inexact floating/complex path
       - exact path (IOPS mode)
    3) Resolve backend (auto/forced) and initialize backend object.
    4) Print run configuration and mode notes.
    5) For each requested matrix size:
       - adjust effective size for q* constraints (if needed)
       - prepare backend-specific tensors once
       - run warmup + measured loops
       - compute throughput numbers and collect rows
    6) Print result table.
    7) Optionally write CSV output.
    """
    args = parse_args()
    cpu_workers = resolve_cpu_workers(args)
    dtype_token = args.dtype.lower()

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
            raise ValueError(f"Unsupported quantized dtype '{args.dtype}'. MLX supports: {supported}.")
        if args.metric == "bandwidth":
            raise ValueError("Quantized dtypes q* are supported only for --metric flops and --metric flops_graph.")
        use_quant_ops = True
        itemsize = quant_bits / 8
    else:
        if not hasattr(mx, dtype_token):
            raise ValueError(f"Unsupported dtype '{args.dtype}'.")
        dtype = getattr(mx, dtype_token)
        is_inexact = mx.issubdtype(dtype, mx.inexact)
        use_iops = args.metric in {"flops", "flops_graph"} and not is_inexact
        itemsize = dtype.size

    torch_module = try_import_torch() if args.backend in {"auto", "torch"} else None
    runtime_backend = resolve_runtime_backend(
        args.backend, args.metric, use_quant_ops, use_iops, dtype_token, torch_module
    )

    if args.backend == "torch" and torch_module is None:
        raise ValueError("Torch backend was requested but torch is not installed.")

    if runtime_backend == "torch" and use_quant_ops:
        raise ValueError(
            'Torch backend does not support q* benchmark mode. '
            'See README: "Why Torch q* Is Disabled". Use --backend mlx for q*.'
        )

    if runtime_backend == "torch":
        torch_dtype = resolve_torch_dtype(torch_module, dtype_token)
        if torch_dtype is None:
            raise ValueError(f"Torch backend does not support dtype '{args.dtype}'.")
        if args.metric in {"flops", "flops_graph"} and use_iops and dtype_token not in TORCH_INT_MATMUL_DTYPE_TOKENS:
            raise ValueError(
                "Torch backend compute metrics support exact dtypes int8/int16/int32/int64 only. "
                "Use --backend mlx for this exact dtype."
            )
        backend = TorchBackend(torch_module, args.device, cpu_workers)
    else:
        torch_dtype = None
        backend = MlxBackend(args.device, cpu_workers)

    results = []

    print(f"Device: {backend.device_summary()}")
    if args.backend == "auto":
        print(f"Backend: {runtime_backend} (auto)")
    else:
        print(f"Backend: {runtime_backend} (forced)")
    if args.device == "cpu":
        print(f"Parallelism: {backend.parallelism_summary()}")
    print(f"Metric: {args.metric}")
    if args.metric == "flops_graph":
        print(f"Graph steps: {args.graph_steps}")
    print(f"DType: {args.dtype}")

    if args.metric == "bandwidth":
        print("Compute mode: bandwidth (a + b)")
    elif use_quant_ops:
        print(f"Compute mode: quantized_matmul (q{quant_bits}, affine)")
    elif use_iops:
        if runtime_backend == "torch" and dtype_token in TORCH_INT_MATMUL_DTYPE_TOKENS:
            print("Compute mode: IOPS (torch int matmul)")
        else:
            print("Compute mode: IOPS (MLX exact fallback kernel)")
    else:
        if runtime_backend == "torch":
            print("Compute mode: FLOPS (torch matmul)")
        else:
            print("Compute mode: FLOPS (MLX matmul)")

    print("Transfer accounting: excluded (device-local tensors reused across runs)")
    if runtime_backend == "mlx" and use_iops and args.metric in {"flops", "flops_graph"}:
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

        elements = work_n * work_n
        matrix_bytes = elements * itemsize

        if runtime_backend == "mlx":
            case = backend.prepare_case(work_n, dtype, is_inexact, use_quant_ops, quant_bits, quant_group_size)
        else:
            case = backend.prepare_case(work_n, torch_dtype, is_inexact)

        times = []
        for i in range(args.warmup + args.runs):
            t0 = time.perf_counter()
            if runtime_backend == "mlx":
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
            else:
                backend.run_once(case, args.metric, args.graph_steps)
            elapsed = time.perf_counter() - t0
            if i >= args.warmup:
                times.append(elapsed)

        median_s = statistics.median(times)
        row = {
            "n": n,
            "matrix_mib": matrix_bytes / (1024**2),
            "median_ms": median_s * 1000,
        }
        if use_quant_ops:
            row["effective_n"] = work_n

        if args.metric == "bandwidth":
            bytes_per_run = 3 * matrix_bytes
            row["io_mib"] = bytes_per_run / (1024**2)
            row["bandwidth_gbps"] = bytes_per_run / median_s / 1e9
        elif args.metric == "flops":
            if use_quant_ops:
                ops = 2 * (work_n**3)
                row["gqops"] = ops / median_s / 1e9
                row["tqops"] = ops / median_s / 1e12
            elif use_iops:
                if runtime_backend == "torch" and dtype_token in TORCH_INT_MATMUL_DTYPE_TOKENS:
                    ops = 2 * (work_n**3)
                else:
                    ops = 5 * elements
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
                if runtime_backend == "torch" and dtype_token in TORCH_INT_MATMUL_DTYPE_TOKENS:
                    ops = args.graph_steps * ((2 * (work_n**3)) + (work_n**2))
                else:
                    ops = args.graph_steps * (5 * elements)
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
            header = f"{'n':>8}{'eff n':>10}{'matrix MiB':>14}{'median ms':>12}{'GQOPS':>12}{'TQOPS':>12}"
        elif use_iops:
            header = f"{'n':>8}{'matrix MiB':>14}{'median ms':>12}{'GIOPS':>12}{'TIOPS':>12}"
        else:
            header = f"{'n':>8}{'matrix MiB':>14}{'median ms':>12}{'GFLOPS':>12}{'TFLOPS':>12}"
    else:
        if use_quant_ops:
            header = f"{'n':>8}{'eff n':>10}{'matrix MiB':>14}{'steps':>8}{'median ms':>12}{'GQOPS':>12}{'TQOPS':>12}"
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
                    f"{row['n']:>8}{row['effective_n']:>10}{row['matrix_mib']:>14.2f}"
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
                    f"{row['n']:>8}{row['effective_n']:>10}{row['matrix_mib']:>14.2f}{row['graph_steps']:>8}"
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
                fieldnames = ["n", "effective_n", "matrix_mib", "median_ms", "gqops", "tqops"]
            elif use_iops:
                fieldnames = ["n", "matrix_mib", "median_ms", "giops", "tiops"]
            else:
                fieldnames = ["n", "matrix_mib", "median_ms", "gflops", "tflops"]
        else:
            if use_quant_ops:
                fieldnames = ["n", "effective_n", "matrix_mib", "graph_steps", "median_ms", "gqops", "tqops"]
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
    except UnsupportedFeatureError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
