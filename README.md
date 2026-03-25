# nn-backend-test

Small MLX benchmark utility for matrix throughput checks.

## Metrics

- `bandwidth`: memory bandwidth estimation with matrix copy (`GB/s`)
- `flops`: single-step compute throughput
- `flops_graph`: lazy/repeated graph-style compute throughput (`--graph-steps`)

## Code Layout

- `main.py`: runtime orchestration, timing loop, and reporting.
- `backends/mlx_backend.py`: MLX execution path and device utilities.
- `backends/common.py`: shared constants and common exceptions.

## Run

```bash
uv run python main.py
```

Example with explicit parameters:

```bash
uv run python main.py --metric flops_graph --device gpu --sizes 2048 4096 --dtype float32 --graph-steps 8 --warmup 2 --runs 5
```

## Install As Tool

Install from Git:

```bash
uv tool install git+https://github.com/anfedoro/nn-backend-test.git
```

Platform-dependent MLX install is handled automatically by dependency markers:
- Apple Silicon (`darwin/arm64`) -> `mlx`
- Linux (`linux`, any CPU arch) -> `mlx[cuda]`

This project targets only those two MLX paths.

Run installed commands:

```bash
nn-backend-test --help
nnbt --help
```

For private repository access, authenticate Git first (`gh auth login` + `gh auth setup-git`) or use SSH URL:

```bash
uv tool install git+ssh://git@github.com/anfedoro/nn-backend-test.git
```

## Key Options

- `--cpu-workers N`: CPU parallelism limit (`0` = all CPU cores).
- `--cpu-streams N`: deprecated alias for `--cpu-workers`.
- `--device {cpu,gpu,hybrid}`: logical target device.
- `--dtype`: MLX dtype token (for example `float32`, `bfloat16`, `int8`, `uint32`, `complex64`) or quantized alias (`q2`, `q3`, `q4`, `q5`, `q6`, `q8`).
- `--metric {bandwidth,flops,flops_graph}`: benchmark mode.
- `--graph-steps N`: steps for `flops_graph`.
- `--warmup N`: warmup iterations.
- `--runs N`: measured iterations.
- `--csv PATH`: optional CSV output path.

## DType Behavior

- `q1` is unsupported.
- `q2/q3/q4/q5/q6/q8` are quantized aliases and are MLX-only.
- For quantized aliases, if `N` is incompatible with MLX quantization group sizes, the benchmark pads to an effective size and reports `eff n`.
- Quantized MLX runs use `quantized_matmul`: activations stay in `float32`, quantized weights are stored as packed weights plus `float32` scales and biases, and the output is materialized in `float32`.
- For exact dtypes (`int*`, `uint*`, `bool`) compute metrics use a simple integer kernel (no integer matmul path).

Units for compute metrics:
- inexact dtypes (`float*`, `bfloat16`, `complex*`) -> `GFLOPS/TFLOPS`
- exact dtypes (`int*`, `uint*`, `bool`) -> `GIOPS/TIOPS`
- quantized aliases (`q*`) -> `GQOPS/TQOPS`

## Device Notes

- `--device gpu` requires a visible MLX GPU backend.
- `--device hybrid` is supported only for `--metric bandwidth`.
- `--device hybrid` runs CPU copy and GPU copy together as an experimental unified-memory contention benchmark.
- `--device hybrid` is not an isolated peak benchmark; results depend on real CPU/GPU overlap inside MLX.
- On Apple platforms this is Metal.
- On Linux this is CUDA.
- If GPU backend is unavailable, the script exits with a clear error.

Hybrid limitations:
- If MLX serializes CPU and GPU execution internally, aggregate bandwidth will be lower than expected.
- Hybrid results are empirical and runtime/platform dependent.

## MLX Integer Fallback Note

When running compute metrics on exact dtypes, MLX integer matmul is not used in this utility.
It uses a simple exact integer kernel and prints a warning.
These IOPS numbers can be below real hardware integer peak.

## Examples

Run floating-point graph benchmark on GPU:

```bash
uv run python main.py --metric flops_graph --device gpu --sizes 4096 --dtype float32 --graph-steps 100
```

Run quantized MLX path:

```bash
uv run python main.py --metric flops_graph --device gpu --sizes 1000 --dtype q8 --graph-steps 100
```

Run integer IOPS path:

```bash
uv run python main.py --metric flops_graph --device gpu --sizes 4096 --dtype int32 --graph-steps 100
```

Use all CPU cores:

```bash
uv run python main.py --metric bandwidth --device cpu --cpu-workers 0 --sizes 1024 2048 --dtype float32
```

Run hybrid CPU+GPU bandwidth contention benchmark:

```bash
uv run python main.py --metric bandwidth --device hybrid --cpu-workers 0 --sizes 1024 2048 --dtype float32
```

Save CSV:

```bash
uv run python main.py --metric bandwidth --sizes 1024 2048 --csv result.csv
```

## Output Fields

- `n`: requested matrix size `N` for `N x N`
- `eff n`: effective matrix size after quantized padding (only for `q*`)
- `matrix MiB`: size of one matrix
- `act MiB`: activation matrix storage for quantized runs
- `weight MiB`: quantized weight package storage for quantized runs (`q_w + scales + biases`)
- `median ms`: median run time

For `--metric bandwidth`:
- `I/O MiB`: estimated per-run data traffic (`read src + write dst`)
- `GB/s`: effective memory bandwidth

For `--metric bandwidth --device hybrid`:
- `matrix MiB`: total source matrix footprint across CPU and GPU
- `I/O MiB`: combined CPU + GPU per-run traffic
- `GB/s`: aggregate bandwidth from shared wall time

For `--metric flops` and `--metric flops_graph`:
- inexact dtypes -> `GFLOPS/TFLOPS`
- exact dtypes -> `GIOPS/TIOPS`
- quantized aliases -> `GQOPS/TQOPS`
