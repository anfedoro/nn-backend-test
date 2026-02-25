# nn-backend-test

Small benchmark utility for MacBook matrix performance with a unified CLI and two backends:
- `mlx`
- `torch`

## Metrics

- `bandwidth`: memory bandwidth estimation with `a + b` (`GB/s`)
- `flops`: single-step compute throughput
- `flops_graph`: lazy/repeated graph-style compute throughput (`--graph-steps`)

## Code Layout

- `main.py`: runtime orchestration, backend selection, timing loop, reporting.
- `backends/mlx_backend.py`: MLX execution path.
- `backends/torch_backend.py`: Torch execution path.
- `backends/common.py`: shared constants and common exceptions.

## Run

```bash
uv run python main.py
```

Example with explicit parameters:

```bash
uv run python main.py --metric flops_graph --backend auto --device gpu --sizes 2048 4096 --dtype float32 --graph-steps 8 --warmup 2 --runs 5
```

## Key Options

- `--backend {auto,mlx,torch}`:
  - `auto` (default): resolves backend by dtype/metric support.
  - `mlx`: force MLX.
  - `torch`: force Torch.
- `--cpu-workers N`: unified CPU parallelism limit for both backends (`0` = all CPU cores).
- `--cpu-streams N`: deprecated alias for `--cpu-workers`.
- `--device {cpu,gpu}`: logical target device.
- `--dtype`: MLX dtype token (for example `float32`, `bfloat16`, `int8`, `uint32`, `complex64`) or quantized alias (`q2`, `q3`, `q4`, `q5`, `q6`, `q8`).

## DType Behavior

- `q1` is unsupported.
- `q2/q3/q4/q5/q6/q8` are quantized aliases and are MLX-only.
- For quantized aliases, if `N` is incompatible with MLX quantization group sizes, the benchmark pads to an effective size and reports `eff n`.
- Torch backend dtype support is limited to: `float16`, `bfloat16`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `bool`, `complex64`, `complex128`.
- On Torch backend, compute metrics for exact dtypes are supported only for `int8/int16/int32/int64`.

Units for compute metrics:
- inexact dtypes (`float*`, `bfloat16`, `complex*`) -> `GFLOPS/TFLOPS`
- exact dtypes (`int*`, `uint*`, `bool`) -> `GIOPS/TIOPS`
- quantized aliases (`q*`) -> `GQOPS/TQOPS`

## Backend Resolution (`--backend auto`)

- `q*` -> MLX
- compute metric + exact integer `int8/int16/int32/int64` -> Torch (if installed), otherwise MLX fallback exact kernel
- everything else -> MLX

## Comparability Policy

- CPU parallelism is controlled by one flag: `--cpu-workers`.
- Transfer time is excluded from throughput measurements:
  - tensors are created on the target device
  - tensors are reused across warmup/measured runs
- Both backends use explicit synchronization before stopping timers.

## Why Torch `q*` Is Disabled

`q*` mode is intentionally disabled on Torch backend and remains MLX-only.

Reasons:
- this benchmark uses MLX `quantized_matmul(..., bits=q*)` semantics
- Torch does not provide one direct, stable equivalent path for this benchmark mode across backends/devices
- keeping `q*` MLX-only avoids mixed semantics and misleading cross-backend comparisons

If you force Torch with `q*`, the script exits with a clear error and points to this section.

## MLX Integer Fallback Note

When running compute metrics on exact dtypes through MLX path, MLX integer matmul is not used in this utility.
It falls back to a simple exact integer kernel and prints a one-time warning.
These IOPS numbers can be below real hardware integer peak.

## Examples

Run with automatic backend resolution:

```bash
uv run python main.py --metric flops_graph --backend auto --device gpu --sizes 4096 --dtype int32 --graph-steps 100
```

Force MLX quantized path:

```bash
uv run python main.py --metric flops_graph --backend mlx --device gpu --sizes 1000 --dtype q8 --graph-steps 100
```

Force Torch integer matmul path:

```bash
uv run python main.py --metric flops_graph --backend torch --device gpu --sizes 4096 --dtype int32 --graph-steps 100
```

Force all CPU cores on both backends:

```bash
uv run python main.py --metric bandwidth --device cpu --cpu-workers 0 --sizes 1024 2048 --dtype float32
```

Save CSV:

```bash
uv run python main.py --metric bandwidth --sizes 1024 2048 --csv result.csv
```

## Output Fields

- `n`: requested matrix size `N` for `N x N`
- `eff n`: effective matrix size after quantized padding (only for `q*`)
- `matrix MiB`: size of one matrix
- `median ms`: median run time

For `--metric bandwidth`:
- `I/O MiB`: estimated per-run data traffic (`read A + read B + write C`)
- `GB/s`: effective memory bandwidth

For `--metric flops` and `--metric flops_graph`:
- output columns are selected automatically by mode:
  - `GFLOPS/TFLOPS`
  - `GIOPS/TIOPS`
  - `GQOPS/TQOPS`
