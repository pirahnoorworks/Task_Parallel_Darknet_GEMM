# Darknet XiTAO DGEMM Demo

Portfolio project showing how Darknet-style `gemm_nt` was modified to use XiTAO task calls and become compatible with a task-parallel runtime.

This project is intentionally focused on **TAO-based DGEMM only** (no tensor template language).

## What this demonstrates

- A full CNN-convolution data path for one layer:
  - image tensor preparation
  - `im2col` transformation
  - DGEMM execution
- Two backend implementations of DGEMM on the same CNN workload:
  - OpenMP `gemm_nt` (traditional threaded path)
  - XiTAO-task `gemm_nt` (task-parallel runtime path)
- XiTAO path split into two TAO stages per row block:
  - `GT1`: initialize output block (`C = 0` for the block)
  - `GT2`: compute DGEMM multiply-accumulate for the block
- DAG wiring with XiTAO-style calls:
  - `AssemblyTask` task objects
  - `make_edge(...)` task dependencies
  - `gotao_init`, `gotao_push`, `gotao_start`, `gotao_fini`
- End-to-end correctness check against OpenMP output
- Benchmark mode suitable for hiring demos

## Darknet integration story reflected in code

In Darknet, one convolution implementation route is `im2col + gemm_nt`. This demo mirrors the framework-level change by:

1. Building the DGEMM problem from CNN layer parameters (`C,H,W,R,S,stride,pad,filters`).
2. Running `im2col` exactly like the common Darknet/Caffe path.
3. Keeping a usual OpenMP DGEMM path as reference.
4. Replacing the DGEMM core with XiTAO tasks (`GT1 -> GT2`) scheduled with runtime calls.

Core files:

- `src/darknet_gemm.cpp` : im2col + OpenMP DGEMM + TAO-task DGEMM implementations
- `include/xitao_compat.h` : XiTAO-compatible API shim for standalone runs
- `src/main.cpp` : CLI, verification, and benchmark driver

## Build and run

### Windows (PowerShell)

```powershell
cd code/portfolio/darknet-xitao-dgemm-demo
./scripts/run_demo.ps1
```

### Linux/macOS

```bash
cd code/portfolio/darknet-xitao-dgemm-demo
bash scripts/run_demo.sh
```

## CLI options

```text
darknet_xitao_dgemm_demo [options]
  --channels <int>    Input channels C, default 64
  --height <int>      Input height H, default 56
  --width <int>       Input width W, default 56
  --kernel <int>      Kernel size R=S, default 3
  --stride <int>      Conv stride, default 1
  --pad <int>         Conv padding, default 1
  --filters <int>     Number of output filters K, default 64
  --block-rows <int>  Rows per TAO block, default 64
  --threads <int>     XiTAO worker threads, default 8
  --warmup <int>      Warmup iterations, default 2
  --iters <int>       Timed iterations, default 10
  --backend <name>    omp|xitao|both (default both)
  --no-verify         Skip omp-vs-xitao correctness check
```

## Demo command examples

Full verification + both backends:

```bash
./build/darknet_xitao_dgemm_demo --backend both --iters 10 --warmup 2
```

XiTAO-only timing with larger workload:

```bash
./build/darknet_xitao_dgemm_demo --backend xitao --channels 64 --height 112 --width 112 --kernel 3 --stride 1 --pad 1 --filters 128 --block-rows 64 --threads 16 --iters 20 --warmup 3
```

## Native XiTAO linkage (optional)

By default this project uses an in-repo XiTAO-compatible runtime shim so the demo is runnable anywhere.

If you have XiTAO installed and want native linkage:

```bash
cmake -S . -B build -DUSE_NATIVE_XITAO=ON
cmake --build build --config Release
```

You may need to provide include/library paths depending on your XiTAO installation.
<<<<<<< HEAD
=======

> Disclaimer: This demo was created for portfolio purposes with assistance from GitHub Copilot.



