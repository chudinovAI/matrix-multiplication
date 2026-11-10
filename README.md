# Matrix multiplication kernel lab

Write and compare our own GEMM kernels against library implementations. CUDA, Apple Metal and ARM NEON are separate experiments, with explicit input precision, FP32 output, independent FP64 references and measured setup/API overhead.

![Apple M5 Max device execution](result/mac-m5-max/kernels/runtime.png)

NVIDIA and Apple measurements stay in different directories. They are **not a hardware ranking**: architectures, memory systems and library versions differ.

## Our kernels

| Backend | Implementation | Input formats |
| --- | --- | --- |
| `tiled` | Original CUDA shared-memory baseline; 32×64 output tile, 1024 threads | FP32 |
| `register` | CUDA 64×64 tile; 4×4 accumulators/thread | FP32 |
| `warp` | CUDA 64×128 warp-aware tile; 4×8 accumulators/thread, guarded `float4` loads/stores, padded shared A | FP32 |
| `warp_large` | CUDA 128×128 tile; 8×8 accumulators/thread, K tile 32 | FP32 |
| `pipeline` | CUDA 128×128 tile; K tile 16, double-buffered PTX `cp.async` loads overlapped with computation | FP32 |
| `wmma` | CUDA Tensor Core kernel; 128×128 block, eight warps, eight accumulator fragments/warp, masked staging/stores | TF32, FP16, BF16 |
| `triton` | Autotuned Triton kernel with grouped tile ordering and frozen prepared plans | FP32, TF32, FP16, BF16 |
| `metal_naive` | Metal scalar baseline: one output/thread | FP32, FP16 |
| `metal_tiled` | Metal shared/register tiling: 32×32 output, 2×2 accumulators/thread | FP32, FP16 |
| `metal_simd` | Metal SIMD-group matrix operations: 64×64 block, four SIMD groups, FP32 accumulation | FP32, FP16 |
| `metal_tensor` | Our Metal 4 shader using MPP TensorOps; 64×64 tile, four SIMD groups, FP32 cooperative destination, direct device-memory access | FP32, FP16, BF16 |
| `metal_tensor_morton` | Same TensorOps arithmetic with blocked 8×8 Morton tile traversal; small grids use row-major traversal | FP32, FP16, BF16 |
| `cpu_neon` | ARM64 NEON 4×8 FMA microkernel; 128×128×256 blocking, packed B, persistent context-owned workers | FP32 |

`warp` and `warp_large` work on older CUDA devices too. `pipeline`, `wmma`, Triton, TF32 and CUDA BF16 require compute capability 8.0+. The native CUDA wrapper also rejects an older virtual-architecture image before launching an unsupported kernel.

All kernels handle rectangular M×K×N problems and partial tiles. CUDA vector loads fall back to guarded scalar loads for unaligned pointers/tails. WMMA TF32 inputs are explicitly rounded to nearest-even, not left to an implicit hardware conversion.

Metal's SIMD variant uses FP16 matrix operands directly for FP16 inputs, while accumulating into FP32. Full output tiles store directly; edges use one 8×8 scratch fragment per SIMD group. This reduces threadgroup storage to 9 KiB for FP32 or 5 KiB for FP16, rather than staging the entire output tile.

TensorOps are [Metal 4 matrix primitives](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4), not a call to MPS matrix multiplication. Our shader controls tiling, bounds and traversal. `relaxed_precision` is disabled. On M5-family GPUs, these APIs can use the GPU's neural accelerators; this is **not the separate Apple Neural Engine**. Apple's [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) explains direct device access and cache-local traversal without mandatory shared-memory staging/software pipelines.

## Library baselines

| Backend | Library/device | Input formats with FP32 output |
| --- | --- | --- |
| `numpy` | CPU BLAS through NumPy | FP32 |
| `accelerate` | Direct Apple Accelerate SGEMM | FP32 |
| `torch_cpu` | PyTorch CPU | FP32 |
| `cublas` | Direct CUDA SGEMM/GemmEx | FP32, TF32, FP16, BF16 |
| `torch` | PyTorch CUDA | FP32, TF32, FP16, BF16 |
| `jax` | Optional JAX GPU JIT with explicit dot-algorithm presets | FP32, TF32, FP16, BF16 |
| `mps` | Native Apple MPSMatrixMultiplication | FP32, FP16 |
| `torch_mps` | PyTorch MPS | FP32 |

**Every output is FP32.** FP16/BF16 mean rounded input storage and FP32 accumulation/output, not a reduced-precision output subsequently cast to FP32. The native MPS baseline accepts FP16 operands with an FP32 result descriptor. MPSMatrixMultiplication rejects BF16 inputs even though the SDK defines that storage type; BF16 is measured through Metal 4 TensorOps instead. PyTorch MPS remains FP32-only here because its CUDA `out_dtype=float32` mixed-input path is not used for MPS.

CUDA FP32 uses IEEE arithmetic with TF32 disabled; cuBLAS uses pedantic FP32 math. Reduced-precision CUDA reductions are disabled. MPS does not enable `AllowReducedPrecision`. Our Metal shaders use safe math and precise floating-point functions.

Library TF32 may use IEEE FP32 or [native ties-away TF32 conversion](https://docs.nvidia.com/cuda/archive/11.0_GA/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt). Validation accepts those legal alternatives and records the closest reference. Custom WMMA/Triton TF32 use nearest-even only. A reference match is not an instruction trace.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11–3.14 is supported; the commands below select Python 3.12. Exact versions for each saved experiment are recorded in its report metadata.

```sh
uv sync --frozen --python 3.12
```

### Apple Silicon Mac

Native CPU/Metal builds require Apple Silicon, macOS 15+ and an Xcode/Command Line Tools SDK exposing the Metal 4 declarations. TensorOps require macOS 26+. Earlier supported macOS versions can use the classic Metal kernels/MPS; unsupported TensorOps cases are recorded with reasons.

```sh
make                  # Metal + CPU on macOS
# Or independently:
make metal
make cpu
```

Metal shaders compile **at runtime** with `newLibraryWithSource`. No offline `metal` compiler, downloaded Metal Toolchain component, PyObjC or Python Metal binding is required. Shaders are loaded relative to the project, not the current directory. Native code uses Objective-C++/ARC with Foundation, Metal and MetalPerformanceShaders. CPU code uses NEON and Accelerate's modern CBLAS interface.

### CUDA

CUDA builds require Linux, a compatible NVIDIA driver and a toolkit with `nvcc`/cuBLAS. Linux x86_64 PyTorch wheels use the CUDA 12.8 index; macOS uses ordinary PyTorch wheels with MPS support.

```sh
make cuda
make cuda CUDA_HOME=/usr/local/cuda-12.8 CUDA_ARCH=sm_86
# The SIMT kernels also build for older devices with CUDA_ARCH=sm_75.
```

`CUDA_ARCH=native` is the default. Cross-compilation is supported: compile `sm_86` on another host with a toolkit, then copy `build/matmul.so` to the GPU host. The binary has a toolkit-library rpath; compatible CUDA runtime/cuBLAS libraries must be available there. CUDA ABI **3** is checked before binding functions; rebuild old ABI-2 binaries. New Metal/CPU libraries use ABI 1.

The default builds use `-O3`, not fast-math. `NVCCFLAGS` can expose resource usage:

```sh
make cuda CUDA_ARCH=sm_86 NVCCFLAGS='-O3 -Xptxas=-v'
```

Optional packages:

```sh
uv sync --frozen --python 3.12 --extra triton
uv sync --frozen --python 3.12 --extra triton --extra jax
```

Triton does not need a system-wide CUDA toolkit, but requires an SM80+ device for this implementation. Its launch helpers also need matching Python development headers (`Python.h`): uv-managed Python includes them; distro Python may need `python3-devel` (Fedora) or `python3-dev` (Debian/Ubuntu). A missing package needed by an active requested backend is an error, not a silent replacement with another implementation.

## Run

`--platform auto` selects Metal on Apple Silicon, CUDA when available elsewhere, or CPU. An explicitly CPU-only backend list selects CPU without opening a GPU context. Defaults include the platform's own native kernels and installed base-library baselines; optional Triton/JAX are selected explicitly.

```sh
# This Mac: all native CPU/Metal variants, MPS and PyTorch baselines.
uv run --frozen --python 3.12 src/main.py --platform metal \
  --precisions fp32 fp16 bf16 --cpu-threads 18 --output result/my-mac

# Ampere-or-newer CUDA: old/new custom kernels plus libraries.
uv run --frozen --python 3.12 --extra triton src/main.py --platform cuda \
  --backends numpy tiled register warp warp_large pipeline wmma cublas torch triton \
  --precisions fp32 tf32 fp16 bf16 --cpu-threads 8 --output result/my-cuda

# Add optional JAX on CUDA; it has resident/end-to-end timing, not CUDA-graph timing.
uv run --frozen --python 3.12 --extra jax src/main.py --platform cuda \
  --backends cublas torch jax --precisions fp32 tf32 fp16 bf16

# Native CPU experiments on this Mac; thread pool persists across matrix plans.
uv run --frozen --python 3.12 src/main.py --platform cpu \
  --backends numpy cpu_neon accelerate torch_cpu --cpu-threads 18 \
  --modes resident end_to_end --output result/my-cpu

# CPU-only libraries: no GPU/native build needed, also on Linux.
uv run --frozen --python 3.12 src/main.py --backends numpy torch_cpu

# Boundaries, short K, rectangular matrices and partial Morton blocks.
uv run --frozen --python 3.12 src/main.py --platform metal \
  --shapes 31x33x65 511x257x129 513x129x577 --precisions fp32 fp16 bf16
```

Defaults: square sizes 64–4096, FP32, 10 warmups, 21 samples, one requested CPU thread. Other controls: `--sizes`, `--shapes MxKxN`, `--modes`, `--seed`, `--device`, `--warmup`, `--repeats`, `--no-plot`. `--library` overrides the CUDA/Metal library, or the CPU library with `--platform cpu`. No valid backend/precision/mode combination is an error; otherwise unsupported combinations are printed and recorded.

## What is timed

| Measurement | Included |
| --- | --- |
| `backend_initialization_ms` | Reusable context/library initialization; includes runtime Metal library compilation where needed |
| `setup_ms` | Host-input placement/conversion/packing, output allocation, preparation, first dispatch and completion; includes initial pipeline creation/JIT/autotuning |
| `device` | CUDA events inside a captured graph, or native Metal command-buffer GPU timestamps; ten GEMMs/batch, normalized per GEMM |
| `resident` | Synchronized wall time with converted/packed resident inputs and reused output; includes host dispatch |
| `end_to_end` | Original FP32 host inputs to owned FP32 host output: data allocations, casts, packing, placement/transfers and readback |

Schema 3 setup **includes input placement/conversion**. Setup is not pure compiler time, and caches may already be warm. Validation/readback used only for correctness, graph capture and the first CUDA graph replay are excluded from steady-state samples.

- CPU backends and PyTorch MPS have no device-timestamp rows. JAX is not timed with unrelated PyTorch CUDA-stream events.
- NumPy now has a resident mode using `matmul(..., out=...)`. JAX keeps its normal result-allocation behavior.
- CUDA native/PyTorch/Triton reuse resident output buffers. Triton end-to-end rebinds its frozen kernel without retuning.
- Metal contexts cache compute pipelines and shape-specific MPS operations. End-to-end plans allocate fresh shared MTLBuffers, copy/convert directly into them, and return an owned NumPy copy. Unified memory does **not** make this path allocation/copy-free or imply PCIe transfers. PyTorch MPS has its own caching allocator/transport.
- NEON packs B once per prepared plan; end-to-end includes packing and owned-result copying. Persistent workers belong to the context, so plan recreation does not recreate OS threads. One-thread execution runs directly on the caller. `actual_threads` is clamped to available row blocks/hardware. Vendor thread counts are not asserted: their limits are requested before numerical imports.
- Same seeded FP32 inputs are shared between implementations. Samples reuse data and describe cache-warm execution; clocks/desktop activity are not controlled by the benchmark.

Throughput is `2*M*K*N / seconds / 1e12` TFLOP/s. End-to-end throughput includes transport/API overhead, not just arithmetic.

## Correctness and format error

Each problem has an independent CPU FP64 reference for the original FP32 inputs. Another FP64 reference multiplies precision-rounded inputs. Library TF32 also permits ties-away/IEEE alternatives. All formats use the same component-wise arithmetic allowance:

```
abs(result - reference) <= 5e-4 + 5e-4*abs(reference) + gamma_K*S
gamma_K = K*u / (1 - K*u), u = 2^-23
S = abs(A_rounded) @ abs(B_rounded)
```

Full FP32 epsilon conservatively covers hardware MMA accumulation whose internal rounding is not necessarily nearest-even. TF32 uses an upper envelope of legal rounded inputs for `S`; if `K*u >= 1`, the bound uses `(1+u)^K - 1`. Cancellation can make this bound loose; passing it is not a promise of small relative error on ill-conditioned inputs.

Reports separate total `relative_frobenius_error`/`max_abs_error` against the original inputs, arithmetic error against the selected reference, quantization error, `max_tolerance_ratio` and `roundoff_bound_max`. `arithmetic_reference` identifies RNE, ties-away or the permitted library IEEE path. Zero-reference relative error is zero for an exact result, otherwise `null`. Wrong shapes, non-FP32/nonfinite outputs, format overflow or failed checks abort. End-to-end paths are checked too; asynchronous GPU errors surface at synchronization.

## Saved experiments and findings

| Directory | Hardware/coverage |
| --- | --- |
| [`result/mac-m5-max/kernels`](result/mac-m5-max/kernels/) | Apple M5 Max, 40 GPU cores, 48 GB unified memory; 260 rows, sizes 64–4096, 18 requested CPU threads, FP32/FP16/BF16 |
| [`result/mac-m5-max/cpu-1`](result/mac-m5-max/cpu-1/) | Same Mac, one requested CPU thread, sizes 64–2048 |
| [`result/mac-m5-max/tails`](result/mac-m5-max/tails/) | Same Mac, rectangular/tail problems including partial Morton blocks |
| [`result/tensorpc/custom`](result/tensorpc/custom/) | RTX 3070 Ti/SM86; 350 rows, sizes 64–4096, custom SIMT/`cp.async`/WMMA, cuBLAS, PyTorch, Triton, JAX and CPU baseline; all four precisions |

The current analysis uses the RTX 3070 Ti for CUDA and the M5 Max for Metal/CPU. The older [GTX 1660 run](result/homepc/custom/) is retained only as a legacy record and is excluded from the findings below.

The Mac run used three warmups/nine samples. Observed 4096² device medians:

| M5 Max backend | FP32 ms | FP16 ms | BF16 ms |
| --- | ---: | ---: | ---: |
| `metal_naive` | 160.53 | 158.28 | — |
| `metal_tiled` | 33.01 | 32.65 | — |
| `metal_simd` | 146.95 | 15.81 | — |
| `metal_tensor` | 8.95 | 2.59 | 2.33 |
| `metal_tensor_morton` | 9.28 | 2.52 | 2.43 |
| `mps` | 9.70 | 2.44 | — |

These are one machine/run, not peak specifications. TensorOps make our shader competitive with MPS here. Shared-memory/SIMD-group complexity alone did not guarantee an improvement, and Morton order was not a universal win. GPU time is not full application latency: `metal_tensor` FP32 end-to-end was 24.29 ms while `torch_mps` was 11.99 ms in this run.

The Mac NEON implementation also scales across its persistent workers. For 2048² resident calls:

| Requested CPU threads | Actual NEON workers | Median ms |
| ---: | ---: | ---: |
| 1 | 1 | 231.47 |
| 18 | 16 | 17.73 |

The 16-row-block limit explains why the second case uses fewer workers than requested. Accelerate/NumPy remain substantially faster CPU baselines; this is not a claim that our NEON implementation beats vendor BLAS.

**RTX 3070 Ti verification is complete.** All nine native CUDA tests, eight precision tests and five Triton tests passed, including masked/unaligned inputs, TF32 halfway rounding and CUDA Graph capture. The new run contains 350 validated measurement rows, with three warmups/nine samples and FP32 outputs. Source/binary hashes, compiler resources and execution evidence are recorded in [`result/tensorpc/custom-build.json`](result/tensorpc/custom-build.json).

Observed 4096² CUDA-device medians (ms; setup/transfers excluded):

| Implementation | FP32 | TF32 | FP16 | BF16 |
| --- | ---: | ---: | ---: | ---: |
| `tiled` | 66.145 | — | — | — |
| `register` | 21.360 | — | — | — |
| `warp` | 13.708 | — | — | — |
| `warp_large` | 12.310 | — | — | — |
| `pipeline` (`cp.async`) | 12.276 | — | — | — |
| `wmma` | — | 17.643 | 13.985 | 10.687 |
| cuBLAS | 10.518 | 3.402 | 1.764 | 1.736 |
| PyTorch | 10.466 | 3.402 | 1.764 | 1.732 |
| Triton | 15.430 | 5.101 | 3.384 | 3.367 |

`pipeline` is 5.39× faster than `tiled` and 1.74× faster than `register` at this size, but only 0.28% ahead of `warp_large`: this run does not establish a meaningful `cp.async` win. It remains 16.7% slower than cuBLAS FP32. The first custom WMMA kernel is correct but 5.2–7.9× slower than cuBLAS in its matching precision; Tensor Core instructions alone did not produce a competitive implementation.

JAX is included in the resident/end-to-end reports, not this device-only table: it has no device timestamp timing here. Cross-machine timings are not an architectural ranking.

## What these experiments teach

The goal is an optimization lab, not a universal replacement for vendor GEMM. The four current reports contain 798 measurement rows; superseded runs have been removed, apart from the separately identified legacy GTX record. Baseline kernels remain deliberately: comparing successive implementations is the experiment.

- **Reuse and layout come first.** Shared/register tiling and warp-aware work distribution make the custom CUDA kernels substantially faster than the original baseline. Writing lower-level instructions is useful when it addresses a measured bottleneck, not by itself.
- **Tuning depends on the workload.** The larger SIMT tile improves the large RTX 3070 Ti case, but asynchronous copies do not establish a meaningful extra win there. Morton traversal on the Mac is not a universal improvement either.
- **Matrix instructions are not a complete GEMM implementation.** Our Metal TensorOps shader is competitive with MPS on the measured large squares; our first CUDA WMMA kernel is not competitive with cuBLAS. Neither API choice guarantees good memory access, scheduling or resource usage.
- **Precision changes the problem.** FP16/BF16 inputs with FP32 output accelerate cuBLAS on the RTX 3070 Ti, but choosing those formats does not make our first WMMA kernel competitive. Input quantization and arithmetic error must be reported separately; these random-input results do not establish accuracy for every application.
- **Kernel time is not application time.** Packing, conversion, allocation, dispatch and readback can reverse a device-only comparison. Unified memory does not remove explicit copies in our wrapper. Library baselines remain essential; a many-worker NEON speedup does not establish a win over Accelerate.
- **Abstraction level is a separate axis.** Native CUDA, Triton and the PyTorch/JAX frontends expose different amounts of control and setup cost. Framework names alone do not identify the instructions or library algorithm that ran.
- **Negative results are useful evidence.** Keep the slow but correct variants. For the next WMMA iteration, profile memory access, register usage, occupancy and matrix-unit utilization before choosing a change. Current timings do not diagnose the cause of its performance gap.

Performance conclusions apply to the saved shapes, precisions and cache-warm runs. Desktop activity and clocks were not controlled; small timing differences are not established improvements.


## Reports, native APIs and checks

Each measured directory contains `results.json`, `results.csv`, and plots for the modes actually measured: `runtime.png` (device), `resident.png`, `end_to_end.png`. JSON schema **3** records raw samples, environment, execution device, unsupported cases and generic `kernel_config`. Configurations come from native geometry constants/Triton's selected plan; library internals are not guessed. Native binaries are hashed, and Metal's runtime shader source is identified separately. Shaded p10–p90 bands are not confidence intervals. Generated files in the chosen output directory are overwritten on subsequent runs.

`NativeMatmul.prepare(name, a, b, out, precision=...)` validates CUDA tensors and returns an asynchronous operation bound to the current stream. Closures retain storage, output cannot overlap inputs, and the C library allocates/copies no matrix buffers. `configuration(name, precision)` returns the actual native geometry. Serialize a context across streams and synchronize before closing it.

`MetalMatmul.prepare(name, a, b, precision)` accepts original contiguous FP32 NumPy inputs. A plan exposes `run()`, `synchronize()`, owned `numpy()`, `config`, `device_samples()` and `close()`; plans retain their Metal resources independently of the creating context. `CpuMatmul(..., threads=N)` has the same prepared run/readback interface without GPU timing. CPU inputs must remain unchanged while prepared; closing a CPU context closes its plans/workers. `triton_matmul.prepare(...)` keeps its `run`, `out`, `config` and frozen `bind` interface.

Triton autotuning varies tiles, warps and stages, with shape/precision/device-specific keys. Tensor modes include 128×128/128×256 candidates. Selected configurations are copied to independent plans and bypass the autotuner during resident launches/rebinding/capture. A new process may retune; setup records observed cost, not an assumed cold compilation time.

```sh
# After the native build appropriate for the machine:
uv run --frozen --python 3.12 python -m unittest discover -s tests -v
```

Regression checks cover independent numerical results, rounded-input ties/subnormals, tails/long K, malformed/overlapping inputs, prepared-plan lifetimes, persistent-worker concurrency, GPU timestamp validity and CUDA graphs. Hardware-dependent tests skip explicitly when prerequisites are missing; compilation alone does not count as a GPU correctness run.
