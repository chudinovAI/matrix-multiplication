"""Prepared benchmark cases; numerical libraries are imported after thread limits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from precision import CPU_BACKENDS, CUDA_BACKENDS, INPUT_DTYPES, METAL_BACKENDS, torch_precision

ROOT = Path(__file__).resolve().parents[1]
DEVICE_BATCH = 10
DEFAULT_BACKENDS = {
    "cuda": ("numpy", "tiled", "register", "warp", "warp_large", "pipeline", "wmma", "cublas", "torch"),
    "metal": ("numpy", "accelerate", "cpu_neon", "torch_cpu", *METAL_BACKENDS),
    "cpu": ("numpy", "torch_cpu"),
}
NATIVE_CUDA = tuple(name for name in CUDA_BACKENDS if name not in ("torch", "triton", "jax"))
NATIVE_METAL = tuple(name for name in METAL_BACKENDS if name != "torch_mps")


def noop():
    pass


def resolve_target(requested: str, backends: list[str] | None) -> str:
    if requested != "auto":
        return requested
    if backends and all(name in CPU_BACKENDS for name in backends):
        return "cpu"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "metal"
    if backends and any(name in CUDA_BACKENDS for name in backends):
        return "cuda"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def mode_supported(backend: str, mode: str) -> bool:
    return mode != "device" or backend in (*NATIVE_CUDA, "torch", "triton", *NATIVE_METAL)


def measure_cuda_device(call, torch, stream, warmup: int, repeats: int) -> list[float]:
    """CUDA event interval around ten graph nodes, not Python enqueue gaps."""
    for _ in range(warmup):
        call()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        start.record(stream)
        for _ in range(DEVICE_BATCH):
            call()
        end.record(stream)
    graph.replay()
    end.synchronize()
    samples = []
    for _ in range(repeats):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end) / DEVICE_BATCH)
    return samples


@dataclass
class PreparedCase:
    run: Callable
    synchronize: Callable
    output: Callable
    end_to_end: Callable
    execution_device: str
    config: dict | None = None
    device_samples: Callable | None = None
    close: Callable = noop

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class BackendRuntime:
    """One platform context per run; matrix placement belongs to case setup."""

    def __init__(self, args):
        self.args = args
        self.stack = ExitStack()
        self.torch = self.stream = self.device = self.capability = None
        self.native = self.metal = self.cpu = self.triton = None
        self.jax = self.jnp = self.jax_device = None
        self.jax_functions = {}
        self.metadata = {"native_libraries": {}, "backend_initialization_ms": {}}

    def __enter__(self):
        import time

        args = self.args
        names = args.backends
        try:
            if args.target == "cuda" and any(name in CUDA_BACKENDS for name in names):
                self._torch()
                if not self.torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable on this host; select --platform cpu/metal or run on tensorpc")
                self.torch.cuda.set_device(args.device)
                self.device = self.torch.device("cuda", args.device)
                self.stream = self.torch.cuda.Stream(device=self.device)
                self.stack.enter_context(self.torch.cuda.stream(self.stream))
                self.capability = self.torch.cuda.get_device_capability(self.device)
                props = self.torch.cuda.get_device_properties(self.device)
                self.metadata.update(cuda=self.torch.version.cuda, gpu={
                    "name": props.name, "compute_capability": list(self.capability),
                    "memory_bytes": props.total_memory, "unified_memory": False, "device": args.device})
            if any(name in ("torch_cpu", "torch_mps") for name in names):
                self._torch()
            if args.target == "metal" and any(name in METAL_BACKENDS for name in names):
                if sys.platform != "darwin" or platform.machine() != "arm64":
                    raise RuntimeError("Metal benchmarks require macOS on Apple Silicon")
                if "torch_mps" in names and not self.torch.backends.mps.is_available():
                    raise RuntimeError("PyTorch MPS is unavailable on this Mac")
                # The native context also identifies the GPU for torch_mps-only runs.
                from metalmat import MetalMatmul

                start = time.perf_counter_ns()
                self.metal = self.stack.enter_context(MetalMatmul(args.library, args.device))
                self.metadata["backend_initialization_ms"]["metal"] = (time.perf_counter_ns() - start) / 1e6
                self.metadata["gpu"] = dict(self.metal.info)
                self._library("metal", args.library or ROOT / "build" / "metal_matmul.dylib", 1)
                shader = ROOT / "src" / "matmul.metal"
                with shader.open("rb") as file:
                    self.metadata["metal_shader"] = {
                        "path": str(shader), "sha256": hashlib.file_digest(file, "sha256").hexdigest()}
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *_):
        self.stack.close()

    def _torch(self):
        if self.torch is None:
            import torch

            self.torch = torch
            torch.set_num_threads(self.args.cpu_threads)
            self.metadata["torch"] = torch.__version__

    def _library(self, name: str, path: Path, abi: int):
        path = path.resolve()
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        self.metadata["native_libraries"][name] = {"path": str(path), "sha256": digest, "abi": abi}

    def initialize(self, supported: set[tuple[str, str, str]]):
        """Load only backends with at least one supported requested case."""
        import time

        names = {name for name, _, _ in supported}
        if names.intersection(NATIVE_CUDA):
            from gpumat import NativeMatmul

            start = time.perf_counter_ns()
            self.native = self.stack.enter_context(NativeMatmul(self.args.library, self.args.device))
            self.metadata["backend_initialization_ms"]["cuda"] = (time.perf_counter_ns() - start) / 1e6
            self._library("cuda", self.args.library or ROOT / "build" / "matmul.so", 3)
        if names.intersection(("cpu_neon", "accelerate")):
            from cpumat import CpuMatmul

            start = time.perf_counter_ns()
            path = self.args.library if self.args.target == "cpu" else None
            self.cpu = self.stack.enter_context(CpuMatmul(path, threads=self.args.cpu_threads))
            self.metadata["backend_initialization_ms"]["cpu"] = (time.perf_counter_ns() - start) / 1e6
            self.metadata["native_cpu"] = dict(self.cpu.info)
            self._library("cpu", path or ROOT / "build" / "cpumat.dylib", 1)
        if "triton" in names:
            try:
                import triton_matmul
            except ImportError as error:
                raise RuntimeError("Triton requested but unavailable; run `uv sync --extra triton` on Linux") from error
            self.triton = triton_matmul
            self.metadata["triton"] = importlib.metadata.version("triton")
        if "jax" in names:
            try:
                import jax
                import jax.numpy as jnp
            except ImportError as error:
                raise RuntimeError("JAX requested but unavailable; run `uv sync --extra jax` on Linux") from error
            self.jax, self.jnp = jax, jnp
            devices = jax.devices("gpu")
            if self.args.device >= len(devices):
                raise RuntimeError(f"JAX has no GPU device {self.args.device}")
            self.jax_device = devices[self.args.device]
            algorithms = {"fp32": "F32_F32_F32", "tf32": "TF32_TF32_F32", "fp16": "F16_F16_F32", "bf16": "BF16_BF16_F32"}
            for precision in self.args.precisions:
                if any(("jax", precision, mode) in supported for mode in self.args.modes):
                    algorithm = getattr(jax.lax.DotAlgorithmPreset, algorithms[precision])
                    self.jax_functions[precision] = jax.jit(lambda a, b, algorithm=algorithm:
                        jnp.matmul(a, b, precision=algorithm, preferred_element_type=jnp.float32))
            self.metadata.update(jax=jax.__version__, jaxlib=importlib.metadata.version("jaxlib"), jax_algorithms=algorithms)

    def unavailable_reason(self, backend: str) -> str | None:
        if backend in ("metal_tensor", "metal_tensor_morton") and self.metal is not None:
            if not self.metadata["gpu"]["tensor_ops"]:
                return "TensorOps require macOS 26 and Metal 4"
        return None

    def policy(self, backend: str, precision: str):
        return torch_precision(self.torch, precision) if backend in CUDA_BACKENDS and backend != "jax" else nullcontext()

    def prepare(self, backend: str, precision: str, a, b) -> PreparedCase:
        import numpy as np

        m, _ = a.shape
        n = b.shape[1]
        if backend == "numpy":
            out = np.empty((m, n), dtype=np.float32)
            return PreparedCase(lambda: np.matmul(a, b, out=out), noop, lambda: out,
                                lambda: a @ b, "cpu", {"output_reused": True, "blas": "numpy"})
        if backend in ("cpu_neon", "accelerate"):
            return self._native_case(self.cpu, backend, precision, a, b, "cpu")
        if backend == "torch_cpu":
            torch = self.torch
            left, right = torch.from_numpy(a), torch.from_numpy(b)
            out = torch.empty((m, n), dtype=torch.float32)
            return PreparedCase(lambda: torch.mm(left, right, out=out), noop, lambda: out.numpy(),
                                lambda: torch.mm(torch.from_numpy(a), torch.from_numpy(b)).numpy(), "cpu",
                                {"output_reused": True, "threads_requested": self.args.cpu_threads})
        if backend in NATIVE_METAL:
            return self._native_case(self.metal, backend, precision, a, b, "metal")
        if backend == "torch_mps":
            torch = self.torch
            device = torch.device("mps")
            left, right = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
            out = torch.empty((m, n), dtype=torch.float32, device=device)
            def end_to_end():
                return torch.mm(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)).cpu().numpy()
            return PreparedCase(lambda: torch.mm(left, right, out=out), torch.mps.synchronize,
                                lambda: out.cpu().numpy(), end_to_end, "metal",
                                {"output_reused": True, "fast_math": False})
        if backend == "jax":
            jax, jnp = self.jax, self.jnp
            dtype = getattr(jnp, INPUT_DTYPES[precision])
            left = jax.device_put(a, self.jax_device).astype(dtype)
            right = jax.device_put(b, self.jax_device).astype(dtype)
            operation = self.jax_functions[precision]
            result = [None]
            def run():
                result[0] = operation(left, right)
            def synchronize():
                if result[0] is not None:
                    result[0].block_until_ready()
            def end_to_end():
                return np.asarray(operation(jax.device_put(a, self.jax_device).astype(dtype),
                                            jax.device_put(b, self.jax_device).astype(dtype)))
            return PreparedCase(run, synchronize, lambda: np.asarray(result[0]), end_to_end, "cuda",
                                {"algorithm": self.metadata["jax_algorithms"][precision]})
        return self._cuda_case(backend, precision, a, b)

    def _native_case(self, native, backend, precision, a, b, execution_device):
        plan = native.prepare(backend, a, b, precision=precision)
        def end_to_end():
            with native.prepare(backend, a, b, precision=precision) as temporary:
                temporary.run()
                return temporary.numpy()
        timer = None
        if execution_device == "metal":
            timer = lambda warmup, repeats: [seconds * 1e3 for seconds in plan.device_samples(warmup, repeats, DEVICE_BATCH)]
        return PreparedCase(plan.run, plan.synchronize, plan.numpy, end_to_end,
                            execution_device, dict(plan.config), timer, plan.close)

    def _cuda_case(self, backend, precision, a, b):
        torch = self.torch
        dtype = getattr(torch, INPUT_DTYPES[precision])
        left, right = torch.from_numpy(a).to(device=self.device, dtype=dtype), torch.from_numpy(b).to(device=self.device, dtype=dtype)
        out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=self.device)
        plan = None
        if backend == "torch":
            run = (lambda: torch.mm(left, right, out_dtype=torch.float32, out=out)) if precision in ("fp16", "bf16") else (lambda: torch.mm(left, right, out=out))
            config = {"reduced_precision_reduction": False}
        elif backend == "triton":
            plan = self.triton.prepare(left, right, out, precision)
            run = plan.run
            config = dict(plan.config)
        else:
            run = self.native.prepare(backend, left, right, out, precision=precision)
            config = self.native.configuration(backend, precision)
        def end_to_end():
            ta, tb = torch.from_numpy(a).to(device=self.device, dtype=dtype), torch.from_numpy(b).to(device=self.device, dtype=dtype)
            result = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=self.device)
            if backend == "triton":
                plan.bind(ta, tb, result)()
            elif backend == "torch":
                if precision in ("fp16", "bf16"):
                    torch.mm(ta, tb, out_dtype=torch.float32, out=result)
                else:
                    torch.mm(ta, tb, out=result)
            else:
                self.native.prepare(backend, ta, tb, result, precision=precision)()
            return result.cpu().numpy()
        timer = lambda warmup, repeats: measure_cuda_device(run, torch, self.stream, warmup, repeats)
        return PreparedCase(run, self.stream.synchronize, lambda: out.cpu().numpy(), end_to_end,
                            "cuda", config, timer)
