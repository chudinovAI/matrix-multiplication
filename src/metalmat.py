"""Prepared, resident Metal and MPS matrix multiplication plans."""

import ctypes
import json
import os
from pathlib import Path
import threading

import numpy as np

from precision import BACKEND_PRECISIONS, validate_host_operands

_ROOT = Path(__file__).resolve().parent.parent
_IMPLEMENTATIONS = {"metal_naive": 0, "metal_tiled": 1, "metal_simd": 2, "mps": 3,
                    "metal_tensor": 4, "metal_tensor_morton": 5}
_ERROR_SIZE = 4096
_JSON_SIZE = 16384
_INT_MAX = 2**31 - 1


def _bind(lib):
    """Bind ABI 1 only after checking its version."""
    try:
        version = lib.metal_abi_version
    except AttributeError as exc:
        raise RuntimeError("Metal library has no ABI version; rebuild with `make metal`.") from exc
    version.argtypes = []
    version.restype = ctypes.c_int
    actual = version()
    if actual != 1:
        raise RuntimeError(f"Metal library ABI {actual} is incompatible with ABI 1; rebuild with `make metal`.")
    p, i, z, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p
    signatures = {
        "metal_create": ([i, s, p, z], p),
        "metal_destroy": ([p], None),
        "metal_info": ([p, p, z, p, z], i),
        "metal_prepare": ([p, i, i, p, p, i, i, i, p, z], p),
        "metal_config": ([p, p, z, p, z], i),
        "metal_run": ([p, p, z], i),
        "metal_synchronize": ([p, p, z], i),
        "metal_copy_output": ([p, p, p, z], i),
        "metal_samples": ([p, i, i, i, p, p, z], i),
        "metal_plan_destroy": ([p, p, z], i),
    }
    try:
        for name, (args, result) in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, result
    except AttributeError as exc:
        raise RuntimeError("Metal library is missing ABI 1 functions; rebuild with `make metal`.") from exc


def _error(operation, error):
    message = error.value.decode("utf-8", errors="replace") or "unknown native error"
    return RuntimeError(f"{operation} failed: {message}")


def _call(lib, operation, *args):
    error = ctypes.create_string_buffer(_ERROR_SIZE)
    result = getattr(lib, operation)(*args, error, len(error))
    if result != 0:
        raise _error(operation, error)


def _json(lib, operation, handle):
    output = ctypes.create_string_buffer(_JSON_SIZE)
    _call(lib, operation, handle, output, len(output))
    return json.loads(output.value)


def _count(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _INT_MAX:
        raise ValueError(f"{name} must be an integer between {minimum} and {_INT_MAX}")


class MetalMatmul:
    """Compile Metal shaders once and prepare plans with shared Metal buffers."""

    def __init__(self, library: Path | str | None = None, device=0):
        _count("device", device, 0)
        self._context = None
        self._lock = threading.RLock()
        path = Path(library) if library is not None else _ROOT / "build" / "metal_matmul.dylib"
        if not path.is_file():
            raise FileNotFoundError(f"Metal library not found at {path}; run `make metal` or pass a library path.")
        try:
            self._lib = ctypes.CDLL(str(path.resolve()))
        except OSError as exc:
            raise OSError(f"Could not load Metal library at {path}; run `make metal`: {exc}") from exc
        _bind(self._lib)
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        source = os.fsencode(_ROOT / "src" / "matmul.metal")
        context = self._lib.metal_create(device, source, error, len(error))
        if not context:
            raise _error("metal_create", error)
        self._context = context

    def _require_open(self):
        if self._context is None:
            raise RuntimeError("MetalMatmul context is closed")

    @property
    def info(self):
        with self._lock:
            self._require_open()
            return _json(self._lib, "metal_info", self._context)

    def prepare(self, implementation, a, b, precision="fp32"):
        if implementation not in _IMPLEMENTATIONS:
            raise ValueError(f"Unknown Metal implementation: {implementation}")
        m, k, n = validate_host_operands(a, b, precision, BACKEND_PRECISIONS[implementation])
        with self._lock:
            self._require_open()
            error = ctypes.create_string_buffer(_ERROR_SIZE)
            input_type = {"fp32": 0, "fp16": 1, "bf16": 2}[precision]
            handle = self._lib.metal_prepare(self._context, _IMPLEMENTATIONS[implementation], input_type,
                                             a.ctypes.data, b.ctypes.data, m, n, k, error, len(error))
            if not handle:
                raise _error("metal_prepare", error)
            return MetalPlan(self._lib, handle, (m, n))

    def close(self):
        with self._lock:
            if self._context is not None:
                handle, self._context = self._context, None
                self._lib.metal_destroy(handle)

    def __enter__(self):
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MetalPlan:
    """A reusable asynchronous operation; owns buffers independently of its context."""

    def __init__(self, lib, handle, shape):
        self._lib, self._handle, self._shape = lib, handle, shape
        self._lock = threading.RLock()
        self._has_output = False

    def _require_open(self):
        if self._handle is None:
            raise RuntimeError("Metal plan is closed")

    @property
    def config(self):
        with self._lock:
            self._require_open()
            return _json(self._lib, "metal_config", self._handle)

    def run(self):
        with self._lock:
            self._require_open()
            _call(self._lib, "metal_run", self._handle)
            self._has_output = True

    def synchronize(self):
        with self._lock:
            self._require_open()
            _call(self._lib, "metal_synchronize", self._handle)

    def numpy(self):
        with self._lock:
            self._require_open()
            if not self._has_output:
                raise RuntimeError("Metal plan has not run")
            result = np.empty(self._shape, dtype=np.float32)
            _call(self._lib, "metal_copy_output", self._handle, result.ctypes.data)
            return result

    def device_samples(self, warmup: int, repeats: int, batch: int = 10):
        _count("warmup", warmup, 0)
        _count("repeats", repeats, 1)
        _count("batch", batch, 1)
        with self._lock:
            self._require_open()
            samples = np.empty(repeats, dtype=np.float64)
            _call(self._lib, "metal_samples", self._handle, warmup, repeats, batch, samples.ctypes.data)
            self._has_output = True
            return samples.tolist()

    def close(self):
        with self._lock:
            if self._handle is not None:
                handle, self._handle = self._handle, None
                _call(self._lib, "metal_plan_destroy", handle)

    def __enter__(self):
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
