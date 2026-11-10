"""Prepared Apple Silicon CPU GEMM plans.

Inputs must remain unchanged for the lifetime of a plan. NEON packs B at
preparation and reads A on each run; Accelerate borrows both original inputs.
Output storage is reused per plan; NEON workers are reused across plans in the
context. A NumPy readback is an owned copy and includes its copy cost.
"""

import ctypes
from pathlib import Path
import platform
import threading
import weakref

import numpy as np

from precision import validate_host_operands


_IMPLEMENTATIONS = {"cpu_neon": 0, "accelerate": 1}
_ERROR_SIZE = 1024
_INT_MAX = 2**31 - 1


class _Config(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in
                ("bm", "bn", "bk", "mr", "nr", "requested_threads", "actual_threads", "packed_b")]


def _error(operation, buffer):
    detail = buffer.value.decode("utf-8", errors="replace") or "unknown native error"
    return RuntimeError(f"{operation} failed: {detail}")


class CpuMatmul:
    """Own CPU plans; closing the context also closes every live plan."""

    def __init__(self, library: Path | str | None = None, threads=1):
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= _INT_MAX:
            raise ValueError("threads must be a positive C int")
        path = Path(library) if library is not None else Path(__file__).resolve().parent.parent / "build" / "cpumat.dylib"
        if not path.is_file():
            raise FileNotFoundError(f"CPU matmul library not found at {path}. Run `make cpu` or pass a custom library path.")
        try:
            lib = ctypes.CDLL(str(path.resolve()))
        except OSError as exc:
            raise OSError(f"Could not load CPU matmul library at {path}: {exc}. Rebuild with `make cpu`.") from exc
        try:
            version = lib.cpumat_abi_version
        except AttributeError as exc:
            raise RuntimeError(f"CPU matmul library at {path} has no ABI version; rebuild with `make cpu`.") from exc
        version.argtypes = []
        version.restype = ctypes.c_int
        actual_version = version()
        if actual_version != 1:
            raise RuntimeError(f"CPU matmul library at {path} has ABI {actual_version}, expected 1; rebuild with `make cpu`.")
        try:
            lib.cpumat_create.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
            lib.cpumat_create.restype = ctypes.c_void_p
            lib.cpumat_destroy_context.argtypes = [ctypes.c_void_p]
            lib.cpumat_destroy_context.restype = None
            lib.cpumat_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_char_p, ctypes.c_size_t]
            lib.cpumat_prepare.restype = ctypes.c_void_p
            lib.cpumat_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.cpumat_run.restype = ctypes.c_int
            lib.cpumat_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Config), ctypes.c_char_p, ctypes.c_size_t]
            lib.cpumat_config.restype = ctypes.c_int
            lib.cpumat_output.argtypes = [ctypes.c_void_p]
            lib.cpumat_output.restype = ctypes.POINTER(ctypes.c_float)
            lib.cpumat_destroy.argtypes = [ctypes.c_void_p]
            lib.cpumat_destroy.restype = None
        except AttributeError as exc:
            raise RuntimeError(f"CPU matmul library at {path} is missing ABI 1 symbols; rebuild with `make cpu`.") from exc
        self._lib = lib
        self._threads = threads
        self._lock = threading.RLock()
        self._plans = weakref.WeakSet()
        self._closed = False
        self._context = None
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        self._context = lib.cpumat_create(threads, error, len(error))
        if not self._context:
            raise _error("cpumat_create", error)
        self.info = {"backend": "cpu", "architecture": platform.machine(), "requested_threads": threads,
                     "library": str(path.resolve()), "abi_version": actual_version}

    def _check_open(self):
        if self._closed:
            raise RuntimeError("CpuMatmul context is closed")

    def prepare(self, implementation, a, b, precision="fp32"):
        """Create a reusable plan; A and B must stay unchanged until it closes."""
        with self._lock:
            self._check_open()
            if implementation not in _IMPLEMENTATIONS:
                raise ValueError(f"Unknown CPU implementation: {implementation}")
            m, k, n = validate_host_operands(a, b, precision)
            error = ctypes.create_string_buffer(_ERROR_SIZE)
            handle = self._lib.cpumat_prepare(self._context, _IMPLEMENTATIONS[implementation], m, k, n,
                                               a.ctypes.data, b.ctypes.data,
                                               error, len(error))
            if not handle:
                raise _error("cpumat_prepare", error)
            plan = None
            try:
                plan = _Plan(self, handle, implementation, a, b, m, n)
                self._plans.add(plan)
            except BaseException:
                if plan is None:
                    self._lib.cpumat_destroy(handle)
                else:
                    plan.close()
                raise
            return plan

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                for plan in list(self._plans):
                    plan.close()
                self._plans.clear()
                if self._context:
                    self._lib.cpumat_destroy_context(self._context)
                    self._context = None

    def __enter__(self):
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, "_lock"):
                self.close()
        except Exception:
            pass


class _Plan:
    def __init__(self, owner, handle, implementation, a, b, m, n):
        config = _Config()
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        if owner._lib.cpumat_config(handle, ctypes.byref(config), error, len(error)):
            raise _error("cpumat_config", error)
        self._owner = owner
        self._a, self._b = a, b
        self._shape = (m, n)
        self._lock = threading.RLock()
        neon = implementation == "cpu_neon"
        self.config = {
            "implementation": implementation, "precision": "fp32",
            "tile": [config.bm, config.bn, config.bk] if neon else None,
            "microtile": [config.mr, config.nr] if neon else None,
            "packing": "B at prepare" if config.packed_b else "none",
            "requested_threads": config.requested_threads,
            "actual_threads": config.actual_threads if neon else None,
        }
        self._handle = handle
        self._error = error

    def _check_open(self):
        if self._handle is None:
            raise RuntimeError("CPU matmul plan is closed")

    def run(self):
        with self._lock:
            self._check_open()
            error = self._error
            if self._owner._lib.cpumat_run(self._handle, error, len(error)):
                raise _error("cpumat_run", error)

    def synchronize(self):
        with self._lock:
            self._check_open()

    def numpy(self):
        """Return an owned copy after a successful run."""
        with self._lock:
            self._check_open()
            pointer = self._owner._lib.cpumat_output(self._handle)
            if not pointer:
                raise RuntimeError("CPU matmul output is unavailable until a successful run")
            return np.ctypeslib.as_array(pointer, shape=self._shape).copy(order="C")

    def close(self):
        with self._lock:
            if self._handle is not None:
                handle, self._handle = self._handle, None
                self._owner._lib.cpumat_destroy(handle)
                self._a = self._b = None

    def __enter__(self):
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, "_lock") and hasattr(self, "_handle"):
                self.close()
        except Exception:
            pass
