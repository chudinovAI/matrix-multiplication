import ctypes
import json
from pathlib import Path

import torch
from precision import PRECISION_IDS, unsupported_reason, validate_operands


_IMPLEMENTATIONS = {"tiled": 0, "register": 1, "cublas": 2, "warp": 3,
                    "warp_large": 4, "pipeline": 5, "wmma": 6}
_ERROR_SIZE = 1024
_INT_MAX = 2**31 - 1


class NativeMatmul:
    """Own a native matmul context for one CUDA device."""

    def __init__(self, library: str | Path | None = None, device: int = 0):
        if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device <= _INT_MAX:
            raise ValueError("device must be a non-negative CUDA device index")
        path = Path(library) if library is not None else Path(__file__).resolve().parent.parent / "build" / "matmul.so"
        if not path.is_file():
            raise FileNotFoundError(
                f"Native matmul library not found at {path}. Run `make cuda` or pass a custom library path."
            )
        try:
            self._library = ctypes.CDLL(str(path.resolve()))
        except OSError as exc:
            raise OSError(f"Could not load native matmul library at {path}: {exc}") from exc

        try:
            abi_version = self._library.matmul_abi_version
        except AttributeError as exc:
            raise RuntimeError(
                f"Native matmul library at {path} has no ABI version; rebuild it with `make cuda`."
            ) from exc
        abi_version.argtypes = []
        abi_version.restype = ctypes.c_int
        version = abi_version()
        if version != 3:
            raise RuntimeError(
                f"Native matmul library at {path} has ABI version {version}, expected 3; "
                "rebuild it with `make cuda`."
            )

        self._library.matmul_create.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        self._library.matmul_create.restype = ctypes.c_void_p
        self._library.matmul_destroy.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self._library.matmul_destroy.restype = ctypes.c_int
        self._library.matmul_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self._library.matmul_launch.restype = ctypes.c_int
        self._library.matmul_kernel_info.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self._library.matmul_kernel_info.restype = ctypes.c_int
        self._device = device
        self._context = None
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        context = self._library.matmul_create(device, error, len(error))
        if not context:
            raise RuntimeError(self._error_message("matmul_create", error))
        self._context = context
        self._capability = torch.cuda.get_device_capability(device)

    @staticmethod
    def _error_message(operation, error):
        detail = error.value.decode("utf-8", errors="replace")
        return f"{operation} failed: {detail or 'unknown native error'}"

    def close(self):
        """Destroy the context once; prepared operations cannot be used afterward."""
        context = self._context
        if context is not None:
            self._context = None
            error = ctypes.create_string_buffer(_ERROR_SIZE)
            if self._library.matmul_destroy(context, error, len(error)) != 0:
                raise RuntimeError(self._error_message("matmul_destroy", error))

    def __enter__(self):
        if self._context is None:
            raise RuntimeError("NativeMatmul context is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            if getattr(self, "_context", None) is not None:
                self.close()
        except Exception:
            pass

    def configuration(self, implementation: str, precision: str = "fp32") -> dict:
        """Return the native kernel's actual launch geometry and staging policy."""
        if self._context is None:
            raise RuntimeError("NativeMatmul context is closed")
        if not isinstance(implementation, str) or implementation not in _IMPLEMENTATIONS:
            raise ValueError(f"unknown implementation: {implementation!r}")
        if not isinstance(precision, str) or precision not in PRECISION_IDS:
            raise ValueError(f"unknown precision: {precision!r}")
        reason = unsupported_reason(implementation, precision, self._capability)
        if reason is not None:
            raise ValueError(reason)
        result = ctypes.create_string_buffer(_ERROR_SIZE)
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        if self._library.matmul_kernel_info(_IMPLEMENTATIONS[implementation],
                                            PRECISION_IDS[precision], result, len(result),
                                            error, len(error)) != 0:
            raise RuntimeError(self._error_message("matmul_kernel_info", error))
        return json.loads(result.value)

    def prepare(self, implementation: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor,
                *, precision: str = "fp32"):
        """Prepare an asynchronous operation on the current CUDA stream."""
        if self._context is None:
            raise RuntimeError("NativeMatmul context is closed")
        if not isinstance(implementation, str) or implementation not in _IMPLEMENTATIONS:
            raise ValueError(f"unknown implementation: {implementation!r}")
        m, k, n = validate_operands(a, b, out, precision, self._device)
        reason = unsupported_reason(implementation, precision, self._capability)
        if reason is not None:
            raise ValueError(reason)

        a_ptr = ctypes.c_void_p(a.data_ptr())
        b_ptr = ctypes.c_void_p(b.data_ptr())
        out_ptr = ctypes.c_void_p(out.data_ptr())
        stream = torch.cuda.current_stream(self._device)
        stream_ptr = ctypes.c_void_p(stream.cuda_stream)
        # Keep the caching allocator from reusing storage while this stream uses it.
        for tensor in (a, b, out):
            tensor.record_stream(stream)
        implementation_id = _IMPLEMENTATIONS[implementation]
        precision_id = PRECISION_IDS[precision]
        launch = self._library.matmul_launch
        error = ctypes.create_string_buffer(_ERROR_SIZE)

        def run():
            # Captures retain all matrix storage and the stream for the lifetime of run.
            _ = a, b, stream
            context = self._context
            if context is None:
                raise RuntimeError("NativeMatmul context is closed")
            if launch(context, implementation_id, precision_id, a_ptr, b_ptr, out_ptr, m, k, n,
                      stream_ptr, error, len(error)) != 0:
                raise RuntimeError(self._error_message("matmul_launch", error))
            return out

        return run
